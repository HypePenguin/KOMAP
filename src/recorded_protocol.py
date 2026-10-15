"""Queue the earlier recorded tile512-selection / full100ep protocol.

The initial five experiments are preserved. This comparison keeps their exact
data split, seed, architecture, crop scale and optimizer assumptions, changes
selection tile to the documented512, and follows the fixed100ep centerline
training recorded in rows22/28/49. Final inference remains tile768 / D4 TTA.
It then independently trains E soft-mask from that new baseline (rows88/96).
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RECOVERY=ROOT/'recorded_protocol'
INITIAL_VARIANTS=['centerline_baseline','e_softmask_geometry','low_lr_context_paste','phase_style_exchange','e_protected_illumination']


def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def initial_complete():
    path=ROOT/'runs'/'summary.json'
    if not path.exists():
        return False
    try:
        summary=json.loads(path.read_text(encoding='utf-8'))
        return all(name in summary and (ROOT/'runs'/name/'result.json').exists() for name in INITIAL_VARIANTS)
    except (OSError,json.JSONDecodeError):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for-initial',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs'/'experiment.json').read_text(encoding='utf-8'))
    config['reference_mIoU']['centerline_baseline']=78.93204859354823
    config['reconstruction_assumptions'] += [
        'Supplement follows earlier rows22/28/49: baseline ImageNet initialization, fixed100ep, selection tile512, final tile768 D4.',
        'Supplement geometry follows rows88/96 from the new baseline; selection tile512 inherited from earlier common protocol, exact fine-tune selection tile not explicitly stated.',
        'Same fixed reconstructed split as initial five; no validation split or metric changes; other unrecorded optimizer/architecture values retained.'
    ]
    plan={'created':datetime.now().astimezone().isoformat(), 'state':'queued' if args.wait_for_initial else 'prepared',
          'source_rows':[22,28,49,88,96], 'baseline_epochs':100, 'baseline_early_stop':False,
          'geometry_max_epochs':100, 'geometry_early_stop':True, 'selection_tile':512,
          'final_tile':768, 'final_tta':'D4', 'original_split_verified':False,
          'split_sha256':hashlib.sha256((ROOT/'configs'/'split_fold0.json').read_bytes()).hexdigest(),
          'config':config}
    save_json(RECOVERY/'plan.json',plan)
    print('RECORDED_PROTOCOL_PLAN',json.dumps({k:v for k,v in plan.items() if k!='config'},ensure_ascii=False),flush=True)
    if args.plan_only:
        return
    while args.wait_for_initial and not initial_complete():
        time.sleep(50)
    import torch
    import experiment as core
    plan['state']='running'; save_json(RECOVERY/'plan.json',plan)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    data,split=core.load_data()
    results={}
    baseline=RECOVERY/'runs'/'centerline_baseline'
    for variant in ['centerline_baseline','e_softmask_geometry']:
        run=RECOVERY/'runs'/variant
        if args.resume and (run/'result.json').exists():
            result=json.loads((run/'result.json').read_text(encoding='utf-8'))
        else:
            _,result=core.run_training(variant,config,data,split,
                init=None if variant=='centerline_baseline' else baseline/'best.pt',
                resume=args.resume,run_directory=run,selection_tile=512,
                early_stop=variant!='centerline_baseline')
        results[variant]=result
        core.save_json(RECOVERY/'runs'/'summary.json',results)
    plan['state']='complete'; save_json(RECOVERY/'plan.json',plan)
    print('RECORDED_PROTOCOL_COMPLETE',flush=True)


if __name__=='__main__':
    main()
