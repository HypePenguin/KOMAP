"""Reproduce five strategies using the original settings recovered in cell notes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import experiment as core

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notes_recovered'


def prepare():
    entries=sorted([e for e in json.loads((ROOT/'records/data_manifest.json').read_text(encoding='utf-8')) if e['mask']],key=lambda e:e['id'])
    groups=np.array([e['id'].split('_')[0] for e in entries])
    unique,inverse=np.unique(groups,return_inverse=True)
    counts=np.bincount(inverse)
    # Reconstructed older GroupKFold tie order; no model scores were consulted.
    order=np.argsort(counts.astype(object),kind='quicksort')[::-1]
    loads=np.zeros(5,dtype=np.int64); assignment=np.zeros(len(unique),dtype=np.int64)
    for index in order:
        fold=int(np.argmin(loads)); assignment[index]=fold; loads[fold]+=counts[index]
    fold_by_id={e['id']:int(assignment[inverse[i]]) for i,e in enumerate(entries)}
    train=[e['id'] for e in entries if fold_by_id[e['id']]!=0]
    valid=[e['id'] for e in entries if fold_by_id[e['id']]==0]
    train_groups={x.split('_')[0] for x in train}; valid_groups={x.split('_')[0] for x in valid}
    assert (len(train),len(valid),len(valid_groups))==(72,18,14)
    assert not train_groups&valid_groups
    assert {'A167','A51','A63','A87'}<=train_groups
    manifest=json.loads((ROOT/'records/data_manifest.json').read_text(encoding='utf-8'))
    split={'train':train,'valid':valid,'test':[e['id'] for e in manifest if e['source_subset']=='test'],
           'group_folds':fold_by_id,'method':'RECONSTRUCTED GroupKFold5: alphabetical A groups, legacy generic quicksort tie ordering',
           'requires_group_disjoint':True,'original_split_verified':False,
           'recorded_original_split_sha256':'8fd0caa0b6bf0241173c089817ce92922d8ef66cee319348bb518b8d007dfd21',
           'selection_basis':'Matches72/18, valid14 A groups, no train/valid A overlap, all four recorded train/test overlaps. No model scores used.',
           'caveat':'Original folds.csv is unavailable; native GT distance<=2 fraction74.350292 differs from recorded73.1, whose exact thin-mask definition is unavailable. Original file-level split is NOT verified.'}
    split_file=ROOT/'configs/split_group_fold0.json'
    if split_file.exists():
        existing=json.loads(split_file.read_text(encoding='utf-8'))
        if existing!=split: raise RuntimeError('Frozen split differs; do not replace a split after model evaluation')
    else: core.save_json(split_file,split)
    config={'encoder':'tu-tf_efficientnetv2_s.in21k_ft_in1k','mean':[.5,.5,.5],'std':[.5,.5,.5],
            'native_crop':384,'input_crop':768,'patches_per_epoch':600,'batch':8,'workers':2,
            'inference_batch':8,'max_epochs':100,'patience_epochs':20,'min_delta_pp':.02,
            'encoder_lr':3e-5,'decoder_lr':3e-4,'weight_decay':.05,'ema_decay':.999,
            'recall_smooth':1e-5,'center_after_final_transform':True,'scheduler_per_step':True,'min_lr':1e-7,
            'reference_mIoU':{'centerline_baseline':78.9309711410583,'e_softmask_geometry':79.00779879382462,
                'low_lr_context_paste':78.94274586106262,'phase_style_exchange':78.92420324957915,'e_protected_illumination':78.92165697045854},
            'source_notes_rows':[1,10,22,28],'expected_parameters':22093908,
            'expected_original_initial_sha256':'401b67f68607def4bcba0a0987983012f2733a1c9dc62243e08433483e5c996e',
            'confirmed_from_notes':['explicit encoder/pretrained tag','AdamW decoder3e-4 encoder multiplier0.1','weight decay0.05',
                'EMA0.999','final-transformed GT skeleton + cross dilation twice intersect E','aux recall smooth1e-5 per-present-E-patch mean',
                'primary checkpoint selection tile512 and secondary tile768','batch8 samples600 seed42 workers2 threads4',
                '22,093,908 model parameters','train72 valid18 valid14 A groups and train/valid A group disjoint'],
            'remaining_assumptions':['Original folds.csv not recovered; documented group split candidate frozen before any model evaluation.',
                'Cosine per-update schedule with minimum1e-7 inferred; exact schedule not specified in recovered config notes.',
                'Fine-tune selection tile512 inherited from earlier common protocol; final Oct7 fine-tune selection tile not explicit.',
                'Original source code/software versions and detailed augmentation gates/intensities unavailable; reconstruct values retained.',
                'Inference batch8, uniform overlap probability mean and FP16 AMP retained; exact original inference batching/blend unknown.',
                'All five runs follow the final Oct7 max100/patience20/minDelta0.02 protocol, including the baseline retrain in row98.']}
    core.save_json(ROOT/'configs/record_notes.json',config)
    core.save_json(OUTPUT/'plan.json',{'state':'prepared','variants':core.VARIANTS,'config':config,
                 'split_file':str(split_file),'split_sha256':hashlib.sha256(split_file.read_bytes()).hexdigest(),
                 'original_split_verified':False,'source':'Google Sheet native cell notes; original records retained unchanged'})
    return config,split


def smoke(config,split):
    core.seed_everything(42)
    model=core.make_model(config,pretrained=True).cuda().to(memory_format=torch.channels_last)
    assert sum(p.numel() for p in model.parameters())==config['expected_parameters']
    data,_=core.load_data()
    dataset=core.PatchDataset(data,split['train'],'e_softmask_geometry',config,1)
    loader=torch.utils.data.DataLoader(dataset,batch_size=config['batch'])
    x,target,center,aug=next(iter(loader))
    assert torch.allclose(target.sum(1),torch.ones_like(center),atol=1e-6)
    assert not torch.any((center>0)&(target.argmax(1)!=3))
    x=x.cuda().to(memory_format=torch.channels_last); target=target.cuda(); center=center.cuda()
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':config['encoder_lr']},
        {'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':config['decoder_lr']}],weight_decay=.05)
    ema=copy.deepcopy(model).eval()
    scaler=torch.amp.GradScaler('cuda')
    with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
    loss=core.loss_fn(logits,target,center,'e_softmask_geometry',1e-5)
    assert torch.isfinite(loss); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
    core.update_ema(ema,model,.999)
    assert all(torch.isfinite(p).all() for p in ema.parameters())
    core.save_json(OUTPUT/'smoke.json',{'passed':True,'batch':8,'input':list(x.shape),'loss':float(loss.detach()),
        'center_is_subset_of_final_E':True,'parameters':sum(p.numel() for p in model.parameters()),
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'original_split_verified':False})
    print('NOTES_RECOVERED_SMOKE_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    config,split=prepare()
    if args.prepare_only: print('NOTES_RECOVERED_PREPARED',flush=True); return
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    if args.smoke: smoke(config,split); return
    initial=OUTPUT/'common/initial_weights.pt'; initial.parent.mkdir(parents=True,exist_ok=True)
    if not initial.exists():
        core.seed_everything(42); model=core.make_model(config,pretrained=True)
        assert sum(p.numel() for p in model.parameters())==config['expected_parameters']
        torch.save({'model':model.state_dict()},initial); del model
    initial_sha=hashlib.sha256(initial.read_bytes()).hexdigest()
    core.save_json(OUTPUT/'common/initial_metadata.json',{'sha256':initial_sha,'recorded_sha256':config['expected_original_initial_sha256'],
        'original_initial_file_verified':initial_sha==config['expected_original_initial_sha256'],'parameters':config['expected_parameters']})
    data,_=core.load_data(); results={}; baseline=OUTPUT/'runs/centerline_baseline/best.pt'
    for variant in core.VARIANTS:
        run=OUTPUT/'runs'/variant
        if args.resume and (run/'result.json').exists(): result=json.loads((run/'result.json').read_text(encoding='utf-8'))
        else:
            _,result=core.run_training(variant,config,data,split,init=initial if variant=='centerline_baseline' else baseline,
                resume=args.resume,run_directory=run,selection_tile=512,early_stop=True)
            result['split_file']=str(ROOT/'configs/split_group_fold0.json'); result['group_split_counts']={'train':72,'valid':18,'valid_A_groups':14}
            result['original_split_verified']=False; result['source_notes_recovered']=True
            core.save_json(run/'result.json',result)
        results[variant]=result; core.save_json(OUTPUT/'runs/summary.json',results)
    core.save_json(OUTPUT/'complete.json',{'complete':True,'measured_results':results,'original_split_verified':False})
    print('NOTES_RECOVERED_ALL_COMPLETE',flush=True)


if __name__=='__main__': main()
