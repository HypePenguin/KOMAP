"""Read-only file evidence for corrected experiments and the frozen v1 runs."""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['centerline_baseline', 'e_softmask_geometry', 'low_lr_context_paste',
         'phase_style_exchange', 'e_protected_illumination']

def inspect_run(name,run,stage='notes_recovered'):
    row = {'variant': name, 'stage':stage,'state': 'pending'}
    try:
        if (run / 'result.json').exists():
            result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
            row.update(state='complete', epoch=result['epochs'],
                       mIoU=result['validation']['mIoU'], best_epoch=result['best_epoch'])
        elif stage=='initial_v1':
            row.update(state='stopped' if name=='low_lr_context_paste' else 'canceled',
                       reason='Superseded by EMA/WD/centerline/group-split corrections')
            if name=='low_lr_context_paste': row['last_checkpoint_epoch']=20
        elif stage=='canceled_supplement':
            row.update(state='canceled',reason='Canceled before training; notes_recovered replaces queue')
        elif (run / 'progress.json').exists():
            progress = json.loads((run / 'progress.json').read_text(encoding='utf-8'))
            age=max(0,time.time()-(run/'progress.json').stat().st_mtime)
            row.update(state='recent_progress' if age<600 else 'stale_progress',
                       epoch=progress.get('epoch_in_progress',progress.get('epoch')),
                       completed_epoch=progress.get('epoch_completed',progress.get('epoch')),
                       best=progress.get('best_no_tta'),batch=progress.get('batch_in_epoch'),
                       batches=progress.get('batches_per_epoch'),updated_seconds_ago=round(age))
        elif (run / 'config.json').exists():
            row['state']='prepared'
    except (OSError, json.JSONDecodeError):
        row['state'] = 'updating'
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-legacy',action='store_true')
    parser.add_argument('--include-supplement',action='store_true',help='Deprecated alias for --include-legacy')
    args=parser.parse_args()
    status=[inspect_run(n,ROOT/'notes_recovered'/'runs'/n) for n in NAMES]
    if args.include_legacy or args.include_supplement:
        status += [inspect_run(n,ROOT/'runs'/n,'initial_v1') for n in NAMES]
        status += [inspect_run(n,ROOT/'recorded_protocol'/'runs'/n,'canceled_supplement') for n in NAMES[:2]]
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__': main()
