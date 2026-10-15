"""Independently score exported RGB masks and verify file/hash/split integrity."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
COLORS=((153,127,76),(76,178,76),(25,76,153),(204,204,204))

def decode(path):
    rgb=np.asarray(Image.open(path).convert('RGB')); labels=np.full(rgb.shape[:2],-1,np.int16)
    for i,c in enumerate(COLORS): labels[np.all(rgb==c,axis=2)]=i
    if np.any(labels<0): raise ValueError(f'Unexpected color: {path}')
    return labels

def main(runs_dir=ROOT/'runs', split_file=ROOT/'configs'/'split_fold0.json'):
    runs_dir=Path(runs_dir)
    if not runs_dir.is_absolute(): runs_dir=ROOT/runs_dir
    split_file=Path(split_file)
    if not split_file.is_absolute(): split_file=ROOT/split_file
    split=json.loads(split_file.read_text(encoding='utf-8'))
    split_sha=hashlib.sha256(split_file.read_bytes()).hexdigest()
    for part in ('train','valid','test'):
        assert len(split[part])==len(set(split[part])), f'Duplicate IDs in {part}'
    assert not set(split['train'])&set(split['valid'])
    assert not set(split['train'])&set(split['test'])
    assert not set(split['valid'])&set(split['test'])
    group_check=None
    if split.get('requires_group_disjoint') or split_file.name=='split_group_fold0.json':
        train_groups={key.split('_',1)[0] for key in split['train']}
        valid_groups={key.split('_',1)[0] for key in split['valid']}
        assert (len(split['train']),len(split['valid']),len(split['test']))==(72,18,10)
        assert len(valid_groups)==14, 'Recovered protocol requires 14 validation A groups'
        assert not train_groups & valid_groups, 'Training and validation A groups overlap'
        group_check={'train_images':72,'valid_images':18,'valid_A_groups':14,'train_valid_group_overlap':[]}
    manifest=json.loads((ROOT/'records'/'data_manifest.json').read_text(encoding='utf-8'))
    entries={e['id']:e for e in manifest}
    for e in manifest:
        assert hashlib.sha256((ROOT/e['image']).read_bytes()).hexdigest()==e['image_sha256']
        if e['mask']: assert hashlib.sha256((ROOT/e['mask']).read_bytes()).hexdigest()==e['mask_sha256']
    checks=[]; rows=[]
    runs_dir.mkdir(parents=True,exist_ok=True)
    for run in sorted(runs_dir.iterdir()):
        if not (run/'result.json').exists(): continue
        result=json.loads((run/'result.json').read_text(encoding='utf-8'))
        if result.get('source_notes_recovered'):
            assert group_check is not None, 'Corrected result must use --split-file configs/split_group_fold0.json'
        reported_ids=[row['id'] for row in result['validation'].get('per_image',[])]
        assert len(reported_ids)==len(split['valid']) and set(reported_ids)==set(split['valid']), 'Result validation IDs differ from selected split'
        if result.get('split_file'):
            recorded_split=Path(result['split_file'])
            if not recorded_split.is_absolute(): recorded_split=ROOT/recorded_split
            assert hashlib.sha256(recorded_split.read_bytes()).hexdigest()==split_sha, 'Result split differs from selected split'
        scores=[]; class_scores=[]
        for key in split['valid']:
            target=decode(ROOT/entries[key]['mask']); pred=decode(run/'valid_masks_d4'/(key+'_mask.png'))
            assert target.shape==pred.shape
            ious=[]
            for c in range(4):
                intersection=np.count_nonzero((target==c)&(pred==c)); union=np.count_nonzero((target==c)|(pred==c))
                ious.append(intersection/union*100 if union else np.nan)
            scores.append(float(np.nanmean(ious))); class_scores.append(ious)
        miou=float(np.mean(scores)); err=abs(miou-result['validation']['mIoU']); assert err<1e-10
        test_files=list((run/'test_masks_d4').glob('*_mask.png')); assert len(test_files)==10
        for key in split['test']:
            pred=decode(run/'test_masks_d4'/(key+'_mask.png'))
            assert pred.shape==(entries[key]['height'],entries[key]['width'])
        class_means=np.nanmean(class_scores,axis=0)
        assert np.allclose(class_means,result['validation']['class_iou'],rtol=0,atol=1e-10)
        checks.append({'variant':run.name,'independent_mIoU':miou,'difference':err,'valid_masks':len(scores),'test_masks':len(test_files),'passed':True,'split_file':str(split_file),'split_sha256':split_sha,'group_invariants':group_check,'source_hashes_verified':len(manifest)})
        rows.append([run.name,miou,*class_means.tolist(),result['epochs'],result['best_epoch'],result['train_minutes'],result['original_record_mIoU'],miou-result['original_record_mIoU'],'reconstructed; original split not verified; historical difference is unpaired, not a controlled comparison'])
    (runs_dir/'independent_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    with (runs_dir/'measured_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['variant','mIoU','Al','Primary Si','Al3Ni','Eutectic Si','epochs','best_epoch','train_minutes','historical_mIoU','difference_pp','status']); writer.writerows(rows)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'runs')
    parser.add_argument('--split-file',type=Path,default=ROOT/'configs'/'split_fold0.json')
    args=parser.parse_args()
    main(args.runs_dir,args.split_file)
