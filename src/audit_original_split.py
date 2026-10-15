"""Independent audit of original supplied train70 / valid20 / test10 runs.

Only writes audit JSON/CSV under original_split_retrain/runs. It never changes
source files, the supplied partition, run results or checkpoint files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
COUNTS={'train':70,'valid':20,'test':10}
COLORS=((153,127,76),(76,178,76),(25,76,153),(204,204,204))


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition: raise ValueError(message)


def source_ids(manifest):
    ids=[entry['id'] for entry in manifest]
    check(len(ids)==100 and len(set(ids))==100,'Manifest must contain 100 unique supplied file IDs')
    check(all(entry['source_subset'] in COUNTS for entry in manifest),'Unknown manifest source_subset')
    groups={name:{entry['id'] for entry in manifest if entry['source_subset']==name} for name in COUNTS}
    check({name:len(values) for name,values in groups.items()}==COUNTS,'Supplied source subsets must be exactly70/20/10')
    return groups


def validate_original_split(manifest,split):
    expected=source_ids(manifest)
    for name,ids in expected.items():
        actual=split.get(name)
        check(isinstance(actual,list) and len(actual)==len(set(actual)),'Missing/duplicate split IDs: '+name)
        check(set(actual)==ids,'Original source_subset boundary changed: '+name)
    check(split.get('source_subset_split_verified') is True,'Split must explicitly identify verified source_subset boundaries')
    check(split.get('requires_group_disjoint') is False,'Do not reshuffle original boundaries to remove A-group overlaps')
    group_sets={name:{key.split('_',1)[0] for key in ids} for name,ids in expected.items()}
    return {'source_subset_counts':COUNTS.copy(),'source_subset_split_verified':True,
            'train_valid_id_overlap':sorted(expected['train']&expected['valid']),
            'train_valid_A_group_overlap':sorted(group_sets['train']&group_sets['valid']),
            'train_test_A_group_overlap':sorted(group_sets['train']&group_sets['test']),
            'valid_test_A_group_overlap':sorted(group_sets['valid']&group_sets['test']),
            'A_group_counts':{name:len(ids) for name,ids in group_sets.items()},
            'group_overlap_policy':'Report supplied overlaps without changing source_subset membership'}


def assert_training_ids(ids,manifest,role='training',require_exact=True):
    allowed=source_ids(manifest)['train']
    check(isinstance(ids,(list,tuple)) and len(ids)==len(set(ids)),role+': missing or duplicate IDs')
    forbidden=set(ids)-allowed
    check(not forbidden,role+': non-training source IDs forbidden: '+', '.join(sorted(forbidden)))
    if require_exact: check(set(ids)==allowed,role+': must use exactly the supplied training70 IDs')


def validate_run_boundary(config,boundary,split,manifest,split_sha256,manifest_sha256):
    validate_original_split(manifest,split)
    check(boundary.get('source_subset_split_verified') is True,'Missing verified per-run source boundary')
    check(boundary.get('split_sha256')==split_sha256,'Per-run boundary split SHA mismatch')
    check(boundary.get('data_manifest_sha256')==manifest_sha256,'Per-run boundary manifest SHA mismatch')
    for label,payload in [('boundary',boundary),('config',config)]:
        assert_training_ids(payload.get('train_ids'),manifest,label+'.train_ids')
        assert_training_ids(payload.get('donor_ids'),manifest,label+'.donor_ids')
        for part in ('valid','test'):
            values=payload.get(part+'_ids')
            check(isinstance(values,list) and len(values)==len(set(values)) and set(values)==set(split[part]),
                  label+'.'+part+'_ids differ from original source subset')
        if 'split_sha256' in payload: check(payload['split_sha256']==split_sha256,label+' split SHA mismatch')
        if 'data_manifest_sha256' in payload: check(payload['data_manifest_sha256']==manifest_sha256,label+' manifest SHA mismatch')


def contained_path(root,value):
    path=Path(value)
    if not path.is_absolute(): path=root/path
    path=path.resolve()
    check(path.is_relative_to(root),'Manifest path escapes project: '+str(value))
    return path


def verify_source_files(root,manifest):
    actual_images={}; image_count=0; mask_count=0
    expected_paths={(subset,kind):set() for subset in COUNTS for kind in ('images','masks')}
    for entry in manifest:
        subset=entry['source_subset']; image_path=contained_path(root,entry['image'])
        check(image_path.parent==(root/'data'/subset/'images').resolve(),'Image path differs from source_subset: '+entry['id'])
        check(image_path.name==entry['id']+'_image.png','Image filename differs from manifest ID')
        expected_paths[(subset,'images')].add(image_path)
        digest=file_hash(image_path)
        check(digest==entry['image_sha256'],'Source image SHA mismatch: '+entry['id'])
        with Image.open(image_path) as image:
            check(image.size==(entry['width'],entry['height']),'Source image dimensions changed: '+entry['id'])
        actual_images[entry['id']]=digest; image_count+=1
        if subset=='test':
            check(entry.get('mask') is None,'Test source unexpectedly contains a ground-truth mask')
        else:
            check(bool(entry.get('mask')),'Labeled source mask is missing: '+entry['id'])
            mask_path=contained_path(root,entry['mask'])
            check(mask_path.parent==(root/'data'/subset/'masks').resolve(),'Mask path differs from source_subset: '+entry['id'])
            check(mask_path.name==entry['id']+'_mask.png','Mask filename differs from manifest ID')
            expected_paths[(subset,'masks')].add(mask_path)
            check(file_hash(mask_path)==entry['mask_sha256'],'Source mask SHA mismatch: '+entry['id'])
            with Image.open(mask_path) as mask:
                check(mask.size==(entry['width'],entry['height']),'Source mask dimensions changed: '+entry['id'])
            mask_count+=1
    check((image_count,mask_count)==(100,90),'Expected100 source images and90 source masks')
    for (subset,kind),expected in expected_paths.items():
        actual={path.resolve() for path in (root/'data'/subset/kind).glob('*.png')}
        check(actual==expected,'Source PNG files differ from manifest: '+subset+'/'+kind)
    ids=source_ids(manifest)
    overlaps=sorted({actual_images[key] for key in ids['train']}&{actual_images[key] for key in ids['valid']})
    check(not overlaps,'Exact source image SHA overlap between training and validation; refusing an independent-validation claim')
    return {'source_image_hashes_verified':image_count,'source_mask_hashes_verified':mask_count,
            'train_valid_image_sha256_overlap':overlaps}


def decode(path):
    with Image.open(path) as image: rgb=np.asarray(image.convert('RGB'))
    labels=np.full(rgb.shape[:2],-1,np.int16)
    for index,color in enumerate(COLORS): labels[np.all(rgb==color,axis=2)]=index
    check(not np.any(labels<0),'Unexpected prediction/GT palette color: '+str(path))
    return labels


def compare_numbers(actual,expected,label):
    left=np.array([np.nan if value is None else value for value in actual],dtype=float)
    right=np.array([np.nan if value is None else value for value in expected],dtype=float)
    check(left.shape==right.shape and np.allclose(left,right,rtol=0,atol=1e-10,equal_nan=True),label+' differs from independent mask metric')


def audit_run(run,root,split,manifest,split_file,split_sha,manifest_sha,source_report):
    result=read_json(run/'result.json'); config=read_json(run/'config.json'); boundary=read_json(run/'data_boundary.json')
    validate_run_boundary(config,boundary,split,manifest,split_sha,manifest_sha)
    check(result.get('variant')==run.name,'Result variant differs from run directory')
    check(result.get('source_subset_split_verified') is True,'Result does not certify original supplied boundaries')
    check(result.get('split_sha256')==split_sha,'Result split SHA mismatch')
    result_split=contained_path(root,result['split_file'])
    check(result_split==split_file and file_hash(result_split)==split_sha,'Result points to a different split')
    records=result['validation']['per_image']; ids=[row['id'] for row in records]
    check(len(ids)==20 and len(set(ids))==20 and set(ids)==set(split['valid']),'Result must evaluate exactly the original valid20')
    records={row['id']:row for row in records}; entries={row['id']:row for row in manifest}
    for part in ('valid','test'):
        expected={key+'_mask.png' for key in split[part]}
        actual={path.name for path in (run/(part+'_masks_d4')).glob('*.png')}
        check(actual==expected,part+' output masks have missing or extra filenames')
    scores=[]; class_scores=[]
    for key in split['valid']:
        target=decode(contained_path(root,entries[key]['mask'])); pred=decode(run/'valid_masks_d4'/(key+'_mask.png'))
        check(target.shape==pred.shape,'Validation prediction shape differs from GT: '+key)
        ious=[]
        for index in range(4):
            intersection=np.count_nonzero((target==index)&(pred==index)); union=np.count_nonzero((target==index)|(pred==index))
            ious.append(intersection/union*100 if union else np.nan)
        score=float(np.nanmean(ious)); compare_numbers([score],[records[key]['mIoU']],key+' mIoU')
        compare_numbers(ious,records[key]['class_iou'],key+' class IoUs')
        scores.append(score); class_scores.append(ious)
    class_array=np.asarray(class_scores)
    class_means=[float(np.nanmean(class_array[:,index])) if np.isfinite(class_array[:,index]).any() else None for index in range(4)]
    miou=float(np.mean(scores))
    compare_numbers([miou],[result['validation']['mIoU']],'Overall mIoU')
    compare_numbers(class_means,result['validation']['class_iou'],'Mean class IoUs')
    for key in split['test']:
        pred=decode(run/'test_masks_d4'/(key+'_mask.png'))
        check(pred.shape==(entries[key]['height'],entries[key]['width']),'Test output dimensions differ: '+key)
    check(result['test'].get('count')==10 and result['test'].get('mIoU') is None,'Unlabeled test must have10 outputs and no mIoU')
    return {'variant':run.name,'passed':True,'independent_mIoU':miou,'independent_class_iou':class_means,
            'difference':abs(miou-result['validation']['mIoU']),'valid_masks':20,'test_masks':10,
            'split_file':str(split_file),'split_sha256':split_sha,'data_manifest_sha256':manifest_sha,
            'result_sha256':file_hash(run/'result.json'),'config_sha256':file_hash(run/'config.json'),
            'data_boundary_sha256':file_hash(run/'data_boundary.json'),
            'source_subset_split_verified':True,'source_hashes_verified':100,
            **source_report,'train_ids_verified':70,'donor_ids_verified':70,
            'historical_score_comparison':'No direct difference is reported against past scores from a different partition'}


def main(runs_dir=None,split_file=None,project_root=None):
    root=Path(project_root or ROOT).resolve()
    runs_dir=Path(runs_dir or root/'original_split_retrain/runs')
    if not runs_dir.is_absolute(): runs_dir=root/runs_dir
    runs_dir=runs_dir.resolve()
    check(runs_dir==(root/'original_split_retrain/runs').resolve(),'Audit output must be the new original_split_retrain/runs directory')
    split_file=Path(split_file or root/'original_split_retrain/split.json')
    if not split_file.is_absolute(): split_file=root/split_file
    split_file=split_file.resolve()
    check(split_file==(root/'original_split_retrain/split.json').resolve(),'Audit requires the new original_split_retrain/split.json')
    manifest_path=root/'records/data_manifest.json'; manifest=read_json(manifest_path); split=read_json(split_file)
    split_sha=file_hash(split_file); manifest_sha=file_hash(manifest_path)
    boundary_report=validate_original_split(manifest,split)
    check(split.get('source_subset_manifest_sha256')==manifest_sha,'Split source manifest SHA mismatch')
    source_report=verify_source_files(root,manifest)
    boundary_report.update(source_report,split_sha256=split_sha,data_manifest_sha256=manifest_sha)
    check(runs_dir.exists(),'New runs directory does not exist')
    checks=[]; rows=[]
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or not (run/'result.json').exists(): continue
        audit=audit_run(run,root,split,manifest,split_file,split_sha,manifest_sha,source_report)
        audit['group_overlap']=boundary_report['train_valid_A_group_overlap']; checks.append(audit)
        result=read_json(run/'result.json')
        rows.append([run.name,audit['independent_mIoU'],*audit['independent_class_iou'],result['epochs'],result['best_epoch'],result['train_minutes'],'original supplied train70/valid20; test10 unlabeled; groups retained as supplied'])
    for name,value in [('independent_audit.json',checks),('source_boundary_audit.json',boundary_report)]:
        path=runs_dir/name; temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); temporary.replace(path)
    csv_path=runs_dir/'measured_results.csv'; temporary=csv_path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream); writer.writerow(['variant','mIoU','Al','Primary Si','Al3Ni','Eutectic Si','epochs','best_epoch','train_minutes','scope']); writer.writerows(rows)
    temporary.replace(csv_path)
    print(json.dumps(checks,ensure_ascii=False,indent=2)); return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir',type=Path); parser.add_argument('--split-file',type=Path)
    parser.add_argument('--project-root',type=Path)
    args=parser.parse_args(); main(args.runs_dir,args.split_file,args.project_root)
