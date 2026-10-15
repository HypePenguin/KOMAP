"""CPU-only boundary and independent-score regressions using synthetic files.

All fixtures live below this script's workspace directory. No real data, model,
checkpoint, GPU or original project file is modified.
"""
import contextlib
import copy
import io
import json
import tempfile
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import audit_original_split as audit

TEMP_ROOT=(Path(__file__).resolve().parent/'boundary_test_tmp').resolve()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def manifest_fixture():
    # A0 intentionally overlaps train/valid: the supplied boundary must survive.
    return [{'id':f'A0_{subset}_{index:02d}','source_subset':subset}
            for subset,count in audit.COUNTS.items() for index in range(count)]


def split_fixture(manifest):
    return {**{subset:sorted(row['id'] for row in manifest if row['source_subset']==subset) for subset in audit.COUNTS},
            'source_subset_split_verified':True,'requires_group_disjoint':False}


def boundary_fixture(split,split_sha='split',manifest_sha='manifest'):
    return {'train_ids':split['train'].copy(),'donor_ids':split['train'].copy(),
            'valid_ids':split['valid'].copy(),'test_ids':split['test'].copy(),
            'source_subset_split_verified':True,'split_sha256':split_sha,'data_manifest_sha256':manifest_sha}


def full_fixture(root):
    manifest=manifest_fixture(); split=split_fixture(manifest); palette=np.array(audit.COLORS,dtype=np.uint8)
    run=root/'original_split_retrain/runs/centerline_baseline'; run.mkdir(parents=True)
    records=[]
    for number,row in enumerate(manifest):
        subset=row['source_subset']; index=int(row['id'].rsplit('_',1)[1])
        image=root/'data'/subset/'images'/(row['id']+'_image.png'); image.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.full((2,2,3),[number,number//2,17],dtype=np.uint8)).save(image)
        row.update(image=str(image.relative_to(root)),width=2,height=2,image_sha256=audit.file_hash(image),mask=None)
        if subset!='test':
            target=np.zeros((2,2),np.uint8)
            if subset=='valid':
                kind=index%3
                if kind==0:
                    target=np.array([[0,0],[1,1]],np.uint8); pred=np.array([[0,0],[0,1]],np.uint8)
                    per_class=[200/3,50,None,None]; score=(200/3+50)/2
                elif kind==1:
                    target=np.full((2,2),2,np.uint8); pred=target.copy()
                    per_class=[None,None,100,None]; score=100
                else:
                    target=np.full((2,2),3,np.uint8); pred=np.array([[3,3],[2,2]],np.uint8)
                    per_class=[None,None,0,50]; score=25
                predicted=run/'valid_masks_d4'/(row['id']+'_mask.png'); predicted.parent.mkdir(exist_ok=True)
                Image.fromarray(palette[pred]).save(predicted)
                records.append({'id':row['id'],'mIoU':score,'class_iou':per_class})
            mask=root/'data'/subset/'masks'/(row['id']+'_mask.png'); mask.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(palette[target]).save(mask)
            row.update(mask=str(mask.relative_to(root)),mask_sha256=audit.file_hash(mask))
        else:
            predicted=run/'test_masks_d4'/(row['id']+'_mask.png'); predicted.parent.mkdir(exist_ok=True)
            Image.fromarray(palette[np.array([[0,1],[2,3]],np.uint8)]).save(predicted)
    manifest_path=root/'records/data_manifest.json'; write_json(manifest_path,manifest)
    manifest_sha=audit.file_hash(manifest_path); split['source_subset_manifest_sha256']=manifest_sha
    split_file=root/'original_split_retrain/split.json'; write_json(split_file,split); split_sha=audit.file_hash(split_file)
    boundary=boundary_fixture(split,split_sha,manifest_sha)
    write_json(run/'data_boundary.json',boundary); write_json(run/'config.json',copy.deepcopy(boundary))
    # Hand-derived fixture metric: 7 images58.333%, 7 images100%, 6 images25%.
    result={'variant':run.name,'epochs':25,'best_epoch':5,'train_minutes':10.0,
            'source_subset_split_verified':True,'split_sha256':split_sha,'split_file':str(split_file),
            'validation':{'mIoU':(7*((200/3+50)/2)+7*100+6*25)/20,
                          'class_iou':[200/3,50,700/13,50],'per_image':records},
            'test':{'count':10,'mIoU':None}}
    write_json(run/'result.json',result)
    return manifest,split,run


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): TEMP_ROOT.mkdir(exist_ok=True)

    def test_original_boundary_is_exact_even_when_A_groups_overlap(self):
        manifest=manifest_fixture(); split=split_fixture(manifest)
        report=audit.validate_original_split(manifest,split)
        self.assertEqual(report['train_valid_A_group_overlap'],['A0'])
        self.assertEqual(len(split['train']),70); self.assertEqual(len(split['valid']),20)

    def test_valid_ID_in_arbitrary_training_list_is_rejected(self):
        manifest=manifest_fixture(); split=split_fixture(manifest)
        mixed=split['train'][:-1]+[split['valid'][0]]
        with self.assertRaisesRegex(ValueError,'non-training source IDs forbidden'):
            audit.assert_training_ids(mixed,manifest,'training list')
        with self.assertRaisesRegex(ValueError,'non-training source IDs forbidden'):
            audit.assert_training_ids([split['valid'][0]],manifest,'minibatch',require_exact=False)

    def test_valid_ID_in_saved_training_config_or_donor_list_is_rejected(self):
        manifest=manifest_fixture(); split=split_fixture(manifest); boundary=boundary_fixture(split)
        for field in ('train_ids','donor_ids'):
            with self.subTest(field=field):
                config=copy.deepcopy(boundary); config[field][-1]=split['valid'][0]
                with self.assertRaisesRegex(ValueError,'non-training source IDs forbidden'):
                    audit.validate_run_boundary(config,boundary,split,manifest,'split','manifest')

    def test_swapping_original_train_and_valid_preserving_counts_is_rejected(self):
        manifest=manifest_fixture(); split=split_fixture(manifest)
        split['train'][0],split['valid'][0]=split['valid'][0],split['train'][0]
        with self.assertRaisesRegex(ValueError,'Original source_subset boundary changed'):
            audit.validate_original_split(manifest,split)

    def test_independent_mask_metric_and_all190_source_hashes(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve(); self.assertTrue(root.is_relative_to(TEMP_ROOT))
            manifest,split,run=full_fixture(root); original_result_sha=audit.file_hash(run/'result.json')
            with contextlib.redirect_stdout(io.StringIO()): rows=audit.main(project_root=root)
            self.assertEqual(len(rows),1); row=rows[0]
            self.assertTrue(row['passed']); self.assertEqual(row['valid_masks'],20); self.assertEqual(row['test_masks'],10)
            self.assertEqual(row['source_image_hashes_verified'],100); self.assertEqual(row['source_mask_hashes_verified'],90)
            self.assertAlmostEqual(row['independent_mIoU'],62.91666666666667,places=10)
            self.assertEqual(row['group_overlap'],['A0'])
            self.assertEqual(audit.file_hash(run/'result.json'),original_result_sha)
            self.assertTrue((run.parent/'source_boundary_audit.json').exists())

    def test_source_tampering_fails_before_reporting(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve(); self.assertTrue(root.is_relative_to(TEMP_ROOT))
            manifest,split,run=full_fixture(root)
            path=root/manifest[0]['image']; path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Source image SHA mismatch'):
                audit.main(project_root=root)
            self.assertFalse((run.parent/'independent_audit.json').exists())

    def test_wrong_reported_score_fails_independent_rescoring(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve(); self.assertTrue(root.is_relative_to(TEMP_ROOT))
            manifest,split,run=full_fixture(root); result=audit.read_json(run/'result.json')
            result['validation']['mIoU']=99; write_json(run/'result.json',result)
            with self.assertRaisesRegex(ValueError,'Overall mIoU differs'):
                audit.main(project_root=root)

    def test_exact_image_duplicate_across_original_subsets_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root=Path(folder).resolve(); self.assertTrue(root.is_relative_to(TEMP_ROOT))
            manifest,split,run=full_fixture(root)
            training=next(row for row in manifest if row['source_subset']=='train')
            validation=next(row for row in manifest if row['source_subset']=='valid')
            (root/validation['image']).write_bytes((root/training['image']).read_bytes())
            validation['image_sha256']=training['image_sha256']
            with self.assertRaisesRegex(ValueError,'Exact source image SHA overlap'):
                audit.verify_source_files(root,manifest)

    def test_runner_checks_actual_crop_and_donor_constructor(self):
        # Imports build no model and allocate no CUDA tensors. The production
        # Dataset constructor only stores object references used by its sampler.
        sys.path.insert(0,str(Path('C:/Users/SSAFY/Desktop/KOMAP/src')))
        import experiment as core
        import original_split_retrain as runner
        manifest=manifest_fixture(); split=split_fixture(manifest)
        data={row['id']:{'image':object(),'label':object(),'entry':row} for row in manifest}
        runner.verify_dataset_boundary(core,data,split,{},'phase_style_exchange')
        mixed=copy.deepcopy(split); mixed['train'][-1]=split['valid'][0]
        with self.assertRaisesRegex(RuntimeError,'Validation/test image reached'):
            runner.verify_dataset_boundary(core,data,mixed,{},'phase_style_exchange')
        class LeakingDonorDataset(core.PatchDataset):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                valid=data[split['valid'][0]]; self.donors[-1]=(valid['image'],valid['label'])
        with self.assertRaisesRegex(RuntimeError,'Validation/test image reached'):
            runner.verify_dataset_boundary(types.SimpleNamespace(PatchDataset=LeakingDonorDataset),data,split,{},'phase_style_exchange')


if __name__=='__main__': unittest.main(verbosity=2)
