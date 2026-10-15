"""Fresh four-method retraining on the user-supplied train70/valid20 boundary.

Only original source_subset=train images may supply training crops or donors.
The original valid20 is used only for validation/model selection and inference.
Existing 72/18 trained checkpoints are never accepted as initialization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import target_search as guards

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'original_split_retrain'
VARIANTS = ('centerline_baseline', 'e_softmask_geometry',
            'low_lr_context_paste', 'phase_style_exchange')
EXPECTED_INITIAL_SHA256 = 'dc9106bf16c414017e7365555e31e78c4a3bfe598267d4270c4ada3fbba98bb2'
INITIAL_SOURCE = ROOT / 'notes_recovered/common/initial_weights.pt'
SOURCE_FILES = ('src/original_split_retrain.py', 'src/experiment.py',
                'src/audit_original_split.py', 'src/target_search.py',
                'src/training_awake.py')
# Registration changes this process's helper registry, not the historical file.
guards.TRAINING_SCRIPTS.add(Path(__file__).name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    guards.save_json(path, value)


def require_identical(path, value):
    guards.require_identical(path, value)


def data_path(root, entry, field):
    relative = Path(entry[field].replace('\\', '/'))
    expected_parent = Path('data') / entry['source_subset'] / ('images' if field == 'image' else 'masks')
    suffix = '_image.png' if field == 'image' else '_mask.png'
    if relative.is_absolute() or relative.parent != expected_parent or relative.name != entry['id'] + suffix:
        raise RuntimeError(f'Unexpected original data path: {entry[field]}')
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to((root / expected_parent).resolve()):
        raise RuntimeError(f'Data path escapes its original subset: {relative}')
    return resolved


def original_boundary(root=ROOT):
    """Validate all original source files and return the unchanged file split."""
    manifest_path = root / 'records/data_manifest.json'
    manifest = read_json(manifest_path)
    if len(manifest) != 100 or len({entry['id'] for entry in manifest}) != 100:
        raise RuntimeError('Expected 100 unique manifest images')
    counts = {'train': 70, 'valid': 20, 'test': 10}
    split = {name: sorted(entry['id'] for entry in manifest if entry['source_subset'] == name)
             for name in counts}
    if {name: len(ids) for name, ids in split.items()} != counts:
        raise RuntimeError('Original source_subset counts must be exactly train70/valid20/test10')
    for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        if set(split[left]) & set(split[right]):
            raise RuntimeError('Original file IDs overlap between subsets')
    expected_paths = {(name, kind): set() for name in counts for kind in ('image', 'mask')}
    for entry in manifest:
        subset = entry['source_subset']
        if subset not in counts or bool(entry.get('mask')) != (subset != 'test'):
            raise RuntimeError('Expected labeled train70/valid20 and unlabeled test10')
        for field in ('image', 'mask'):
            if not entry.get(field):
                continue
            path = data_path(root, entry, field)
            if not path.is_file() or file_hash(path) != entry[field + '_sha256']:
                raise RuntimeError(f'Original data hash mismatch: {path}')
            expected_paths[(subset, field)].add(path)
    for (subset, field), expected in expected_paths.items():
        folder = root / 'data' / subset / ('images' if field == 'image' else 'masks')
        actual = {path.resolve() for path in folder.glob('*.png')}
        if actual != expected:
            raise RuntimeError(f'Manifest and original {subset}/{field} PNG files differ')
    train_hashes = {entry['image_sha256'] for entry in manifest if entry['source_subset'] == 'train'}
    valid_hashes = {entry['image_sha256'] for entry in manifest if entry['source_subset'] == 'valid'}
    identical_train_valid = sorted(train_hashes & valid_hashes)
    if identical_train_valid:
        raise RuntimeError('Identical image bytes occur in train and valid; preserve the split and resolve before training')
    groups = {name: {key.split('_', 1)[0] for key in ids} for name, ids in split.items()}
    overlaps = {left + '_' + right: sorted(groups[left] & groups[right])
                for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test'))}
    split.update({'method': 'Exact original manifest source_subset; no reshuffle or regrouping',
                  'source_subset_split_verified': True, 'requires_group_disjoint': False,
                  'historical_hsson_fold_verified': False,
                  'source_subset_manifest_sha256': file_hash(manifest_path),
                  'train_valid_image_sha256_overlap': identical_train_valid,
                  'group_counts': {name: len(value) for name, value in groups.items()},
                  'group_overlap': overlaps,
                  'group_overlap_policy': 'Report original overlaps only; never change source_subset boundaries',
                  'validation_usage': 'Original valid20 is evaluation/model selection only; never crops or donors'})
    return manifest, split


def corrected_config(base, split):
    expected = {'encoder': 'tu-tf_efficientnetv2_s.in21k_ft_in1k',
                'encoder_lr': 3e-5, 'decoder_lr': 3e-4, 'weight_decay': .05,
                'ema_decay': .999, 'native_crop': 384, 'input_crop': 768,
                'patches_per_epoch': 600, 'batch': 8, 'max_epochs': 100,
                'patience_epochs': 20, 'min_delta_pp': .02, 'recall_smooth': 1e-5,
                'center_after_final_transform': True, 'scheduler_per_step': True,
                'min_lr': 1e-7}
    if any(base.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Historical-method hyperparameters differ from the reviewed protocol')
    config = copy.deepcopy(base)
    config['confirmed_from_notes'] = [text for text in base.get('confirmed_from_notes', [])
                                     if 'train72 valid18' not in text]
    config['remaining_assumptions'] = [text for text in base.get('remaining_assumptions', [])
                                      if 'Original folds.csv' not in text and 'All five runs' not in text]
    config['remaining_assumptions'] += [
        'Historical hsson fold is unavailable. This run intentionally preserves original source_subset train70/valid20/test10.',
        'Four user-requested methods use max100/patience20/minDelta0.02; historical72/18 scores are not comparable.']
    config.update({'experiment_phase': 'original_source_split_correction_20261015',
                   'user_requested_sheet_rows': [122, 123, 124, 125],
                   'requested_variants': list(VARIANTS),
                   'source_subset_split_verified': True,
                   'historical_hsson_fold_verified': False,
                   'train_ids': split['train'], 'donor_ids': split['train'],
                   'valid_ids': split['valid'], 'test_ids': split['test'],
                   'validation_usage': 'Evaluation/model selection only; original valid20 is never training or donor data',
                   'baseline_initial_sha256': EXPECTED_INITIAL_SHA256,
                   'historical_reference_scores_comparable': False})
    return config


def prepare(root=ROOT):
    output = root / 'original_split_retrain'
    manifest, split = original_boundary(root)
    source_initial = root / 'notes_recovered/common/initial_weights.pt'
    if file_hash(source_initial) != EXPECTED_INITIAL_SHA256:
        raise RuntimeError('Baseline must use the approved untouched ImageNet-encoder/untrained-decoder initialization')
    base_path = root / 'configs/record_notes.json'
    config = corrected_config(read_json(base_path), split)
    require_identical(output / 'split.json', split)
    require_identical(output / 'config.json', config)
    initial = output / 'common/initial_weights.pt'
    initial.parent.mkdir(parents=True, exist_ok=True)
    if not initial.exists():
        with source_initial.open('rb') as source, initial.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    if file_hash(initial) != EXPECTED_INITIAL_SHA256:
        raise RuntimeError('Frozen fresh initialization copy differs; refusing any trained checkpoint')
    plan = {'version': 1, 'experiment_phase': 'original_source_split_correction_20261015',
            'user_requested_sheet_rows': [122, 123, 124, 125], 'variants': list(VARIANTS),
            'scope': 'Exactly four requested methods; E-protected illumination excluded',
            'config': config, 'config_sha256': file_hash(output / 'config.json'),
            'source_config_sha256': file_hash(base_path), 'split_file': str(output / 'split.json'),
            'split_sha256': file_hash(output / 'split.json'),
            'data_manifest_sha256': file_hash(root / 'records/data_manifest.json'),
            'source_sha256': {name: file_hash(root / name) for name in SOURCE_FILES},
            'initial_checkpoint': str(initial), 'initial_checkpoint_sha256': EXPECTED_INITIAL_SHA256,
            'initial_checkpoint_policy': 'Fresh approved initialization only; no prior trained best/last weights',
            'fine_tune_policy': 'Each of the three variants starts from this new train70-only baseline best',
            'source_subset_split_verified': True, 'historical_hsson_fold_verified': False,
            'selection_tile': 512, 'final_tta_tile': 768, 'early_stop_enabled': True,
            'counts': {'train': 70, 'valid': 20, 'test': 10},
            'group_overlap': split['group_overlap']}
    require_identical(output / 'plan.json', plan)
    require_identical(output / 'common/initial_provenance.json', {
        'source': str(source_initial), 'copy': str(initial), 'sha256': EXPECTED_INITIAL_SHA256,
        'purpose': 'ImageNet pretrained encoder plus untrained seed42 decoder; no segmentation training'})
    return config, split, plan


def load_original_data(manifest, core, root=ROOT):
    """Use core's same native targets while avoiding its legacy split loader."""
    data = {}
    for entry in manifest:
        with core.Image.open(data_path(root, entry, 'image')) as handle:
            image = core.np.array(handle.convert('RGB'))
        label = core.decode_mask(data_path(root, entry, 'mask')) if entry.get('mask') else None
        center = thin = None
        if label is not None:
            eutectic = label == 3
            center = core.ndi.binary_dilation(core.skeletonize(eutectic),
                structure=core.ndi.generate_binary_structure(2, 1), iterations=2) & eutectic
            thin = (core.ndi.distance_transform_edt(eutectic) <= 2) & eutectic
        data[entry['id']] = {'image': image, 'label': label, 'center': center,
                             'thin': thin, 'entry': entry}
    return data


def verify_dataset_boundary(core, data, split, config, variant):
    dataset = core.PatchDataset(data, split['train'], variant, config, 1)
    if dataset.ids != split['train'] or len(dataset.donors) != 70:
        raise RuntimeError('Training dataset or donor count differs from original train70')
    for key, (image, label) in zip(dataset.ids, dataset.donors):
        entry = data[key]
        if (entry['entry']['source_subset'] != 'train' or image is not entry['image']
                or label is not entry['label'] or key in split['valid'] or key in split['test']):
            raise RuntimeError('Validation/test image reached training crops or donors')


def audit_completed(output, split_file, variant, result, plan):
    import audit_original_split
    audit_original_split.main(output / 'runs', split_file)
    checks = {item['variant']: item for item in read_json(output / 'runs/independent_audit.json')}
    check = checks.get(variant, {})
    value = check.get('independent_mIoU')
    if not (check.get('passed') is True and check.get('valid_masks') == 20
            and check.get('test_masks') == 10 and check.get('split_sha256') == plan['split_sha256']
            and isinstance(value, (int, float)) and math.isfinite(value)
            and abs(value - result['validation']['mIoU']) < 1e-10):
        raise RuntimeError(f'Original-split independent audit failed: {variant}')
    return check


def expected_run_config(config, variant, initial):
    value = {**config, 'variant': variant, 'initial_checkpoint': str(initial),
             'selection_tile': 512, 'early_stop_enabled': True,
             'status': 'reconstructed, original split/code unavailable'}
    if variant == 'low_lr_context_paste':
        value.update(encoder_lr=1e-6, decoder_lr=1e-5)
    return value


def shared_new_baseline(output, plan):
    run = output / 'runs/centerline_baseline'
    result = read_json(run / 'result.json')
    if not result.get('source_subset_split_verified') or result.get('split_sha256') != plan['split_sha256']:
        raise RuntimeError('Fine-tuning requires this new original train70 baseline')
    value = {'checkpoint': str(run / 'best.pt'), 'sha256': file_hash(run / 'best.pt'),
             'baseline_result_sha256': file_hash(run / 'result.json'),
             'split_sha256': plan['split_sha256'], 'baseline_initial_sha256': EXPECTED_INITIAL_SHA256}
    require_identical(output / 'common/fine_tune_initialization.json', value)
    return Path(value['checkpoint']), value['sha256']


def run_all(config, split, plan, resume, root=ROOT):
    import torch
    import experiment as core
    from training_awake import keep_system_awake
    output = root / 'original_split_retrain'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for original-split retraining')
    manifest = read_json(root / 'records/data_manifest.json')
    data = load_original_data(manifest, core, root)
    results = {}
    with keep_system_awake(output / 'power_guard.json'):
        for variant in VARIANTS:
            guards.assert_no_other_training()
            if prepare(root)[2] != plan:
                raise RuntimeError('Frozen inputs changed before training')
            initial, initial_sha = ((Path(plan['initial_checkpoint']), EXPECTED_INITIAL_SHA256)
                if variant == 'centerline_baseline' else shared_new_baseline(output, plan))
            if file_hash(initial) != initial_sha:
                raise RuntimeError('Initialization hash mismatch')
            run = output / 'runs' / variant
            provenance = {'variant': variant, 'plan_sha256': object_hash(plan),
                          'source_sha256': plan['source_sha256'], 'split_sha256': plan['split_sha256'],
                          'data_manifest_sha256': plan['data_manifest_sha256'],
                          'initial_checkpoint': str(initial), 'initial_checkpoint_sha256': initial_sha,
                          'source_subset_split_verified': True,
                          'initialization_lineage': 'fresh approved initial -> new train70 baseline -> independent variant'}
            provenance_file = run / 'provenance.json'
            existed = provenance_file.exists()
            if run.exists() and not existed and any(run.iterdir()):
                raise RuntimeError(f'Unrecognized existing run; do not reuse previous weights: {run}')
            require_identical(provenance_file, provenance)
            boundary = {key: config[key] for key in ('train_ids', 'donor_ids', 'valid_ids', 'test_ids')}
            boundary.update(split_sha256=plan['split_sha256'],
                            data_manifest_sha256=plan['data_manifest_sha256'], source_subset_split_verified=True)
            require_identical(run / 'data_boundary.json', boundary)
            expected_config = expected_run_config(config, variant, initial)
            saved_config = run / 'config.json'
            if saved_config.exists() and read_json(saved_config) != expected_config:
                raise RuntimeError(f'Run config mismatch; refusing checkpoint resume: {variant}')
            started = any((run / name).exists() for name in ('config.json', 'progress.json', 'last.pt', 'best.pt', 'result.json'))
            if started and not resume:
                raise RuntimeError(f'Run already started; use --resume, never overwrite: {variant}')
            if not (run / 'result.json').exists():
                if started and not (run / 'last.pt').exists():
                    raise RuntimeError(f'No resumable checkpoint in partial run; preserve it for review: {variant}')
                verify_dataset_boundary(core, data, split, config, variant)
                save_json(output / 'status.json', {'state': 'running', 'variant': variant,
                    'completed_variants': list(results), 'split_sha256': plan['split_sha256']})
                core.run_training(variant, config, data, split, init=initial,
                                  resume=started and resume, run_directory=run,
                                  selection_tile=512, early_stop=True)
            if prepare(root)[2] != plan or file_hash(initial) != initial_sha:
                raise RuntimeError('Frozen inputs changed during training; result requires review')
            if not saved_config.exists() or read_json(saved_config) != expected_config:
                raise RuntimeError('Actual training config differs from frozen original-split protocol')
            result = read_json(run / 'result.json')
            metadata = {'experiment_phase': 'original_source_split_correction_20261015',
                        'source_subset_split_verified': True, 'historical_hsson_fold_verified': False,
                        'historical_reference_scores_comparable': False,
                        'split_file': str(output / 'split.json'), 'split_sha256': plan['split_sha256'],
                        'source_data_counts': {'train': 70, 'valid': 20, 'test': 10},
                        'plan_sha256': object_hash(plan), 'provenance_sha256': object_hash(provenance),
                        'initial_checkpoint_sha256': initial_sha, 'source_sha256': plan['source_sha256'],
                        'validation_usage': config['validation_usage']}
            if result.get('variant') != variant:
                raise RuntimeError('Result variant mismatch')
            for key, value in metadata.items():
                if key in result and result[key] != value:
                    raise RuntimeError('Result provenance mismatch: ' + key)
            result.update(metadata)
            save_json(run / 'result.json', result)
            check = audit_completed(output, output / 'split.json', variant, result, plan)
            save_json(run / 'verification.json', check)
            results[variant] = result
            save_json(output / 'runs/summary.json', results)
            save_json(output / 'status.json', {'state': 'run_completed', 'variant': variant,
                'completed_variants': list(results), 'mIoU': check['independent_mIoU']})
            print(f'ORIGINAL_SPLIT_RESULT {variant} mIoU={check["independent_mIoU"]:.9f}', flush=True)
        save_json(output / 'complete.json', {'complete': True, 'variants': list(VARIANTS),
            'measured_results': results, 'plan_sha256': object_hash(plan),
            'source_subset_split_verified': True, 'counts': {'train': 70, 'valid': 20, 'test': 10}})
        save_json(output / 'status.json', {'state': 'complete', 'completed_variants': list(results)})
    print('ORIGINAL_SPLIT_FOUR_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only:
        guards.assert_no_other_training()
    with guards.exclusive_lock(OUTPUT / 'active.lock'):
        if not args.prepare_only:
            guards.assert_no_other_training()
        config, split, plan = prepare()
        if args.prepare_only:
            print(json.dumps({'prepared': True, 'training_started': False,
                'plan_sha256': object_hash(plan), 'counts': plan['counts'],
                'variants': list(VARIANTS), 'group_overlap': split['group_overlap']}), flush=True)
            return
        run_all(config, split, plan, args.resume)


if __name__ == '__main__':
    main()
