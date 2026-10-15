"""User-prioritized row125 phase style training: train70 + valid19, A170 excluded.

The one-image validation score is a reference only: the clean train70 baseline
previously used A170 for checkpoint selection, never training/backpropagation.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import time
import uuid

import original_split_retrain as original
import target_search as guards

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = 'priority_style89_20261015'
VARIANT = 'phase_style_exchange'
EXCLUDED_ID = 'A170_step4_sample3'
BASELINE_SHA256 = '6bf6809c18cb49d576e704fa0738d54e9e83fc0abc47a21e227ec796ee2054be'
FRESH_INITIAL_SHA256 = original.EXPECTED_INITIAL_SHA256
SOURCE_FILES = ('src/priority_style89.py', 'src/experiment.py',
                'src/original_split_retrain.py', 'src/target_search.py',
                'src/training_awake.py', 'src/audit_original_split.py',
                'src/audit_priority_style89.py', 'src/build_priority_style89_share.py',
                'src/share_model_priority.py')
VALIDATION_USAGE = ('A170_step4_sample3 only: evaluation/checkpoint selection, never training or donor. '
                    'Single-image reference score, not original valid20 performance or a new independent test; '
                    'A170 was previously used for clean train70 baseline checkpoint selection.')
guards.TRAINING_SCRIPTS.update({'priority_style89.py', 'original_split_retrain.py'})
read_json = guards.read_json
file_hash = guards.file_hash
save_json = guards.save_json
object_hash = guards.object_hash


def priority_split(manifest, original_split):
    if EXCLUDED_ID not in original_split['valid']:
        raise RuntimeError('User-excluded image must belong to original valid20')
    train = sorted(original_split['train'] + [x for x in original_split['valid'] if x != EXCLUDED_ID])
    split = {'train': train, 'valid': [EXCLUDED_ID], 'test': original_split['test'][:],
             'method': 'Explicit user request: original train70 plus original valid19; exclude exact A170 image',
             'excluded_id': EXCLUDED_ID, 'excluded_image': EXCLUDED_ID + '_image.png',
             'counts': {'train': 89, 'valid': 1, 'test': 10},
             'training_source_counts': {'train': 70, 'valid': 19, 'test': 0},
             'source_subset_manifest_sha256': original_split['source_subset_manifest_sha256'],
             'validation_usage': VALIDATION_USAGE,
             'original_valid20_independent_evaluation': False,
             'historical_hsson_fold_verified': False}
    validate_split(manifest, split)
    return split


def validate_split(manifest, split):
    by_id = {x['id']: x for x in manifest}
    if len(by_id) != 100 or len(manifest) != 100:
        raise RuntimeError('Expected 100 unique original source images')
    expected_train = sorted(x['id'] for x in manifest
                            if x['source_subset'] in ('train', 'valid') and x['id'] != EXCLUDED_ID)
    expected_test = sorted(x['id'] for x in manifest if x['source_subset'] == 'test')
    if (len(expected_train) != 89 or split['train'] != expected_train
            or split['valid'] != [EXCLUDED_ID] or split['test'] != expected_test):
        raise RuntimeError('Exact train89/holdoutA170/test10 membership required')
    if EXCLUDED_ID in split['train'] or len(set(split['train'])) != 89:
        raise RuntimeError('Excluded image or duplicates reached training')
    if by_id[EXCLUDED_ID]['source_subset'] != 'valid':
        raise RuntimeError('Excluded original image source differs')
    if any(by_id[x]['image_sha256'] == by_id[EXCLUDED_ID]['image_sha256'] for x in split['train']):
        raise RuntimeError('An identical copy of the excluded image reached training')
    if set(split['train']) & set(split['test']):
        raise RuntimeError('Test image reached training')


def verified_baseline(root, manifest, original_split):
    base = root / 'original_split_retrain'
    run = base / 'runs/centerline_baseline'
    checkpoint = run / 'best.pt'
    result = read_json(run / 'result.json')
    audit = read_json(run / 'verification.json')
    provenance = read_json(run / 'provenance.json')
    boundary = read_json(run / 'data_boundary.json')
    plan = read_json(base / 'plan.json')
    if file_hash(checkpoint) != BASELINE_SHA256:
        raise RuntimeError('Only the frozen clean train70 baseline best is allowed')
    if read_json(base / 'split.json') != original_split:
        raise RuntimeError('Clean baseline split does not match original train70/valid20/test10')
    split_sha = file_hash(base / 'split.json')
    manifest_sha = file_hash(root / 'records/data_manifest.json')
    if not (audit.get('passed') is True and audit.get('valid_masks') == 20
            and audit.get('test_masks') == 10 and audit.get('train_ids_verified') == 70
            and audit.get('donor_ids_verified') == 70
            and audit.get('result_sha256') == file_hash(run / 'result.json')
            and audit.get('config_sha256') == file_hash(run / 'config.json')
            and audit.get('data_boundary_sha256') == file_hash(run / 'data_boundary.json')
            and audit.get('split_sha256') == split_sha
            and audit.get('data_manifest_sha256') == manifest_sha
            and result.get('initial_checkpoint_sha256') == FRESH_INITIAL_SHA256
            and provenance.get('initial_checkpoint_sha256') == FRESH_INITIAL_SHA256
            and provenance.get('split_sha256') == split_sha
            and provenance.get('plan_sha256') == object_hash(plan)
            and result.get('source_subset_split_verified') is True):
        raise RuntimeError('Clean baseline provenance or independent verification failed')
    for key, split_key in (('train_ids', 'train'), ('donor_ids', 'train'),
                           ('valid_ids', 'valid'), ('test_ids', 'test')):
        if boundary.get(key) != original_split[split_key]:
            raise RuntimeError('Baseline training or donor boundary differs')
    for name, expected in plan['source_sha256'].items():
        if file_hash(root / name) != expected:
            raise RuntimeError('Clean baseline source changed: ' + name)
    return {'checkpoint': str(checkpoint), 'sha256': BASELINE_SHA256,
            'baseline_result_sha256': file_hash(run / 'result.json'),
            'baseline_verification_sha256': file_hash(run / 'verification.json'),
            'baseline_provenance_sha256': file_hash(run / 'provenance.json'),
            'baseline_data_boundary_sha256': file_hash(run / 'data_boundary.json'),
            'baseline_split_sha256': split_sha, 'baseline_initial_sha256': FRESH_INITIAL_SHA256,
            'baseline_training_ids': original_split['train'],
            'baseline_validation_ids': original_split['valid'],
            'excluded_image_used_in_baseline_training': False,
            'excluded_image_used_in_baseline_selection': True}


def priority_config(base, split):
    expected = {'encoder_lr': 3e-5, 'decoder_lr': 3e-4, 'weight_decay': .05,
                'ema_decay': .999, 'native_crop': 384, 'input_crop': 768,
                'patches_per_epoch': 600, 'batch': 8, 'max_epochs': 100,
                'patience_epochs': 20, 'min_delta_pp': .02,
                'center_after_final_transform': True, 'scheduler_per_step': True}
    if any(base.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Row125 method hyperparameters differ from reviewed protocol')
    config = copy.deepcopy(base)
    config.update({'experiment_phase': OUTPUT_NAME, 'user_requested_sheet_rows': [125],
        'requested_variants': [VARIANT], 'train_ids': split['train'], 'donor_ids': split['train'],
        'valid_ids': split['valid'], 'test_ids': split['test'], 'validation_usage': VALIDATION_USAGE,
        'source_subset_split_verified': False, 'requested_boundary_verified': True,
        'source_data_counts': {'train': 89, 'valid': 1, 'test': 10},
        'training_source_counts': {'train': 70, 'valid': 19, 'test': 0},
        'excluded_id': EXCLUDED_ID, 'initial_checkpoint_sha256': BASELINE_SHA256,
        'validation_scope': 'single_image_reference',
        'baseline_holdout_used_for_checkpoint_selection': True,
        'original_valid20_independent_evaluation': False,
        'historical_reference_scores_comparable': False})
    config['remaining_assumptions'] = [s for s in config.get('remaining_assumptions', [])
                                      if 'train70/valid20' not in s and 'Four user-requested' not in s]
    config['remaining_assumptions'].append(VALIDATION_USAGE)
    return config


def prepare(root=ROOT):
    root = Path(root)
    output = root / OUTPUT_NAME
    manifest, original_split = original.original_boundary(root)
    split = priority_split(manifest, original_split)
    baseline = verified_baseline(root, manifest, original_split)
    config = priority_config(read_json(root / 'original_split_retrain/config.json'), split)
    guards.require_identical(output / 'split.json', split)
    config.update(split_sha256=file_hash(output / 'split.json'),
                  data_manifest_sha256=file_hash(root / 'records/data_manifest.json'))
    guards.require_identical(output / 'config.json', config)
    initial = output / 'common/initial_weights.pt'
    initial.parent.mkdir(parents=True, exist_ok=True)
    if not initial.exists():
        with Path(baseline['checkpoint']).open('rb') as source, initial.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=1024*1024)
            destination.flush()
            os.fsync(destination.fileno())
    if file_hash(initial) != BASELINE_SHA256:
        raise RuntimeError('Frozen initial checkpoint copy differs from clean baseline')
    guards.require_identical(output / 'common/initial_provenance.json', baseline)
    plan = {'version': 1, 'experiment_phase': OUTPUT_NAME, 'variant': VARIANT,
        'counts': split['counts'], 'training_source_counts': split['training_source_counts'],
        'excluded_id': EXCLUDED_ID, 'split_sha256': file_hash(output / 'split.json'),
        'config_sha256': file_hash(output / 'config.json'),
        'data_manifest_sha256': file_hash(root / 'records/data_manifest.json'),
        'source_config_sha256': file_hash(root / 'original_split_retrain/config.json'),
        'source_sha256': {name: file_hash(root / name) for name in SOURCE_FILES},
        'initial_checkpoint': str(initial), 'initial_checkpoint_sha256': BASELINE_SHA256,
        'baseline': baseline, 'selection_tile': 512, 'final_tta_tile': 768,
        'early_stop_enabled': True, 'fresh_optimizer_scheduler_ema': True,
        'validation_usage': VALIDATION_USAGE, 'original_valid20_independent_evaluation': False,
        'checkpoint_writer': 'file handle torch.save, flush/fsync, atomic replace, up to3 attempts; failed files retained'}
    guards.require_identical(output / 'plan.json', plan)
    return config, split, plan


def robust_save_checkpoint(path, value, serializer=None):
    """Keep the previous checkpoint intact unless a fully flushed replacement succeeds."""
    if serializer is None:
        import torch
        serializer = torch.save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(1, 4):
        temporary = path.with_name(path.name + '.writing.' + uuid.uuid4().hex)
        try:
            with temporary.open('xb') as stream:
                serializer(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size == 0:
                raise OSError('Checkpoint writer produced an empty file')
            os.replace(temporary, path)
            return
        except Exception as error:
            failed = temporary.with_name(temporary.name + '.failed')
            if temporary.exists():
                try:
                    temporary.rename(failed)
                except OSError:
                    failed = temporary
            errors.append(f'{type(error).__name__}: {error}; preserved={failed}')
            print(f'CHECKPOINT_WRITE_RETRY {path.name} attempt={attempt}/3 {errors[-1]}', flush=True)
            if attempt < 3:
                time.sleep(1)
    raise RuntimeError('Checkpoint write failed after three attempts: ' + ' | '.join(errors))


def verify_dataset_boundary(core, data, split, config):
    if config['train_ids'] != split['train'] or config['donor_ids'] != split['train']:
        raise RuntimeError('Config train/donor IDs differ from frozen train89')
    dataset = core.PatchDataset(data, split['train'], VARIANT, config, 1)
    if dataset.ids != split['train'] or len(dataset.donors) != 89:
        raise RuntimeError('Actual dataset or augmentation donor count differs from89')
    for key, (image, label) in zip(dataset.ids, dataset.donors):
        entry = data[key]
        if (key == EXCLUDED_ID or key in split['test']
                or entry['entry']['source_subset'] not in ('train', 'valid')
                or image is not entry['image'] or label is not entry['label']):
            raise RuntimeError('Excluded/test image or unapproved donor reached training')


def checkpoint_write_probe(root, output):
    """Exercise the actual large CPU checkpoint path before allocating CUDA memory."""
    import torch
    source = root / 'original_split_retrain/runs/centerline_baseline/last.pt'
    probe = output / 'common/checkpoint_write_probe.pt'
    receipt = output / 'common/checkpoint_write_probe.json'
    source_sha = file_hash(source)
    if receipt.exists() and probe.exists():
        previous = read_json(receipt)
        if (previous.get('passed') is True and previous.get('source_sha256') == source_sha
                and previous.get('probe_sha256') == file_hash(probe)):
            return
        raise RuntimeError('Existing checkpoint probe or receipt differs; preserve for review')
    print('CHECKPOINT_WRITE_PROBE_START CPU clean baseline last.pt', flush=True)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    robust_save_checkpoint(probe, checkpoint)
    loaded = torch.load(probe, map_location='cpu', weights_only=False)
    if loaded.keys() != checkpoint.keys() or loaded['epoch'] != checkpoint['epoch']:
        raise RuntimeError('Large checkpoint probe reload metadata differs')
    for section in ('model', 'ema'):
        if loaded[section].keys() != checkpoint[section].keys():
            raise RuntimeError('Large checkpoint probe state keys differ')
        for key, tensor in checkpoint[section].items():
            if not torch.equal(tensor, loaded[section][key]):
                raise RuntimeError('Large checkpoint probe tensor differs: ' + section + '.' + key)
    save_json(receipt, {'passed': True, 'source': str(source), 'source_sha256': source_sha,
        'probe': str(probe), 'probe_sha256': file_hash(probe), 'bytes': probe.stat().st_size,
        'epoch': loaded['epoch'], 'model_and_ema_tensors_equal': True,
        'optimizer_scheduler_scaler_present': all(k in loaded for k in ('optimizer', 'schedule', 'scaler'))})
    print('CHECKPOINT_WRITE_PROBE_PASSED bytes=' + str(probe.stat().st_size), flush=True)


def run(config, split, plan, resume=False, root=ROOT):
    import torch
    import experiment as core
    from training_awake import keep_system_awake
    output = root / OUTPUT_NAME
    run_dir = output / 'runs' / VARIANT
    initial = Path(plan['initial_checkpoint'])
    provenance = {'variant': VARIANT, 'plan_sha256': file_hash(output / 'plan.json'),
        'source_sha256': plan['source_sha256'], 'split_sha256': plan['split_sha256'],
        'data_manifest_sha256': plan['data_manifest_sha256'],
        'initial_checkpoint': str(initial), 'initial_checkpoint_sha256': BASELINE_SHA256,
        'baseline_initial_sha256': FRESH_INITIAL_SHA256, 'baseline': plan['baseline'],
        'initialization_lineage': 'untouched initial -> audited train70 baseline -> fresh style89 fine-tune',
        'requested_boundary_verified': True, 'excluded_id': EXCLUDED_ID,
        'validation_usage': VALIDATION_USAGE}
    existed = (run_dir / 'provenance.json').exists()
    if run_dir.exists() and not existed and any(run_dir.iterdir()):
        raise RuntimeError('Unrecognized output directory; refusing overwrite')
    guards.require_identical(run_dir / 'provenance.json', provenance)
    boundary = {k: config[k] for k in ('train_ids', 'donor_ids', 'valid_ids', 'test_ids')}
    boundary.update(split_sha256=plan['split_sha256'], data_manifest_sha256=plan['data_manifest_sha256'],
                    requested_boundary_verified=True, excluded_id=EXCLUDED_ID)
    guards.require_identical(run_dir / 'data_boundary.json', boundary)
    expected_config = {**config, 'variant': VARIANT, 'initial_checkpoint': str(initial),
        'selection_tile': 512, 'early_stop_enabled': True,
        'status': 'reconstructed, original split/code unavailable'}
    saved_config = run_dir / 'config.json'
    if saved_config.exists() and read_json(saved_config) != expected_config:
        raise RuntimeError('Saved run configuration differs; refuse resume')
    started = any((run_dir / name).exists() for name in ('config.json', 'progress.json', 'last.pt', 'best.pt', 'result.json'))
    if started and not resume:
        raise RuntimeError('Run already started; explicit --resume required')
    if started and not (run_dir / 'result.json').exists() and not (run_dir / 'last.pt').exists():
        raise RuntimeError('Partial run has no last.pt; preserve and review, never blindly restart')
    if shutil.disk_usage(output).free < 4 * 1024**3:
        raise RuntimeError('At least4GiB free space required before training/checkpoint/ZIP')
    torch.set_num_threads(4)
    checkpoint_write_probe(root, output)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for priority training')
    core.save_checkpoint = robust_save_checkpoint
    with keep_system_awake(output / 'power_guard.json'):
        if not (run_dir / 'result.json').exists():
            manifest = read_json(root / 'records/data_manifest.json')
            data = original.load_original_data(manifest, core, root)
            verify_dataset_boundary(core, data, split, config)
            save_json(output / 'status.json', {'state': 'running', 'variant': VARIANT,
                'counts': split['counts'], 'excluded_id': EXCLUDED_ID})
            core.run_training(VARIANT, config, data, split, init=initial,
                resume=started and resume, run_directory=run_dir, selection_tile=512, early_stop=True)
        if prepare(root)[2] != plan or file_hash(initial) != BASELINE_SHA256:
            raise RuntimeError('Frozen inputs changed during training')
        if read_json(saved_config) != expected_config:
            raise RuntimeError('Actual run config differs from approved89-image configuration')
        result = read_json(run_dir / 'result.json')
        result.update({'experiment_phase': OUTPUT_NAME, 'requested_boundary_verified': True,
            'source_subset_split_verified': False, 'source_data_counts': split['counts'],
            'training_source_counts': split['training_source_counts'],
            'split_file': str(output / 'split.json'), 'split_sha256': plan['split_sha256'],
            'plan_sha256': file_hash(output / 'plan.json'), 'provenance_sha256': object_hash(provenance),
            'initial_checkpoint_sha256': BASELINE_SHA256, 'source_sha256': plan['source_sha256'],
            'excluded_id': EXCLUDED_ID, 'validation_usage': VALIDATION_USAGE,
            'validation_scope': 'single_image_reference',
            'baseline_holdout_used_for_checkpoint_selection': True,
            'original_valid20_independent_evaluation': False,
            'historical_reference_scores_comparable': False, 'target_79_reached': None,
            'single_image_reference_mIoU': result['validation']['mIoU']})
        save_json(run_dir / 'result.json', result)
        save_json(output / 'status.json', {'state': 'auditing', 'variant': VARIANT})
        import audit_priority_style89
        check = audit_priority_style89.main(root=root)
        if check.get('passed') is not True:
            raise RuntimeError('Independent style89 audit failed; ZIP not produced')
        save_json(output / 'status.json', {'state': 'packaging', 'variant': VARIANT})
        import build_priority_style89_share
        package = build_priority_style89_share.main(root=root)
        save_json(output / 'complete.json', {'complete': True, 'variant': VARIANT,
            'counts': split['counts'], 'excluded_id': EXCLUDED_ID, 'verification': check,
            'package': package, 'validation_usage': VALIDATION_USAGE})
        save_json(output / 'status.json', {'state': 'complete', 'variant': VARIANT,
            'single_image_reference_mIoU': result['validation']['mIoU'], 'package': package})
    print('PRIORITY_STYLE89_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only:
        guards.assert_no_other_training()
    with guards.exclusive_lock(ROOT / OUTPUT_NAME / 'active.lock'):
        if not args.prepare_only:
            guards.assert_no_other_training()
        config, split, plan = prepare()
        if args.prepare_only:
            print({'prepared': True, 'training_started': False, 'counts': split['counts'],
                   'excluded_id': EXCLUDED_ID, 'plan_sha256': object_hash(plan)}, flush=True)
            return
        try:
            run(config, split, plan, args.resume)
        except Exception as error:
            save_json(ROOT / OUTPUT_NAME / 'status.json', {'state': 'failed', 'variant': VARIANT,
                'error': type(error).__name__ + ': ' + str(error)})
            raise


if __name__ == '__main__':
    main()
