"""CPU-only independent audit of the explicitly requested train89/holdout1 run.

The one-image score is a model-selection reference, not a valid20 estimate.
Only priority_style89_20261015/audit.json and its run verification are written.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import audit_original_split as original

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'priority_style89_20261015'
VARIANT = 'phase_style_exchange'
EXCLUDED = 'A170_step4_sample3'
BASELINE_SHA256 = '6bf6809c18cb49d576e704fa0738d54e9e83fc0abc47a21e227ec796ee2054be'
FRESH_INITIAL_SHA256 = 'dc9106bf16c414017e7365555e31e78c4a3bfe598267d4270c4ada3fbba98bb2'
VALIDATION_SCOPE = 'single_image_reference'
read_json = original.read_json
file_hash = original.file_hash
check = original.check


def expected_ids(manifest):
    supplied = original.source_ids(manifest)
    check(EXCLUDED in supplied['valid'], 'Excluded A170 image must belong to supplied valid20')
    return {'train': supplied['train'] | (supplied['valid'] - {EXCLUDED}),
            'valid': {EXCLUDED}, 'test': supplied['test']}


def validate_split(manifest, split):
    expected = expected_ids(manifest)
    for part, ids in expected.items():
        values = split.get(part)
        check(isinstance(values, list) and len(values) == len(set(values)), 'Missing/duplicate split IDs: ' + part)
        check(set(values) == ids, 'Priority split must be exact train89/holdout1/test10: ' + part)
    return expected


def assert_training_ids(ids, manifest, role='training', require_exact=True):
    allowed = expected_ids(manifest)['train']
    check(isinstance(ids, (list, tuple)) and len(ids) == len(set(ids)), role + ': missing/duplicate IDs')
    check(EXCLUDED not in ids, role + ': excluded A170 image must never enter training or donors')
    check(not (set(ids) - allowed), role + ': test or unknown image is forbidden')
    if require_exact:
        check(set(ids) == allowed, role + ': must contain exact requested train89')


def validate_boundary(manifest, split, payload, label, split_sha, manifest_sha):
    validate_split(manifest, split)
    assert_training_ids(payload.get('train_ids'), manifest, label + '.train_ids')
    assert_training_ids(payload.get('donor_ids'), manifest, label + '.donor_ids')
    for part in ('valid', 'test'):
        values = payload.get(part + '_ids')
        check(isinstance(values, list) and len(values) == len(set(values)) and set(values) == set(split[part]),
              label + '.' + part + '_ids mismatch')
    check(payload.get('split_sha256') == split_sha, label + ': split SHA mismatch')
    check(payload.get('data_manifest_sha256') == manifest_sha, label + ': manifest SHA mismatch')


def validate_scope(payload, label):
    check(payload.get('validation_scope') == VALIDATION_SCOPE, label + ': single-image reference disclosure missing')
    check(payload.get('baseline_holdout_used_for_checkpoint_selection') is True,
          label + ': baseline checkpoint-selection use of A170 must be disclosed')


def verify_initialization(root, manifest, provenance):
    baseline = root / 'original_split_retrain/runs/centerline_baseline'
    source = baseline / 'best.pt'
    check(file_hash(source) == BASELINE_SHA256, 'Clean train70 baseline checkpoint SHA mismatch')
    initial = original.contained_path(root, provenance['initial_checkpoint'])
    check(file_hash(initial) == BASELINE_SHA256 and provenance.get('initial_checkpoint_sha256') == BASELINE_SHA256,
          'Priority initialization is not the approved clean train70 baseline')
    baseline_split_file = root / 'original_split_retrain/split.json'
    baseline_split = read_json(baseline_split_file)
    split_sha = file_hash(baseline_split_file)
    manifest_sha = file_hash(root / 'records/data_manifest.json')
    original.validate_original_split(manifest, baseline_split)
    original.validate_run_boundary(read_json(baseline / 'config.json'), read_json(baseline / 'data_boundary.json'),
                                   baseline_split, manifest, split_sha, manifest_sha)
    evidence = read_json(baseline / 'verification.json')
    check(evidence.get('passed') is True and evidence.get('valid_masks') == 20 and evidence.get('test_masks') == 10,
          'Clean baseline lacks a passed original valid20/test10 audit')
    check(evidence.get('split_sha256') == split_sha and evidence.get('data_manifest_sha256') == manifest_sha,
          'Clean baseline audit split/manifest SHA mismatch')
    for key, name in [('result_sha256', 'result.json'), ('config_sha256', 'config.json'),
                      ('data_boundary_sha256', 'data_boundary.json')]:
        check(evidence.get(key) == file_hash(baseline / name), 'Clean baseline audit is stale: ' + name)
    result = read_json(baseline / 'result.json')
    baseline_provenance = read_json(baseline / 'provenance.json')
    check(result.get('initial_checkpoint_sha256') == FRESH_INITIAL_SHA256 and
          baseline_provenance.get('initial_checkpoint_sha256') == FRESH_INITIAL_SHA256,
          'Clean baseline must descend from the approved fresh initialization')
    frozen_baseline = provenance.get('baseline', {})
    expected_baseline = {'sha256': BASELINE_SHA256,
                        'baseline_result_sha256': file_hash(baseline / 'result.json'),
                        'baseline_verification_sha256': file_hash(baseline / 'verification.json'),
                        'baseline_provenance_sha256': file_hash(baseline / 'provenance.json'),
                        'baseline_data_boundary_sha256': file_hash(baseline / 'data_boundary.json'),
                        'baseline_split_sha256': split_sha, 'baseline_initial_sha256': FRESH_INITIAL_SHA256,
                        'baseline_training_ids': baseline_split['train'],
                        'baseline_validation_ids': baseline_split['valid'],
                        'excluded_image_used_in_baseline_training': False,
                        'excluded_image_used_in_baseline_selection': True}
    check(all(frozen_baseline.get(key) == value for key, value in expected_baseline.items()),
          'Frozen clean baseline provenance changed or is incomplete')
    check(original.contained_path(root, frozen_baseline['checkpoint']) == source,
          'Frozen baseline source points to a different checkpoint')
    original.compare_numbers([evidence['independent_mIoU']], [result['validation']['mIoU']], 'Baseline audit score')
    check(result.get('best_epoch') == 65, 'Approved baseline must be its selected epoch65')
    return {'baseline_checkpoint_sha256': BASELINE_SHA256, 'baseline_best_epoch': 65,
            'baseline_train_ids_verified': 70, 'baseline_donor_ids_verified': 70,
            'baseline_verification_sha256': file_hash(baseline / 'verification.json'),
            'baseline_holdout_used_for_checkpoint_selection': True}


def audit_masks(root, run, manifest, split, result):
    entries = {row['id']: row for row in manifest}
    records = result['validation']['per_image']
    check(len(records) == 1 and records[0]['id'] == EXCLUDED, 'Validation result must contain only excluded A170')
    for part in ('valid', 'test'):
        names = {key + '_mask.png' for key in split[part]}
        check({path.name for path in (run / (part + '_masks_d4')).glob('*.png')} == names,
              part + ': saved mask filenames differ from the frozen split')
    target = original.decode(original.contained_path(root, entries[EXCLUDED]['mask']))
    prediction = original.decode(run / 'valid_masks_d4' / (EXCLUDED + '_mask.png'))
    check(target.shape == prediction.shape, 'A170 prediction dimensions differ from ground truth')
    scores = []
    for index in range(4):
        intersection = np.count_nonzero((target == index) & (prediction == index))
        union = np.count_nonzero((target == index) | (prediction == index))
        scores.append(float(intersection / union * 100) if union else None)
    score = float(np.mean([value for value in scores if value is not None]))
    for payload, label in [(records[0], 'A170'), (result['validation'], 'Single-image aggregate')]:
        original.compare_numbers([score], [payload['mIoU']], label + ' mIoU')
        original.compare_numbers(scores, payload['class_iou'], label + ' class IoUs')
    if 'global_mIoU' in result['validation']:
        original.compare_numbers([score], [result['validation']['global_mIoU']], 'Single-image global mIoU')
    if 'confusion' in records[0]:
        confusion = np.bincount((target.astype(np.int64) * 4 + prediction).ravel(), minlength=16).reshape(4, 4)
        check(np.array_equal(confusion, records[0]['confusion']), 'Saved A170 confusion differs from mask pixels')
    for key in split['test']:
        predicted = original.decode(run / 'test_masks_d4' / (key + '_mask.png'))
        check(predicted.shape == (entries[key]['height'], entries[key]['width']), 'Test mask dimensions mismatch: ' + key)
    check(result['test'].get('count') == 10 and result['test'].get('mIoU') is None,
          'Test has no labels: require10 masks and no test mIoU')
    return {'independent_mIoU': score, 'independent_class_iou': scores,
            'difference': abs(score - result['validation']['mIoU']), 'valid_masks': 1, 'test_masks': 10,
            'prediction_sha256': {str(path.relative_to(run)): file_hash(path)
                                  for part in ('valid', 'test')
                                  for path in sorted((run / (part + '_masks_d4')).glob('*.png'))}}


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main(root=None):
    root = Path(root or ROOT).resolve()
    output = (root / OUTPUT).resolve()
    run = output / 'runs' / VARIANT
    check(output.is_relative_to(root) and run.resolve().is_relative_to(output), 'Audit output escapes project')
    manifest_file = root / 'records/data_manifest.json'
    split_file = output / 'split.json'
    manifest, split = read_json(manifest_file), read_json(split_file)
    split_sha, manifest_sha = file_hash(split_file), file_hash(manifest_file)
    ids = validate_split(manifest, split)
    check(split.get('source_subset_manifest_sha256') == manifest_sha, 'Priority split manifest SHA mismatch')
    source_report = original.verify_source_files(root, manifest)
    source_hashes = {row['id']: row['image_sha256'] for row in manifest}
    check(source_hashes[EXCLUDED] not in {source_hashes[key] for key in ids['train']},
          'A170 exact image duplicate appears in training89')
    result, config = read_json(run / 'result.json'), read_json(run / 'config.json')
    provenance = read_json(run / 'provenance.json')
    for label, payload in [('run config', config), ('data boundary', read_json(run / 'data_boundary.json')),
                           ('shared config', read_json(output / 'config.json'))]:
        validate_boundary(manifest, split, payload, label, split_sha, manifest_sha)
    for label, payload in [('result', result), ('run config', config)]:
        validate_scope(payload, label)
    plan = read_json(output / 'plan.json')
    plan_sha = file_hash(output / 'plan.json')
    check(plan.get('config_sha256') == file_hash(output / 'config.json'), 'Shared configuration differs from frozen plan')
    check(plan.get('baseline') == provenance.get('baseline'), 'Run baseline provenance differs from frozen plan')
    source_files = plan.get('source_sha256')
    check(isinstance(source_files, dict) and bool(source_files), 'Frozen source hashes are missing')
    for relative, digest in source_files.items():
        check(file_hash(original.contained_path(root, relative)) == digest, 'Frozen source changed: ' + relative)
    for label, payload in [('result', result), ('provenance', provenance)]:
        check(payload.get('source_sha256') == source_files, label + ': frozen source hashes mismatch')
    for label, payload in [('result', result), ('provenance', provenance)]:
        check(payload.get('split_sha256') == split_sha and payload.get('plan_sha256') == plan_sha,
              label + ': plan/split SHA mismatch')
        check(payload.get('initial_checkpoint_sha256') == BASELINE_SHA256, label + ': initialization SHA mismatch')
    check(provenance.get('data_manifest_sha256') == manifest_sha, 'Provenance manifest SHA mismatch')
    check(result.get('variant') == VARIANT, 'Wrong priority variant')
    check(original.contained_path(root, result['split_file']) == split_file, 'Result points to a different split')
    check(result.get('source_data_counts') == {'train': 89, 'valid': 1, 'test': 10}, 'Result split counts must be89/1/10')
    check(result.get('selection_tile') == 512 and result.get('final_tta_tile') == 768,
          'Expected selection512/final768 D4 protocol')
    check(result.get('model_kind') == 'EMA' and result.get('ema_decay') == 0.999 and result.get('weight_decay') == 0.05,
          'Expected EMA0.999/weight decay0.05')
    initialization = verify_initialization(root, manifest, provenance)
    measured = audit_masks(root, run, manifest, split, result)
    report = {'variant': VARIANT, 'passed': True, **measured, **source_report, **initialization,
              'split_sha256': split_sha, 'data_manifest_sha256': manifest_sha, 'plan_sha256': plan_sha,
              'result_sha256': file_hash(run / 'result.json'), 'config_sha256': file_hash(run / 'config.json'),
              'provenance_sha256': file_hash(run / 'provenance.json'),
              'data_boundary_sha256': file_hash(run / 'data_boundary.json'),
              'best_checkpoint_sha256': file_hash(run / 'best.pt'),
              'train_ids_verified': 89, 'donor_ids_verified': 89, 'excluded_id': EXCLUDED,
              'excluded_from_backpropagation_and_donors': True,
              'validation_scope': VALIDATION_SCOPE, 'valid20_evaluation': False,
              'metric_caveat': 'Single A170 image used for model selection, including baseline selection; not a valid20 score or an independent generalization estimate.'}
    write_json_atomic(run / 'verification.json', report)
    write_json_atomic(output / 'audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    main(parser.parse_args().root)
