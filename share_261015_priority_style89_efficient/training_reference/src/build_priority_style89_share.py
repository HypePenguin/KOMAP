"""Package the audited 89-image priority style model without converting weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
NAME = 'share_261015_priority_style89_A170_excluded'
LIMIT = 100_000_000
EXCLUDED = 'A170_step4_sample3'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)

def verified_copy(source, destination):
    before = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(source) != before or sha(destination) != before:
        raise RuntimeError('Source changed while packaging: '+str(source))

def manifest(stage, metadata):
    files = [{'path':path.relative_to(stage).as_posix(), 'bytes':path.stat().st_size, 'sha256':sha(path)}
             for path in sorted(stage.rglob('*')) if path.is_file() and path.name != 'PACKAGE_MANIFEST.json']
    write(stage/'PACKAGE_MANIFEST.json', {**metadata, 'files':files})
    return files

def validate_archive(archive, stage):
    expected = {NAME+'/'+p.relative_to(stage).as_posix():p for p in stage.rglob('*') if p.is_file()}
    with zipfile.ZipFile(archive) as zipped:
        actual = [p.filename for p in zipped.infolist() if not p.is_dir()]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise RuntimeError('ZIP file list differs from package')
        if zipped.testzip() is not None:
            raise RuntimeError('ZIP CRC verification failed')
        for name, path in expected.items():
            with zipped.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != sha(path):
                raise RuntimeError('ZIP member SHA mismatch: '+name)
    if archive.stat().st_size >= LIMIT:
        raise RuntimeError('ZIP is not under 100 MB; never silently change model precision')

def main(root=None):
    root = Path(root or ROOT).resolve()
    output = root/'priority_style89_20261015'
    run = output/'runs/phase_style_exchange'
    # Re-run the independent CPU audit before packaging. No GPU inference here.
    import audit_priority_style89
    audit_priority_style89.main(root)
    audit = read(run/'verification.json')
    if audit.get('passed') is not True:
        raise RuntimeError('Independent audit has not passed')
    split, result, config, plan = (read(p) for p in (output/'split.json',run/'result.json',run/'config.json',output/'plan.json'))
    if len(split['train']) != 89 or split['valid'] != [EXCLUDED] or EXCLUDED in split['train']:
        raise RuntimeError('Unexpected requested training boundary')
    if set(config['train_ids']) != set(split['train']) or set(config['donor_ids']) != set(split['train']):
        raise RuntimeError('Actual crop/donor boundary differs')
    target = root/(NAME+'.zip')
    receipt = output/'package_complete.json'
    if receipt.exists():
        prior = read(receipt)
        if not target.is_file() or sha(target) != prior['zip_sha256'] or target.stat().st_size != prior['zip_bytes']:
            raise RuntimeError('Recorded ZIP is missing or changed')
        if sha(run/'best.pt') != prior['model_sha256']:
            raise RuntimeError('Final model changed after packaging')
        return prior
    if target.exists():
        raise RuntimeError('Unrecorded ZIP already exists; preserve it for review')
    stage = output/'share_package'/NAME
    if stage.exists():
        raise RuntimeError('Partial package exists; preserve it for review')
    stage.mkdir(parents=True)
    for name in ('best.pt','result.json','config.json','history.json','verification.json','provenance.json','data_boundary.json'):
        if (run/name).exists():
            verified_copy(run/name, stage/'model'/name)
    for directory in ('valid_masks_d4','test_masks_d4'):
        for path in sorted((run/directory).glob('*.png')):
            verified_copy(path,stage/'predictions'/directory/path.name)
    for name in ('plan.json','split.json','config.json','audit.json','initial_provenance.json','write_probe.json'):
        if (output/name).exists():
            verified_copy(output/name,stage/'records'/name)
    for name in ('initial_provenance.json','checkpoint_write_probe.json'):
        if (output/'common'/name).exists():
            verified_copy(output/'common'/name,stage/'records'/name)
    for name in ('data_manifest.json','environment.json'):
        verified_copy(root/'records'/name,stage/'records'/name)
    for relative, digest in plan['source_sha256'].items():
        path = root/relative
        if sha(path) != digest:
            raise RuntimeError('Frozen training source changed: '+relative)
        verified_copy(path,stage/relative)
    for name in ('experiment.py','training_awake.py','priority_style89.py','audit_priority_style89.py','original_split_retrain.py','audit_original_split.py','target_search.py'):
        if not (stage/'src'/name).exists():
            raise RuntimeError('Required training source missing from frozen plan: '+name)
    verified_copy(root/'src/share_model_priority.py',stage/'src/share_model.py')
    verified_copy(root/'requirements.lock.txt',stage/'requirements.lock.txt')
    packages = {'numpy','pillow','opencv-python-headless','scipy','scikit-image','scikit-learn','segmentation_models_pytorch','timm'}
    pins = [line for line in (root/'requirements.lock.txt').read_text(encoding='utf-8-sig').splitlines()
            if line.partition('==')[0].lower() in packages]
    if len(pins) != len(packages):
        raise RuntimeError('Incomplete dependency pins')
    (stage/'requirements.training.txt').write_text('# Install CUDA-enabled torch/torchvision first; see environment.json.\n'+'\n'.join(pins)+'\n',encoding='utf-8')
    readme = f'''KOMAP — 요청한 89장 학습 모델 (2026-10-15)

모델 위치: model/best.pt (EMA 선택 모델, 원본 FP32 저장 그대로)
학습기법: 시트125행 phase_style_exchange. 이번에 새로 학습한 별도 모델입니다.
학습/증강: 원본 train70 + valid19 =89장.
제외 이미지: {EXCLUDED}_image.png. 이미지와 정답은 학습 crop/donor/역전파에 사용하지 않았습니다.
초기값: 원본 train70만 학습한 새 baseline best. 과거 valid15를 학습한 80.2575 모델은 사용하지 않았습니다.
기존 baseline은 원본 valid20으로 체크포인트를 선택했으며, 제외 이미지도 그 평가에 포함됐습니다.

이번 최종 D4 mIoU: {result['validation']['mIoU']:.9f}% (제외한 A170 단1장 기준)
이 수치는 원본 valid20 성능이나 독립 test 성능이 아닙니다. 19장이 이번 학습에 들어갔으므로 기존20장 평가와 직접 비교할 수 없습니다.
test10은 정답이 없어 추론 마스크만 포함합니다. 실제 측정 결과와 독립 검산은 model/result.json, model/verification.json에 있습니다.

압축을 풀고 해당 폴더에서 실행:
  python src/share_model.py verify
  python src/share_model.py check-model
  python src/share_model.py infer --input "이미지폴더" --output "새출력폴더"
추론은 Windows/NVIDIA CUDA 환경이며 tile768, D4 8방향을 사용합니다.
환경 기록: records/environment.json 및 requirements.lock.txt.
CUDA에 맞는 torch/torchvision을 먼저 설치한 다음 requirements.training.txt를 설치하세요.

학습 코드와 정확한89/1/10 목록 및 원본 데이터 해시를 포함했습니다.
100MB 미만 공유를 위해 원본 데이터, 가상환경, optimizer/재개 상태, 초기 baseline 가중치는 포함하지 않았습니다.
학습 재현에는 원본 데이터와 plan.json에 명시된 초기 baseline 체크포인트를 별도로 준비해야 합니다.
이 ZIP의 best.pt는 완성된 추론 모델이며, 초기 baseline 없이 바로 불러올 수 있습니다.
'''
    (stage/'README.txt').write_text(readme,encoding='utf-8-sig')
    metadata = {'package':NAME,'created_utc':datetime.now(timezone.utc).isoformat(),
                'train_count':89,'excluded_image':EXCLUDED+'_image.png','holdout_count':1,
                'metric_scope':'single-image reference, not original valid20 or independent test',
                'model_sha256':sha(run/'best.pt'),'weight_conversion':'none','final_holdout_mIoU':result['validation']['mIoU']}
    manifest(stage,metadata)
    checked = subprocess.run([sys.executable,'-B','-X','utf8',str(stage/'src/share_model.py'),'check-model'],
        cwd=stage,capture_output=True,text=True,encoding='utf-8',timeout=180)
    if checked.returncode:
        raise RuntimeError('CPU model load failed: '+checked.stdout+'\n'+checked.stderr)
    check = json.loads(checked.stdout.strip().splitlines()[-1])
    if check['parameters'] != 22093908 or check['selected_epoch'] != result['best_epoch'] or check['model_kind'] != 'EMA':
        raise RuntimeError('CPU model metadata mismatch')
    write(stage/'PACKAGE_CHECKS.json',{'cpu_load':check,'independent_audit_passed':True,'model_precision_changed':False})
    manifest(stage,metadata)
    partial = target.with_name(target.stem+'.partial.zip')
    if partial.exists():
        raise RuntimeError('Partial ZIP already exists; preserve it for review')
    bandizip = Path(r'C:\Program Files\Bandizip\bz.exe')
    if not bandizip.is_file():
        raise RuntimeError('Bandizip CLI missing')
    packed = subprocess.run([str(bandizip),'c','-l:9','-r',str(partial),NAME],cwd=stage.parent,capture_output=True,timeout=600)
    (output/'bandizip_create.log').write_bytes(packed.stdout+packed.stderr)
    if packed.returncode:
        raise RuntimeError('Bandizip failed: '+str(packed.returncode))
    tested = subprocess.run([str(bandizip),'t',str(partial)],capture_output=True,timeout=120)
    (output/'bandizip_test.log').write_bytes(tested.stdout+tested.stderr)
    if tested.returncode:
        raise RuntimeError('Bandizip integrity test failed')
    validate_archive(partial,stage)
    partial.replace(target)
    record = {**metadata,'zip_path':str(target),'zip_bytes':target.stat().st_size,'zip_sha256':sha(target),
              'under_100MB':True,'zip_integrity_verified':True,'cpu_checkpoint_verified':True,
              'package_completed_utc':datetime.now(timezone.utc).isoformat()}
    write(receipt,record)
    print('PRIORITY_STYLE89_ZIP '+str(target),flush=True)
    return record

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    args=parser.parse_args()
    main(args.root)
