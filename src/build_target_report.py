"""Emit measured target-search Sheet rows only after independent result audit.

No native Sheet writes, workbook edits, model imports or GPU operations.
The caller must verify A130:K135 is blank (or equals its own last write).
"""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from target_search import (CANDIDATES, candidate_config, file_hash, object_hash,
                           read_json, save_json, target_evidence, validate_split)

ROOT=Path(__file__).resolve().parents[1]
HEADER=['실험명','모델','mIoU','TTA','Al','Primary Si','Al3Ni','Eutectic Si','설정','학습 시간','비고']


def build_blocks(root=ROOT):
    output=root/'target_search'; runs=output/'runs'
    finished=[name for name in CANDIDATES if (runs/name/'result.json').exists()]
    if not finished: return None
    audit_path=runs/'independent_audit.json'
    if not audit_path.exists(): raise RuntimeError('Independent audit required before target report')
    audits={row['variant']:row for row in read_json(audit_path)}
    plan=read_json(output/'plan.json')
    if set(plan['candidates'])!=set(CANDIDATES): raise RuntimeError('Unexpected candidate plan')
    split_file=root/'configs/split_group_fold0.json'; split=read_json(split_file); validate_split(split)
    split_sha=file_hash(split_file)
    if split_sha!=plan['split_sha256']: raise RuntimeError('Frozen split hash changed')
    shared=read_json(output/'common_initial_checkpoint.json')
    if shared['split_sha256']!=split_sha or shared['baseline_checkpoint']!=plan['baseline_checkpoint']:
        raise RuntimeError('Shared baseline provenance mismatch')
    if file_hash(Path(shared['baseline_checkpoint']))!=shared['baseline_checkpoint_sha256']:
        raise RuntimeError('Shared baseline checkpoint changed')
    measured={}
    for name in finished:
        run=runs/name; result_path=run/'result.json'; result=read_json(result_path)
        audit=audits.get(name)
        if audit is None: raise RuntimeError('Independent audit missing candidate: '+name)
        if audit_path.stat().st_mtime_ns<result_path.stat().st_mtime_ns:
            raise RuntimeError('Result changed after independent audit: '+name)
        provenance=read_json(run/'candidate_plan.json')
        config=candidate_config(plan['base_config'],name)
        expected={'candidate_id':name,'underlying_variant':'e_softmask_geometry',
                  'improvement_not_historical_reproduction':True,'config_sha256':object_hash(config),
                  'candidate_plan_sha256':object_hash(provenance),
                  'split_sha256':split_sha,'original_split_verified':False,
                  'baseline_checkpoint_sha256':shared['baseline_checkpoint_sha256']}
        if any(result.get(key)!=value for key,value in expected.items()):
            raise RuntimeError('Candidate result provenance mismatch: '+name)
        if (provenance.get('config')!=config or provenance.get('global_plan_sha256')!=object_hash(plan) or
            provenance.get('baseline_checkpoint_sha256')!=shared['baseline_checkpoint_sha256'] or
            provenance.get('source_sha256')!=plan['source_sha256']):
            raise RuntimeError('Candidate plan does not use frozen settings and shared initialization: '+name)
        evidence=target_evidence(result,audit,split_sha)
        classes=result['validation']['class_iou']
        if len(classes)!=4 or not all(math.isfinite(value) for value in classes):
            raise RuntimeError('Class IoUs missing/non-finite: '+name)
        if not math.isfinite(result['train_minutes']) or result['train_minutes']<0:
            raise RuntimeError('Training time missing/non-finite: '+name)
        measured[name]={'result':result,'evidence':evidence}
    reached=any(item['evidence']['target_met'] for item in measured.values())
    rows=[['추가 성능 개선 · E softmask 학습률 2종 · 동일 고정 분할·동일 초기 모델']+['']*10,HEADER.copy()]
    for name,definition in CANDIDATES.items():
        scale='0.1배' if name=='e_softmask_lr_x0p1' else '1/30배'
        settings=(f"성능 개선(LR만 {scale}): enc{definition['encoder_lr']:g}/dec{definition['decoder_lr']:g}; "
                  '동일 교정 중심선 best에서 각각 시작; EMA0.999·WD0.05; E softmask 기하 변형 유지; '
                  '고정 train72/valid18·검증14 A그룹; 600patch/ep·batch8·max100·patience20·minΔ0.02%p; '
                  '선택tile512 no-TTA·최종tile768 D4')
        row=[f'E softmask · 학습률 {scale}','U-Net + EfficientNetV2-S','','8방향','','','','',settings,
             '실행 불필요(다른 후보에서 79% 달성)' if reached else '대기','최종 점수 미측정.']
        if name in measured:
            result=measured[name]['result']; evidence=measured[name]['evidence']
            row[2]=evidence['mIoU']; row[4:8]=result['validation']['class_iou']
            row[9]=f"{result['epochs']}epoch {result['train_minutes']:.1f}분(경과); 선택{result['best_epoch']}epoch"
            row[10]=('완료·독립 검산 통과(valid18/test10 예측); 79% '+('달성' if evidence['target_met'] else '미달')+
                     '. 학습률을 조정한 성능 개선 실험이며 원본 재현 결과가 아님. test 정답 없어 test 점수 없음.')
        elif (runs/name/'progress.json').exists() or (runs/name/'config.json').exists():
            row[9]='실행 기록 있음·최종 평가 대기'; row[10]='최종 점수 미측정; 프로세스 진행 여부는 별도 확인.'
        rows.append(row)
    best=max(item['evidence']['mIoU'] for item in measured.values())
    rows.append([f'추가 개선 판정 · {len(measured)}/2 완료']+['']*7+[
        f"검산된 후보 최고 D4 mIoU {best:.6f}%. 이 고정 후보 분할에서 79% {'달성' if reached else '미달'}.",
        '', '검증18장에 맞춘 추가 최적화 결과. 일반화 성능이나 원본 분할 성능을 보장하지 않음.'])
    rows.append(['비교 조건·해석']+['']*7+[
        '두 후보는 동일 baseline SHA256·동일 split SHA256을 사용하며 초기 encoder/decoder LR만 변경. 원본 folds.csv·원본 가중치 미복구.',
        '', '원본 실험의 정확한 재현과 구분. 클래스 점수·mIoU는 새 실측만 기록하며 미측정 셀은 비움.'])
    return {'updated_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
            'sheet':'승현_ 실험기록','blocks':[{'start_row':130,'rows':rows}],
            'write_precondition':'Verify A130:K135 is blank or equals the last verified tool write before editing; preserve prior rows.',
            'target_met':reached,'audited_candidates':list(measured),'split_sha256':split_sha,
            'shared_baseline_sha256':shared['baseline_checkpoint_sha256']}


def main():
    result=build_blocks()
    if result is None:
        print(json.dumps({'written':False,'reason':'No completed target-search result to report'})); return
    path=ROOT/'target_search/report_sheet_blocks.json'; save_json(path,result)
    print(json.dumps({'written':True,'path':str(path),'target_met':result['target_met'],
                      'audited_candidates':result['audited_candidates']},ensure_ascii=False))


if __name__=='__main__': main()
