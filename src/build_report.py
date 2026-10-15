"""Build measured TXT/JSON report blocks; refresh XLSX by exporting native Sheet.

This script deliberately does not author the workbook. The existing source and
native Sheet preserve formats; export that Sheet after applying these blocks.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from monitor_status import inspect_run

ROOT=Path(__file__).resolve().parents[1]
NAMES=['centerline_baseline','e_softmask_geometry','low_lr_context_paste',
       'phase_style_exchange','e_protected_illumination']
LABELS=['기존 E 중심선 재학습','얇은 E 보존 기하 변형 (E softmask)',
        '저학습률 E 문맥 합성','상별 스타일 통계 교환','E 보호·주변 조명']
HEADER=['실험명','모델','mIoU','TTA','Al','Primary Si','Al3Ni','Eutectic Si','설정','학습 시간','비고']
COMMON=('셀 메모 복원: WD0.05·EMA0.999; native384→768; 600patch/ep·batch8·seed42; '
        '최대100ep·검증5ep·patience20ep·minΔ0.02%p; 선택tile512 no-TTA·최종tile768 D4·overlap0.5; '
        'A그룹 비중복 후보 train72/valid18·검증14그룹(원본 folds.csv 미확인)')
DETAILS=[
    'ImageNet 새 초기화; CE+Dice+0.1 E 중심선 recall; enc3e-5/dec3e-4; 최종 변환 GT skeleton→십자 팽창2회→E와 교집합; recall smooth1e-5',
    '교정 중심선 best에서 시작; E softmask·면적/연결성 검사; ±3–7°·704→768; 검사 임계값은 복원 가정',
    '동일 교정 중심선 best에서 시작; enc1e-6/dec1e-5; CE+Dice+0.1 dense E recall; 작은 E 성분 최대8개·Al 영역 feather paste; donor는 train만',
    '동일 교정 중심선 best에서 시작; train donor의 상별 median/IQR 교환·경계 smoothing; 세부 강도는 복원 가정',
    '동일 교정 중심선 best에서 시작; E+3px 보호·주변 gradient/contrast; 세부 강도는 복원 가정',
]
CAVEAT=('원본 코드·folds.csv·초기 가중치·패키지 버전 미복구. EMA/WD 등 확인값은 configs/record_notes.json. '
        'cosine per-update·최저LR1e-7, 일부 증강 세부값·fine-tune 선택tile512는 가정. '
        '후보 분할은 모델 점수를 보기 전에 고정했으며 원본·v1과 점수를 직접 비교할 수 없음.')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def audited_result(run,audits,expected_split=None):
    if not (run/'result.json').exists(): return None
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    audit=audits.get(run.name)
    if not audit or not audit['passed'] or abs(audit['independent_mIoU']-result['validation']['mIoU'])>=1e-10:
        raise RuntimeError(f'Independent audit required before reporting {run}')
    if expected_split and Path(audit.get('split_file','')).name!=expected_split:
        raise RuntimeError(f'Audit uses the wrong split for {run}')
    return result


def audit_map(directory):
    path=directory/'independent_audit.json'
    return {row['variant']:row for row in json.loads(path.read_text(encoding='utf-8'))} if path.exists() else {}


def measured_cells(row,result):
    row[2]=result['validation']['mIoU']; row[4:8]=result['validation']['class_iou']
    row[9]=f"{result['epochs']}epoch {result['train_minutes']:.1f}분; 선택{result['best_epoch']}epoch"
    row[10]=(f"완료·저장 마스크 독립 검산 일치; 얇은 E recall{result['validation']['thin_e_recall']:.3f}%; "
             'valid18·test10 추론 완료(test 정답 없음); 원본 분할 미확인으로 과거 점수와 직접 비교 불가.')
    if result['variant']=='e_softmask_geometry':
        row[10]+=' 이 후보 분할에서 79% '+('달성.' if row[2]>=79 else '미달.')


def main():
    legacy=json.loads((ROOT/'records/sheet_rows_base.json').read_text(encoding='utf-8'))
    legacy[0]=['2026-10-13 | 초기 v1 복원 기록 보존 · 셀 메모 설정 누락으로 종료']+['']*10
    legacy_results={}; audits=audit_map(ROOT/'runs')
    for i,name in enumerate(NAMES):
        row=legacy[i+2]; row[0]='초기 v1 · '+LABELS[i]
        row[8]+='; 초기 누락 조건: WD1e-4·EMA 없음·일반 이미지 KFold·native GT 중심선 선계산'
        result=audited_result(ROOT/'runs'/name,audits)
        if result:
            legacy_results[name]=result; measured_cells(row,result)
            row[10]+=' 셀 메모 설정 누락된 초기 복원 실측값으로 보존.'
        else:
            row[2]=''; row[4:8]=['']*4
            row[9]='중단; 재개용 last.pt 20epoch 보존' if name=='low_lr_context_paste' else '취소·미실행'
            row[10]='교정된 notes_recovered 5종으로 대체. 최종 D4 mIoU 미측정.'
    legacy[7]=['초기 v1 종료 · 2개 완료·1개 중단·2개 취소']+['']*7+[
        '중심선77.262805%·E softmask77.499622%는 초기 조건의 실제 측정값. 동일 환경 재현 성공을 뜻하지 않음.',
        '', 'mIoU: 이미지별 클래스 IoU 평균 후 이미지 평균(%); empty union 제외.']
    legacy[8]=['초기 복원 한계']+['']*7+[
        '이후 셀 메모에서 WD0.05·EMA0.999·최종 GT 중심선·A그룹 비중복 조건을 확인. 아래 교정 5종에 적용.',
        '', '원본 데이터·소스·실측 결과·체크포인트는 보존. runs/는 초기 v1 기록.']
    save(ROOT/'runs/sheet_rows_measured.json',legacy)

    canceled=json.loads((ROOT/'records/supplement_sheet_rows_base.json').read_text(encoding='utf-8'))
    canceled[0]=['2026-10-13 | 앞선 2종 보완 큐 취소 · 셀 메모 복원 5종으로 대체']+['']*10
    for row in canceled[2:4]:
        row[2]=''; row[4:8]=['']*4; row[9]='취소·학습 미실행'
        row[8]='tile512 선택·100ep만 보완하려던 계획. EMA·WD·그룹 분할 누락을 확인하여 실행 전 취소.'
        row[10]='최종 점수 없음. notes_recovered의 교정 5종으로 대체.'
    canceled[4]=['취소 판정 · 0/2 완료']+['']*7+['이 큐는 실행하지 않음. 교정 5종만 실행.','','']
    canceled[5]=['기록 보존']+['']*7+['recorded_protocol/plan.json 및 기존 소스는 취소된 계획의 이력.','','재개 명령은 notes_recovered.py 사용.']
    save(ROOT/'recorded_protocol/sheet_rows_measured.json',canceled)

    current=[['2026-10-13 | 셀 메모 설정 복원 5종 · EMA0.999·WD0.05·A그룹 분할 후보']+['']*10,HEADER.copy()]
    completed={}; audits=audit_map(ROOT/'notes_recovered/runs')
    for i,name in enumerate(NAMES):
        run=ROOT/'notes_recovered/runs'/name
        row=['셀 메모 복원 · '+LABELS[i],'U-Net + EfficientNetV2-S','','8방향','','','','',DETAILS[i]+'; '+COMMON,'대기','최종 점수 미측정.']
        result=audited_result(run,audits,'split_group_fold0.json')
        if result:
            completed[name]=result; measured_cells(row,result)
            if name=='centerline_baseline' and (ROOT/'records/power_interruption_20261014.json').exists():
                row[9]=f"{result['epochs']}epoch 경과 {result['train_minutes']:.1f}분(밤사이 절전 포함); 선택{result['best_epoch']}epoch"
                row[10]+=' 10/13 18:01~10/14 08:21 절전·최대절전으로 중단 후 동일 프로세스에서 계속됨. 경과시간은 순수 GPU 학습시간이 아님.'
        else:
            status=inspect_run(name,run)
            state=status['state']
            row[9]={'pending':'대기','prepared':'설정 준비됨','updating':'상태 갱신 중',
                    'recent_progress':f"진행 기록: {status.get('epoch')}epoch; 최종 평가 대기",
                    'stale_progress':f"최근 진행 확인 필요: {status.get('epoch')}epoch"}.get(state,state)
            row[10]='최종 D4 mIoU 미측정; 완료 후 독립 검산하여 기록. 원본 folds.csv와 일치 여부 미확인.'
        current.append(row)
    judgment=f'교정 5종 {len(completed)}/5 완료. '
    if completed:
        best=max(completed.values(),key=lambda r:r['validation']['mIoU'])
        judgment+=f"이 후보 분할의 완료 모델 최고 mIoU {best['validation']['mIoU']:.6f}%. "
    geometry=completed.get('e_softmask_geometry')
    judgment+=f"E softmask {geometry['validation']['mIoU']:.6f}%: 79% {'달성' if geometry['validation']['mIoU']>=79 else '미달'}." if geometry else 'E softmask 79% 달성 여부 미측정.'
    current.append([f'교정 판정 · {len(completed)}/5 완료']+['']*7+[judgment,'','과거/v1과 검증 파일이 달라 직접 성능 차이를 해석할 수 없음.'])
    current.append(['확인된 설정·남은 가정']+['']*7+[CAVEAT,'','notes_recovered/: 교정 모델·로그·valid/test 마스크·실측 결과.'])
    save(ROOT/'notes_recovered/sheet_rows_measured.json',current)
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    blocks=[{'start_row':103,'rows':legacy},{'start_row':113,'rows':canceled},{'start_row':120,'rows':current}]
    save(ROOT/'records/report_sheet_blocks.json',{'updated_kst':stamp,'sheet':'승현_ 실험기록','blocks':blocks,'workbook_refresh':'Export updated native Google Sheet to 재실험_결과.xlsx after applying and verifying blocks.'})
    text=[f'KOMAP 셀 메모 기반 재구성 ({stamp})','',judgment,'',COMMON,CAVEAT,'']
    for i,name in enumerate(NAMES):
        row=current[i+2]
        text.append(f'{LABELS[i]}: '+(f'D4 mIoU {row[2]:.6f}%; ' if isinstance(row[2],(int,float)) else '')+row[9])
    text+=['','초기 v1: 중심선77.26280490514819%, E softmask77.49962168453507% 실측값 보존.',
           'v1 low LR는 last.pt 20epoch를 보존한 뒤 중단; style/illum 미실행 취소.',
           'recorded_protocol의 2종 큐는 학습 시작 전 취소. 이 조건으로 재개하지 않습니다.','',
           '평가: 이미지별 클래스 IoU를 평균한 뒤 이미지 평균; empty union 제외; 단위%.',
           'Test 10장의 정답이 없어 test mIoU는 산출하지 않으며 예측 마스크만 저장합니다.',
           '후보 A그룹 분할은 모델 평가 전에 고정: train72/valid18, 검증14 A그룹, 학습/검증 A그룹 비중복.',
           '원본 folds.csv 미복구: 과거 및 v1 점수와 직접 비교하는 통제 실험이 아닙니다.','',
           'data/: 제공한 원본 복사본; records/data_manifest.json: 원본 해시',
           'records/historical_record_notes.json: 새로 확인한 원본 셀 메모',
           'configs/record_notes.json + split_group_fold0.json: 교정 설정 및 고정 후보 분할',
           'notes_recovered/runs/<실험>/: best.pt·last.pt·history·result·valid/test 마스크',
           'src/audit_results.py: 분할별 저장 마스크 독립 검산·그룹 조건·원본 해시 확인',
           'run_experiments.ps1: 교정 5종 재개 → 교정 분할 감사 → TXT/JSON 보고서 생성',
           '재실험_결과.xlsx는 검증 후 갱신한 원본 Google Sheet를 내보내어 저장합니다.',
           'requirements.lock.txt/records/environment.json: 현재 PC 환경; .venv는 다른 PC로 이동 실행 불가.']
    (ROOT/'실험결과.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'corrected_completed':len(completed),'legacy_completed':len(legacy_results),'report':str(ROOT/'실험결과.txt'),'sheet_blocks':str(ROOT/'records/report_sheet_blocks.json'),'xlsx_written':False},ensure_ascii=False))


if __name__=='__main__': main()
