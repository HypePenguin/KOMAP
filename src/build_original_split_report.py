"""Report original-folder 70/20 experiments and disclose prior repartitioning."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['centerline_baseline', 'e_softmask_geometry', 'low_lr_context_paste', 'phase_style_exchange']
LABELS = ['기존 E 중심선 재학습', '얇은 E 보존 기하 변형 (E softmask)', '저학습률 E 문맥 합성', '상별 스타일 통계 교환']
WARNING = ('정정(2026-10-15): 원본 train70+valid20을 재분할하여 원본 valid 15장이 학습에 포함됨. '
           '아래 실측 점수는 별도 재분할 valid18 기준이며, 원본 valid20 성능·동일 조건 재현·원본 분할 목표 달성의 근거로 사용할 수 없음.')
COMMON = ('원본 폴더 경계 고정 train70/valid20/test10; valid는 평가·체크포인트 선택만, 학습·증강 donor 제외; '
          'EMA0.999·WD0.05·seed42·batch8·600patch/ep; 최대100ep·검증5ep·patience20·minΔ0.02%p; '
          '선택tile512 no-TTA·최종tile768 D4')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    old = next(b for b in read(ROOT/'records/report_sheet_blocks.json')['blocks'] if b['start_row'] == 120)
    old['rows'][0][0] = '2026-10-13 | 재분할 참고 실험 · 원본 valid 15장 학습 사용 · 원본 valid20 평가 아님'
    for row in old['rows'][2:7]:
        row[0] = row[0].replace('셀 메모 복원 · ', '재분할 참고 · ')
        row[10] = WARNING + ' ' + row[10]
    old['rows'][7][0] = '재분할 결과 정정 · 원본 분할 목표 미검증'
    old['rows'][7][8] = '기존 5종 점수는 재분할 valid18 실측값으로 보존. 원본 train70/valid20의 4종 결과는 아래 별도 기록.'
    old['rows'][7][10] = '기존 79~80% 점수를 원본 valid20 기준 성능으로 제시하지 않음.'
    old['rows'][8][8] = WARNING + ' ' + old['rows'][8][8]

    run_root = ROOT/'original_split_retrain/runs'
    audit_path = run_root/'independent_audit.json'
    audits = {a['variant']: a for a in read(audit_path)} if audit_path.exists() else {}
    current = [['2026-10-15 | 원본 train70만 학습 · valid20 평가 전용 · 122~125행 기법 재실험']+['']*10,
               ['실험명', '모델', 'mIoU', 'TTA', 'Al', 'Primary Si', 'Al3Ni', 'Eutectic Si', '설정', '학습 시간', '비고']]
    completed = {}
    for i, name in enumerate(NAMES):
        row = ['원본 분할 · '+LABELS[i], 'U-Net + EfficientNetV2-S', '', '8방향', '', '', '', '', COMMON, '대기', '원본 valid20 최종 점수 미측정.']
        row[8] += '; ' + ('KOMAP 미학습 초기 가중치에서 새 학습' if i == 0 else '이번 원본 train70 기준모델 best에서 독립 시작')
        if name == 'low_lr_context_paste':
            row[8] += '; enc1e-6/dec1e-5'
        else:
            row[8] += '; enc3e-5/dec3e-4'
        result_path = run_root/name/'result.json'
        if result_path.exists():
            result = read(result_path)
            audit = audits.get(name, {})
            if not (audit.get('passed') and audit.get('valid_masks') == 20 and audit.get('test_masks') == 10):
                raise RuntimeError('Original source split audit required for '+name)
            if abs(audit['independent_mIoU']-result['validation']['mIoU']) >= 1e-10:
                raise RuntimeError('Audit differs from result: '+name)
            completed[name] = result
            row[2] = audit['independent_mIoU']
            row[4:8] = result['validation']['class_iou']
            row[9] = f"{result['epochs']}epoch 경과 {result['train_minutes']:.1f}분; 선택{result['best_epoch']}epoch"
            row[10] = '완료·원본 valid20 저장 마스크 독립 검산 통과; test10 정답 없음. 과거 folds.csv·초기 가중치 미확인.'
        elif (run_root/name/'progress.json').exists():
            p = read(run_root/name/'progress.json')
            epoch = p.get('epoch_in_progress', p.get('epoch', p.get('epoch_completed', '?')))
            row[9] = f'진행 기록: {epoch}epoch; 최종 평가 대기'
        current.append(row)
    judgment = f'원본 분할 {len(completed)}/4 완료. '
    soft = completed.get('e_softmask_geometry')
    judgment += (f"E softmask valid20 {soft['validation']['mIoU']:.6f}%: 79% {'달성' if soft['validation']['mIoU'] >= 79 else '미달'}." if soft else 'E softmask 원본 valid20 79% 달성 여부 미측정.')
    current.append([f'원본 분할 판정 · {len(completed)}/4 완료']+['']*7+[judgment, '', '실측값만 기록; 기존 재분할 모델 가중치 재사용 금지.'])
    current.append(['분할·초기화 원칙']+['']*7+[
        '원본 source_subset 경계 그대로 유지; 원본 valid20 학습/증강 donor 제외. 기준모델부터 새로 학습. A그룹 겹침 여부는 기록하되 임의 재분할하지 않음.', '',
        '원본 folds.csv·과거 초기 체크포인트 미확인. data/와 이전 결과/공유 ZIP 보존; 이전 ZIP은 원본 valid20 평가용으로 사용 불가.'])
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    payload = {'updated_kst': stamp, 'sheet': '승현_ 실험기록', 'blocks': [old, {'start_row': 130, 'rows': current}]}
    save(ROOT/'records/original_split_report_blocks.json', payload)
    text = ['KOMAP 원본 폴더 분할 재학습 ('+stamp+')', WARNING, '', COMMON, '', judgment, '']
    for row in current[2:6]:
        text.append(row[0]+': '+(f'mIoU {row[2]:.6f}%; ' if isinstance(row[2], (int, float)) else '')+row[9])
    text += ['', '기존 공유 ZIP의 모델은 원본 valid 일부를 학습한 모델이므로 원본 valid20 평가에 사용할 수 없습니다.',
             'original_split_retrain/에 이번 4종의 별도 설정·분할·초기화·결과를 보관합니다.']
    (ROOT/'원본분할_재학습결과.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps({'completed': len(completed), 'blocks': [120, 130], 'xlsx_written': False}))


if __name__ == '__main__':
    main()
