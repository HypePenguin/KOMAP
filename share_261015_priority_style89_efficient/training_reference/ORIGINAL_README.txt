KOMAP — 요청한 89장 학습 모델 (2026-10-15)

모델 위치: model/best.pt (EMA 선택 모델, 원본 FP32 저장 그대로)
학습기법: 시트125행 phase_style_exchange. 이번에 새로 학습한 별도 모델입니다.
학습/증강: 원본 train70 + valid19 =89장.
제외 이미지: A170_step4_sample3_image.png. 이미지와 정답은 학습 crop/donor/역전파에 사용하지 않았습니다.
초기값: 원본 train70만 학습한 새 baseline best. 과거 valid15를 학습한 80.2575 모델은 사용하지 않았습니다.
기존 baseline은 원본 valid20으로 체크포인트를 선택했으며, 제외 이미지도 그 평가에 포함됐습니다.

이번 최종 D4 mIoU: 86.614280177% (제외한 A170 단1장 기준)
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
