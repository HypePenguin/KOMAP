KOMAP train89 모델 — 새 이미지 추론 안내

1. 포함 파일과 모델 범위

efficient.py에서 모델 구성, 가중치 로딩, 전처리와 추론을 실행합니다.
model/best.pt는 학습 완료된 원본 체크포인트이며, 이번 추론 코드 정리 과정에서 변경하지 않았습니다.
training_reference/에는 기존 학습 소스를 참고용으로 보존했습니다. efficient.py 추론은 이 폴더의 코드를 불러오지 않습니다.

이 모델은 원본 train 70장과 원본 valid 중 19장을 합친 89장으로 학습했습니다.
선택된 체크포인트는 25 epoch이며, 학습에서 제외한 A170 이미지 1장의 mIoU는 86.6143%입니다.
이 값은 단일 이미지 평가 결과입니다. 원본 valid 20장 전체 점수나 새로운 데이터에 대한 일반화 성능으로 해석하면 안 됩니다.

2. 설치

압축을 풀고 efficient.py가 있는 폴더에서 실행하세요. Python 환경에 맞는 torch와 torchvision을 먼저 설치한 뒤, 동봉된 추론용 의존성 파일을 설치합니다.

  python -m pip install -r requirements-inference.txt

GPU 추론에는 해당 PC의 NVIDIA 드라이버와 호환되는 CUDA용 torch/torchvision이 필요합니다.
검증 환경은 CUDA입니다. CUDA를 사용할 수 없으면 CPU로 실행할 수 있지만 속도가 느리고, 연산 환경에 따라 확률과 경계 픽셀 결과에 작은 수치 차이가 생길 수 있습니다.

3. 폴더의 이미지 추론

  python efficient.py --input "이미지폴더" --output "결과폴더" --save-probs

--input에는 추론할 이미지가 있는 폴더, --output에는 결과를 저장할 폴더를 지정합니다.
--save-probs를 포함하면 분할 결과와 함께 클래스별 확률을 저장합니다. 확률 저장이 필요 없으면 이 옵션을 빼면 됩니다.

추론은 기존 방식인 타일 크기 768, 타일 이동 간격 384, D4 8방향 변환을 사용합니다.
각 변환 결과를 원래 방향으로 되돌리고 겹치는 타일 및 변환의 확률을 평균하여 원본 이미지 해상도의 결과를 만듭니다.

4. 파이썬에서 확률 가져오기

  import numpy as np
  from PIL import Image
  from efficient import EfficientPredictor

  predictor = EfficientPredictor()
  rgb_uint8 = np.asarray(Image.open("example.png").convert("RGB"), dtype=np.uint8)
  probs = predictor.predict_proba(rgb_uint8)
  labels = probs.argmax(axis=0).astype(np.uint8)

입력 rgb_uint8: 원본 해상도의 RGB 이미지, numpy.uint8, 모양 (H, W, 3).
OpenCV로 이미지를 읽었다면 BGR을 RGB로 변환해서 전달하세요.

반환 probs: numpy.float32, 모양 (4, H, W), 원본 이미지 해상도.
클래스 순서는 아래와 같습니다.

  0: Al
  1: Primary Si
  2: Al3Ni
  3: Eutectic Si

predict_proba는 확률을 반환합니다. 최종 클래스 번호가 필요하면 위 예시처럼 채널 축(axis=0)에 argmax를 별도로 적용합니다.
여러 이미지를 처리할 때는 EfficientPredictor를 한 번 생성한 뒤 재사용하세요.

5. 다른 모델과 앙상블

각 모델의 확률을 같은 클래스 순서, 같은 해상도, 같은 이미지 위치와 방향으로 맞춘 뒤 평균합니다.
예를 들어 동일 조건의 두 확률 배열은 아래처럼 결합할 수 있습니다.

  ensemble_probs = (probs_a + probs_b) / 2.0
  ensemble_labels = ensemble_probs.argmax(axis=0).astype(np.uint8)

자르기, 크기 변경, 회전 등을 사용한 모델은 먼저 원본 이미지 좌표로 복원해야 합니다.
클래스 번호나 색상 마스크를 평균하지 말고, 정렬된 클래스별 확률을 평균한 다음 argmax를 적용하세요.
앙상블 성능은 별도의 평가 데이터로 확인해야 합니다.

6. 출력과 추가 옵션

결과폴더/*_mask.png: 원래 색상표의 RGB 마스크.
결과폴더/labels/*_label.png: 0~3 클래스 번호 PNG.
결과폴더/probabilities/*_prob.npy: --save-probs 사용 시 float32 (4,H,W) 확률.
  probs = np.load("결과폴더/probabilities/example_prob.npy", allow_pickle=False)
inference_info.json에 클래스 순서, 색상표, 추론 조건이 기록됩니다.
입력 폴더는 하위 폴더를 탐색하지 않습니다. 기존 출력 폴더는 덮어쓰지 않습니다.
마스크·레이블·원본 이미지가 섞이지 않도록 입력에는 추론할 이미지만 넣으세요.

  python efficient.py --check-model
  python efficient.py --input "example.png" --output "one_image" --device cuda

--device cpu로 CPU를 명시할 수 있습니다. VRAM 부족 시 --batch-size 2로 줄일 수 있습니다.
원본 검증 조건은 CUDA/FP16 추론, batch-size 8입니다. 배치 변경도 미세한 수치 차이를 만들 수 있습니다.
다른 폴더에서 실행해도 기본 모델 경로는 efficient.py 옆 model/best.pt입니다.
다른 위치에 복사한 동일 체크포인트는 --weights로 지정할 수 있습니다. 다른 모델의 체크포인트는 SHA 검사에서 거부합니다.

7. 기록 보존과 재현 범위

training_reference는 이전 소스를 원문 그대로 보존한 참고 자료입니다.
단일 파일로 정리한 범위는 모델 구성·로딩·전처리·추론·확률 출력이며 재학습 runner를 합친 것은 아닙니다.
원본 데이터와 초기 baseline 가중치, 가상환경, optimizer 상태는 이번 ZIP에도 포함하지 않았습니다.
기존 학습 코드를 다시 실행하려면 해당 데이터와 초기 가중치, 원래 경로 배치가 별도로 필요합니다.
이 ZIP의 efficient.py 추론은 이런 파일을 요구하지 않으며 새 모델 다운로드도 하지 않습니다.
모델 정밀도 변환이나 추가 학습은 하지 않았습니다.

8. 이번 코드 검증

모델 체크포인트 SHA256은 원본과 동일합니다.
A170와 test10 총11장의 저장 마스크를 새 코드와 비교했습니다.
그중 10장은 원본 코드를 같은 프로세스에서 다시 실행하여 확률 배열이 완전히 같은지 확인했고 모두 일치했습니다.
작은 비정방형 이미지의 패딩·크기 복원도 원본 코드와 일치했습니다.
과거 저장 마스크와 재추론 비교는 9,398,964픽셀 중 39픽셀 차이였습니다.
원본 코드를 다시 실행해도 같은 차이가 나므로 새 코드 변경으로 생긴 차이가 아닙니다.
CUDA FP16/cuDNN 실행의 미세한 수치 차이로 경계 픽셀과 최종 소수점 점수가 조금 달라질 수 있습니다.
기존 학습 결과의 86.6143%는 model/result.json의 원래 측정값으로 보존했습니다.
비교 상세는 records/inference_equivalence.json, CPU/API 검증은 records/cpu_api_check.json에 있습니다.
단일 명령 실행으로 RGB 마스크·클래스 번호·확률 파일을 저장하는 것도 확인했습니다.
비정상 수치(NaN/Inf)가 나온 입력은 임의 마스크를 만들지 않고 오류로 중단합니다. 입력 확인 후 --device cpu로 FP32 추론을 시도할 수 있습니다.
