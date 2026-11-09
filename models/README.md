# models/

학습된 가중치(.pt)와 배포용 모델(.onnx, .engine)을 두는 곳. 파일 자체는 git에 올리지 않는다.

파일 이름: `yolo26s-ywhob-v{N}.{pt,onnx}`

새 버전을 추가할 때 아래 표에 기록한다 (spec 4.2 모델 카드).

| 버전 | 데이터 | imgsz | 정밀도 | person mAP50 | 최소 검출 크기 (spec 3.7) | 인원 MAPE | 비고 |
|---|---|---|---|---|---|---|---|
| COCO 기준선 (`yolo26s.pt`, 학습 없음) | COCO | 960 / 1280 | FP32 | - | 128px / 128px (입력 기준, 가시율 ≥ 0.75, CrowdHuman val) | - | 2026-09-29 M2. 가시율 ≥ 0.75 recall 0.624 / 0.689 |
| `yolo26s-ywhob-v1.pt` | CrowdHuman | 1280 | FP32 | 0.946 | - | - | `runs/yolo26s-ywhob/weights/best.pt`. 서비스 별칭 `s` (기본) |
| `yolo26n-ywhob-v1.pt` | CrowdHuman | 1280 | FP32 | 0.928 | - | - | `runs/yolo26n-ywhob/weights/best.pt`. 서비스 별칭 `n` |

서비스에서 쓸 모델은 `uv run ywhob --model s|n`으로 고른다 (별칭은 `configs/service.yaml`의 `detector.models`).
