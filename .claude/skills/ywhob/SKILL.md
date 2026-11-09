---
name: ywhob
description: YwHob 혼잡도 파이프라인 작업 절차. 파이프라인 단계(capture, detect, aggregate, blur, publish)를 구현하거나 고칠 때, 모델 학습·평가·export·벤치마크를 돌릴 때, 임계값이나 ROI를 조정할 때 사용한다.
---

# YwHob 작업 절차

모든 작업은 `spec.md`를 기준으로 한다. 시작 전에 해당 spec 절을 읽는다.

> 현재는 뼈대만 있다. 모듈과 스크립트 본문은 `NotImplementedError`이므로 아래 명령은 구현 전까지 실행되지 않는다. 사용자가 요청한 단계만 구현한다.

## 1. 파이프라인 단계를 구현하거나 고칠 때

| 모듈 | spec | 핵심 요구사항 |
|---|---|---|
| `ywhob/capture.py` | 5.1 | 최신 프레임만 유지, 3초간 균등 5장, min_frames 미만이면 error, 원본은 메모리에만 |
| `ywhob/detect.py` | 5.2 | 5장 배치 추론, 0번(person)만 남김, person_conf, max_det |
| `ywhob/matching.py` | 5.3 | (보류) 머리 확인 규칙. 현재 파이프라인에서 쓰지 않음 |
| `ywhob/aggregate.py` | 5.4~5.6 | 발 위치로 구역 배정, 중앙값, 절대+비율 편차 신뢰도, EMA, 히스테리시스 |
| `ywhob/blur.py` | 5.7 | 사람 박스만 축소 → 확대 블러, 박스별 선명도 검증, 실패 시 None |
| `ywhob/publish.py` | 5.8, 9 | 이미지 먼저 JSON 나중, `?v=` 버전 쿼리, low/error 처리, Redis 장애 무시 |
| `ywhob/pipeline.py` | 2, 6 | 카메라별 시점 분산, 크기 제한 큐(오래된 것 버림), 처리 시간 경고 |

순서:
1. spec 절과 모듈의 `TODO(구현)` docstring을 읽는다.
2. 튜닝 값은 `configs/service.yaml` / `ywhob/config.py`에서 읽는다. 새 값이 필요하면 두 곳에 같이 추가하고 (잠정) 표시를 단다.
3. 테스트를 `tests/test_<모듈>.py`에 추가한다. 카메라, Redis(`fakeredis`), 모델은 가짜로 대체한다.
4. `uv run pytest -q` 통과를 확인한다.
5. 동작이 spec과 달라졌으면 spec 수정안을 사용자에게 보고한다. 커밋은 사용자 검토 후.

## 2. 모델 작업 (spec 3~4)

```bash
uv run python scripts/convert_crowdhuman.py   # datasets/ywhob 생성
uv run python scripts/train.py                # configs/train.yaml 사용, 결과는 runs/
uv run python scripts/evaluate_count.py       # 인원 MAE/MAPE, person 박스 높이별 recall
uv run python scripts/export.py               # ONNX (+INT8), max_det=1000 필수
uv run python scripts/benchmark.py            # 5장 배치 시간, 디코딩 부하, 수용 카메라 수
```

- 모델 채택 기준은 mAP가 아니라 **인원 MAPE**다 (목표 20% 이하, 잠정).
- 양자화 후 MAPE가 FP32 대비 +2%p를 넘게 나빠지면 FP16/FP32를 쓴다.
- 채택한 모델은 `models/yolo26s-ywhob-v{N}.onnx`로 두고 `models/README.md` 표에 지표를 기록한다.
- RTX 4070(12GB)에서 imgsz 1280 학습은 batch가 작게 잡힌다. 메모리 부족이면 batch를 직접 낮춘다.

## 3. 임계값과 ROI 조정

- 카메라, ROI, capacity, 마스크: `configs/cameras.yaml` (예시는 `cameras.example.yaml`)
- 신뢰도, EMA, 단계 임계값, 블러: `configs/service.yaml`
- 바꾼 뒤 `uv run ywhob --check`로 설정 검증 (히스테리시스 간격, 구역 id 중복 등을 잡아냄)
- 각 값의 근거는 바탕화면 `YwHob_수치결정근거.docx`와 spec 참고. 기준 밀도와 단계 임계값은 운영 정책이라 사용자 확인 없이 바꾸지 않는다.

## 하지 말 것

- 원본 프레임을 파일로 저장하거나 로그에 남기기 (디버그 목적 포함)
- 튜닝 값을 코드에 하드코딩
- COCO 가중치(`yolo26s.pt`)를 쓸 때 0번(person) 필터 없이 쓰기 (80개 클래스가 전부 사람으로 세짐)
- 사용자 검토 없이 커밋/푸시
