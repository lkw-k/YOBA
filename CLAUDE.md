# CLAUDE.md

YOBA: 박람회 CCTV를 10초마다 캡처해 구역별 혼잡도(3단계 + 연속 점수)와 블러 이미지를 Redis로 웹 서버에 넘기는 시스템.

## 기준 문서

- **spec.md가 구현 기준이다.** 코드의 동작이 spec과 다르면 코드를 고치거나, 사용자 확인 후 spec을 같이 고친다.
- README.md는 개요와 결정 배경. 확정된 내용만 담는다.
- spec의 표기: (확정) / (잠정: 실측 후 조정) / (수정 예정: 방향만 정해짐). 코드와 설정 주석에도 같은 표기를 유지한다.

## 현재 상태

**구현 완료, 실사용 종료.** 2026-10-31 ~ 11-01 박람회에서 실제 CCTV로 운영을 마쳤다 (10초 주기, 노트북 1대).
capture → detect → aggregate → blur → publish와 pipeline 연결까지 구현돼 있다. `matching.py`(spec 5.3, 머리 확인)는 보류 결정대로 `NotImplementedError`로 남긴다.
검출 모델은 파인튜닝한 스몰(`s`, 기본)과 나노(`n`) 중 `--model`로 고른다 (`configs/service.yaml`의 `detector.models`).
스몰 가중치는 Hugging Face `illimax/YolPe`에 공개돼 있고, 학습 코드는 `github.com/lkw-k/YolPe`에 있다.

## 환경

- Python 3.11, uv로 관리. torch는 CUDA 12.6 빌드 (개발 PC: RTX 4070 12GB)
- 이 PC에는 `python` 명령이 없다 (Windows Store 별칭만 있음). 항상 `uv run` 또는 `.venv/Scripts/python` 사용

```bash
uv sync --extra train --group dev   # 전체 설치 (학습 포함)
uv run pytest -q                    # 테스트
uv run yoba --cameras configs/cameras.example.yaml --check   # 설정 검증
uv run yoba                         # 서비스 실행 (configs/cameras.yaml 필요, 스몰 모델)
uv run yoba --model n               # 나노 모델로 실행
uv run yoba --webcam [번호]          # CCTV 대신 연결된 웹캠으로 실행 (configs/cameras.webcam.yaml, 화면 전체 한 구역)
uv run yoba --webcam --live         # 매 프레임 처리해 블러 화면을 PC 창에 띄움 (--publish 없으면 Redis 전송 없음, q/Esc 종료)
uv run yoba --rtsp "rtsp://..." --live   # CCTV 한 대를 RTSP로 실시간 테스트 (화면 전체 한 구역, 같은 설정 파일)
uv run yoba --rtsp "rtsp://..." --live --publish   # 실시간 모드: 프레임마다 Redis(yoba:live, yoba:frame)에도 씀 (spec 9.4)
uv run python scripts/roi_editor.py --id cam01 --url <rtsp 주소>   # CCTV 화면에 ROI 그리기
```

`uv sync`는 지정하지 않은 extra를 지운다. 학습 패키지를 유지하려면 `--extra train`을 빼먹지 않는다.

## 구조

```
yoba/       capture → detect → aggregate → blur → publish, pipeline.py가 연결 (matching.py는 보류)
configs/    service.yaml(주기, 임계값, Redis), cameras.example.yaml(카메라, ROI), data.yaml, train.yaml
scripts/    데이터 변환, 학습, 평가, export, 벤치마크, ROI 편집 (spec 3~4, 7)
models/     가중치 (git 제외, README.md에 버전 기록)
tests/
```

## 규칙

- **원본 프레임은 디스크, 로그, 네트워크 어디에도 남기지 않는다.** 디버그용 원본 저장 옵션도 만들지 않는다. 블러 실패 시 이미지는 보내지 않는다.
- 임계값, 비율, 주기 같은 튜닝 값은 `configs/`에 두고 코드에 하드코딩하지 않는다.
- 모델 클래스는 `0: person` 하나다 (`head`는 보류, spec 3.2). COCO 가중치를 그대로 쓸 때는 `classes=[0]`으로 person만 남긴다.
- YOLO26은 end-to-end(NMS 없음) 모델이다. ONNX/TensorRT로 export할 때 `max_det`이 그래프에 고정되므로 `max_det=1000`을 넘긴다 (기본 300이면 혼잡 장면에서 인원이 잘림).
- `configs/cameras.yaml`은 RTSP 계정이 들어가므로 git에 올리지 않는다 (.gitignore 처리됨).
- 새 로직에는 테스트를 같이 추가한다. 카메라/Redis/모델 없이 돌 수 있게 가짜 객체를 쓴다.

## 작업 방식

- **커밋과 푸시는 사용자가 검토한 뒤에만 한다.** 요청이 없으면 변경만 하고 보고한다.
- 문서, 주석, 로그 메시지는 한국어로 쓴다.
- 스크립트와 `yoba` 실행은 저장소 루트에서 한다 (`configs/`, `datasets/` 상대 경로 기준).
