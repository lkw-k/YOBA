"""실시간 모드: `uv run yoba --webcam --live`, `uv run yoba --rtsp "rtsp://..." --live [--publish]`

10초 주기 파이프라인과 같은 모델, 구역 집계, 블러 설정을 쓰되 매 프레임 처리해서 PC 창에 띄운다.
- 카메라는 CameraStream으로 읽어 최신 프레임만 처리한다. 추론이 CCTV 프레임 속도보다 느려도
  버퍼에 밀린 프레임 때문에 화면이 늦어지지 않고, 끊기면 종료 대신 재연결한다
- 프레임마다 검출 → 구역별 인원 → ZoneTracker(EMA, 히스테리시스)를 한 프레임 = 한 주기로 갱신
- 화면에는 블러된 이미지만 띄운다. 블러 검증이 실패하면 검은 화면으로 비운다
- --publish면 프레임마다 구역별 원본 인원과 블러 JPEG(테두리·박스 없음)를 Redis에 덮어쓴다 (spec 9.4)
- 원본 프레임은 저장하지 않는다
- 창에서 q 또는 Esc, 창 닫기로 종료
"""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np

from .aggregate import ZoneAssigner, ZoneResult, ZoneTracker
from .blur import privacy_blur
from .capture import WEBCAM_SCHEME, CameraStream, open_video
from .config import CameraConfig, ServiceConfig
from .detect import Detector, YoloDetector
from .publish import LivePublisher

log = logging.getLogger(__name__)

WINDOW = "YOBA live (blurred)"
_LEVEL_COLORS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 0, 255), None: (200, 200, 200)}  # BGR


class LiveView:
    """프레임 한 장 → 화면 한 장. 카메라·창과 분리해 테스트할 수 있게 한다."""

    def __init__(self, service: ServiceConfig, camera: CameraConfig, detector: Detector):
        self.service = service
        self.camera = camera
        self.detector = detector
        self.assigner = ZoneAssigner(camera.zones, camera.mask)
        self.trackers = [ZoneTracker(z, service.aggregate) for z in camera.zones]

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[ZoneResult], bytes | None]:
        """(화면, 구역 결과, 블러 JPEG). JPEG는 테두리·박스를 그리기 전 이미지이고 블러 실패면 None."""
        persons = self.detector.detect([frame])[0].persons
        counts = self.assigner.count(persons)
        results = [t.update([counts[t.zone.zone_id]]) for t in self.trackers]

        h, w = frame.shape[:2]
        jpeg = privacy_blur(frame, persons, self.service.blur)
        if jpeg is None:
            scale = min(1.0, self.service.blur.output_long_side / max(h, w))
            return np.zeros((round(h * scale), round(w * scale), 3), np.uint8), results, None

        view = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        scale = view.shape[1] / w  # privacy_blur가 줄인 비율에 맞춰 좌표를 옮긴다
        for zone, r in zip(self.camera.zones, results):
            pts = np.round(np.array(zone.roi) * scale).astype(np.int32)
            cv2.polylines(view, [pts], True, _LEVEL_COLORS[r.level], 2)
        for x0, y0, x1, y1 in np.asarray(persons).reshape(-1, 4):
            cv2.rectangle(view, (round(x0 * scale), round(y0 * scale)), (round(x1 * scale), round(y1 * scale)),
                          (255, 255, 0), 2)
        return view, results, jpeg


def _overlay(view: np.ndarray, results: list[ZoneResult], blurred: bool, fps: float) -> None:
    lines = [f"fps {fps:.1f}  " + ("blur OK" if blurred else "blur FAILED (image hidden)")]
    lines += [f"{r.zone_id}: {r.median:g} persons  score {r.score or 0:.2f}  level {r.level}" for r in results]
    for i, text in enumerate(lines):
        color = (0, 255, 255) if i == 0 else _LEVEL_COLORS[results[i - 1].level]
        cv2.putText(view, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _waiting_screen(service: ServiceConfig) -> np.ndarray:
    """첫 프레임을 받기 전 화면. 창을 먼저 띄워 두어야 q/Esc와 창 닫기가 동작한다."""
    long_side = service.blur.output_long_side
    view = np.zeros((long_side * 9 // 16, long_side, 3), np.uint8)
    cv2.putText(view, "connecting...", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return view


def run_live(
    service: ServiceConfig,
    camera: CameraConfig,
    detector: Detector | None = None,
    stream: CameraStream | None = None,
    publisher: LivePublisher | None = None,
) -> None:
    detector = detector or YoloDetector(service.detector)
    stream = stream or CameraStream(camera, service.capture)  # 연결 실패·끊김은 스트림이 로그를 남기고 재연결
    view_fn = LiveView(service, camera, detector)
    log.info("%s: 실시간 미리보기 시작. 창에서 q 또는 Esc로 종료", camera.id)
    if sys.platform == "darwin" and camera.url.lower().startswith(WEBCAM_SCHEME):
        # macOS는 카메라 권한 요청을 메인 스레드에서만 띄울 수 있어, 스트림 스레드보다 먼저 한 번 열었다 닫는다
        open_video(camera.url, service.capture).release()
    stream.start()
    cv2.imshow(WINDOW, _waiting_screen(service))
    last_seq = -1
    prev = time.perf_counter()
    try:
        while True:
            got = stream.latest()
            if got is not None and got[0] != last_seq:  # 새 프레임만 처리. 끊긴 동안은 마지막 화면을 유지
                last_seq, received_at, frame = got
                h, w = frame.shape[:2]
                view, results, jpeg = view_fn.process(frame)
                del frame, got  # 원본은 화면에 띄우지도 남기지도 않는다
                blurred = jpeg is not None
                now = time.perf_counter()
                fps, prev = 1 / max(now - prev, 1e-6), now
                if publisher is not None:
                    # 한 프레임 = 한 주기라 median이 그 프레임의 원본 인원 (스무딩은 웹이 한다)
                    counts = {r.zone_id: r.median for r in results}
                    publisher.publish((w, h), counts, jpeg, fps, (time.monotonic() - received_at) * 1000)
                _overlay(view, results, blurred, fps)
                cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        log.info("%s: 실시간 미리보기 종료", camera.id)
