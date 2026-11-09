"""카메라 수신과 5장 캡처 (spec 5.1).

- CameraStream: RTSP(또는 동영상 파일, 테스트용 웹캠 webcam:N)를 백그라운드 스레드에서 계속 읽어 최신 프레임 하나만 유지.
  끊기면 1초부터 두 배씩 reconnect_max_backoff까지 늘려 가며 재연결
- capture_burst: window_seconds 동안 frames장을 균등 간격(0, 0.75, ... 3.0초)으로 수집.
  같은 프레임이 두 번 들어가거나 오래된 프레임이 섞이지 않게 한다 (끊긴 스트림이 정상 5장처럼 보이는 것 방지)
- CCTV RTSP는 rtsp_transport(기본 TCP)로 받고, 연결·수신 타임아웃을 둔다.
  CCTV나 NVR이 응답 없이 멈춰도 read()가 무한정 걸리지 않고 read_timeout 뒤 재연결한다
- 원본 프레임은 메모리에만 둔다. 로그에는 URL(RTSP 계정 포함) 대신 카메라 id만 남긴다
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .config import CameraConfig, CaptureConfig

log = logging.getLogger(__name__)

WEBCAM_SCHEME = "webcam:"  # webcam:N → PC에 연결된 N번 카메라 (테스트용)
_LIVE_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", WEBCAM_SCHEME)


def open_video(url: str, cfg: CaptureConfig) -> cv2.VideoCapture:
    """FFmpeg 백엔드로 연다. RTSP 전송 방식은 FFmpeg 옵션 환경변수로만 줄 수 있어 열기 전에 설정한다.
    webcam:N은 OpenCV 기본 백엔드로 장치 번호를 연다."""
    if url.lower().startswith(WEBCAM_SCHEME):
        return cv2.VideoCapture(int(url[len(WEBCAM_SCHEME):]))
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{cfg.rtsp_transport}"
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(cfg.open_timeout * 1000),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(cfg.read_timeout * 1000),
    ]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


class CameraStream:
    def __init__(
        self,
        camera: CameraConfig,
        cfg: CaptureConfig,
        opener: Callable[[str], cv2.VideoCapture] | None = None,
    ):
        self.camera = camera
        self.cfg = cfg
        self._opener = opener or (lambda url: open_video(url, cfg))  # 테스트에서 가짜 캡처로 바꿔 끼운다
        self._live = camera.url.lower().startswith(_LIVE_SCHEMES)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0  # 새 프레임을 받을 때마다 1 증가
        self._received_at = 0.0  # time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.camera.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._lock:
            self._frame = None

    def latest(self) -> tuple[int, float, np.ndarray] | None:
        """(순번, 수신 시각, 프레임). 아직 받은 프레임이 없으면 None."""
        with self._lock:
            if self._frame is None:
                return None
            return self._seq, self._received_at, self._frame

    def _run(self) -> None:
        backoff = min(1.0, self.cfg.reconnect_max_backoff)
        while not self._stop.is_set():
            cap = self._opener(self.camera.url)
            try:
                if not cap.isOpened():
                    log.warning("%s: 스트림 연결 실패, %.0f초 후 재시도", self.camera.id, backoff)
                else:
                    log.info("%s: 스트림 연결됨", self.camera.id)
                    if self._read_loop(cap):
                        backoff = min(1.0, self.cfg.reconnect_max_backoff)  # 프레임을 받았으면 대기 초기화
                    if self._stop.is_set():
                        return
                    log.warning("%s: 스트림 끊김, %.0f초 후 재연결", self.camera.id, backoff)
            finally:
                cap.release()
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.cfg.reconnect_max_backoff)

    def _read_loop(self, cap) -> bool:
        """읽기에 실패하거나 stop까지 최신 프레임을 갱신한다. 프레임을 하나라도 받았으면 True."""
        # 동영상 파일은 디코딩 속도로 읽으면 몇 초 만에 끝나므로 원래 fps에 맞춰 읽는다
        interval = 0.0
        if not self._live:
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = 1.0 / fps if fps and fps > 0 else 0.0

        got = False
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                return got
            got = True
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._received_at = time.monotonic()
            if interval:
                self._stop.wait(interval)
        return got


def capture_burst(stream: CameraStream, cfg: CaptureConfig) -> list[np.ndarray]:
    """window_seconds 동안 frames장을 균등 간격으로 모은다.

    시점마다 최신 프레임을 가져오되, 직전에 가져온 것과 같은 프레임이거나 간격보다 오래된 프레임은 버린다.
    그래서 스트림이 멈추거나 끊기면 frames장보다 적게 돌려준다 (min_frames 판정은 호출 측).
    """
    step = cfg.window_seconds / (cfg.frames - 1) if cfg.frames > 1 else 0.0
    max_age = step if step > 0 else cfg.window_seconds
    start = time.monotonic()
    frames: list[np.ndarray] = []
    last_seq = -1

    for i in range(cfg.frames):
        delay = start + i * step - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        got = stream.latest()
        if got is None:
            continue
        seq, received_at, frame = got
        if seq == last_seq or time.monotonic() - received_at > max_age:
            continue
        last_seq = seq
        frames.append(frame)

    if len(frames) < cfg.min_frames:
        log.warning("%s: 캡처 %d/%d장 (min_frames %d 미만)", stream.camera.id, len(frames), cfg.frames, cfg.min_frames)
    return frames
