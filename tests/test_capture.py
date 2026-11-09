import logging
import threading
import time

import cv2
import numpy as np

from ywhob.capture import CameraStream, capture_burst
from ywhob.config import CameraConfig, CaptureConfig, ZoneConfig

ZONE = ZoneConfig("z1", ((0, 0), (1, 0), (1, 1)), 10)
URL = "rtsp://admin:secret@10.0.0.1:554/stream1"
FAST = CaptureConfig(window_seconds=0.4, reconnect_max_backoff=0.05)  # 테스트용으로 짧게


def _camera(url=URL):
    return CameraConfig("cam01", url, (ZONE,))


class FakeCapture:
    """cv2.VideoCapture 대역. 프레임 픽셀 값에 순번을 넣어 어떤 프레임인지 구분한다."""

    def __init__(self, opened=True, n_frames=None, period=0.01, stall_after=None, fps=0.0):
        self.opened = opened
        self.n_frames = n_frames  # 이만큼 읽으면 read 실패 (끊김)
        self.period = period
        self.stall_after = stall_after  # 이만큼 읽은 뒤 read가 멈춤
        self.fps = fps
        self.count = 0
        self.released = False
        self.unstall = threading.Event()

    def isOpened(self):
        return self.opened

    def read(self):
        if self.n_frames is not None and self.count >= self.n_frames:
            return False, None
        if self.stall_after is not None and self.count >= self.stall_after:
            self.unstall.wait(5)
            return False, None
        time.sleep(self.period)
        self.count += 1
        return True, np.full((4, 4, 3), self.count % 256, np.uint8)

    def get(self, prop):
        return self.fps

    def release(self):
        self.released = True


class Opener:
    """호출될 때마다 caps에서 하나씩 꺼내 준다. 다 쓰면 마지막 것을 새로 만든다."""

    def __init__(self, *caps):
        self.caps = list(caps)
        self.calls = 0

    def __call__(self, url):
        self.calls += 1
        if len(self.caps) > 1:
            return self.caps.pop(0)
        last = self.caps[0]
        return FakeCapture(opened=last.opened, n_frames=last.n_frames, period=last.period)


def _run(stream, fn):
    stream.start()
    try:
        return fn()
    finally:
        stream.stop()


def _wait_frame(stream, timeout=2.0):
    end = time.monotonic() + timeout
    while stream.latest() is None and time.monotonic() < end:
        time.sleep(0.01)


def test_burst_collects_distinct_frames():
    stream = CameraStream(_camera(), FAST, opener=Opener(FakeCapture()))

    def go():
        _wait_frame(stream)
        return capture_burst(stream, FAST)

    frames = _run(stream, go)
    assert len(frames) == 5
    assert len({int(f[0, 0, 0]) for f in frames}) == 5  # 모두 다른 프레임


def test_burst_spans_window():
    stream = CameraStream(_camera(), FAST, opener=Opener(FakeCapture()))

    def go():
        _wait_frame(stream)
        t0 = time.monotonic()
        capture_burst(stream, FAST)
        return time.monotonic() - t0

    elapsed = _run(stream, go)
    assert 0.38 <= elapsed < 0.6


def test_unreachable_camera_returns_no_frames_and_retries():
    opener = Opener(FakeCapture(opened=False))
    stream = CameraStream(_camera(), FAST, opener=opener)
    frames = _run(stream, lambda: capture_burst(stream, FAST))
    assert frames == []
    assert opener.calls >= 3  # 백오프로 계속 재시도


def test_reconnects_after_disconnect():
    first = FakeCapture(n_frames=3)
    opener = Opener(first, FakeCapture())
    stream = CameraStream(_camera(), FAST, opener=opener)

    def go():
        end = time.monotonic() + 2
        while opener.calls < 2 and time.monotonic() < end:
            time.sleep(0.01)
        _wait_frame(stream)
        time.sleep(0.05)
        return capture_burst(stream, FAST)

    frames = _run(stream, go)
    assert first.released
    assert opener.calls >= 2
    assert len(frames) == 5


def test_stalled_stream_does_not_repeat_frame():
    # 프레임 2장 뒤 멈춘 스트림: 마지막 프레임을 5번 넣어 정상처럼 보이면 안 된다
    cap = FakeCapture(stall_after=2)
    stream = CameraStream(_camera(), FAST, opener=Opener(cap, FakeCapture(opened=False)))

    def go():
        end = time.monotonic() + 2
        while cap.count < 2 and time.monotonic() < end:
            time.sleep(0.01)
        time.sleep(0.15)  # 멈춘 뒤 캡처 간격(0.1초)보다 오래 지남
        return capture_burst(stream, FAST)

    try:
        frames = _run(stream, go)
    finally:
        cap.unstall.set()
    assert frames == []


def test_stale_frame_rejected():
    # 오래전에 받은 프레임은 burst 첫 장으로도 쓰지 않는다
    stream = CameraStream(_camera(), FAST, opener=Opener(FakeCapture(opened=False)))
    stream._frame = np.zeros((4, 4, 3), np.uint8)
    stream._seq = 1
    stream._received_at = time.monotonic() - 10
    assert capture_burst(stream, FAST) == []


def test_stop_is_prompt_during_backoff():
    cfg = CaptureConfig(reconnect_max_backoff=30)
    stream = CameraStream(_camera(), cfg, opener=Opener(FakeCapture(opened=False)))
    stream.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    stream.stop()
    assert time.monotonic() - t0 < 1.0
    assert stream.latest() is None


def test_video_file_read_at_source_fps():
    opener = Opener(FakeCapture(period=0, fps=20.0), FakeCapture(period=0, fps=20.0))
    stream = CameraStream(_camera("sample.mp4"), FAST, opener=opener)

    def go():
        time.sleep(0.5)
        return stream.latest()[0]

    seq = _run(stream, go)
    assert 5 <= seq <= 15  # 20fps × 0.5초 ≈ 10장. 제한이 없으면 수천 장


def test_real_video_file_with_ffmpeg_opener(tmp_path):
    # 가짜 없이 OpenCV FFmpeg 백엔드(open_video)로 실제 파일을 읽는다
    path = tmp_path / "sample.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(60):
        w.write(np.full((48, 64, 3), i * 4, np.uint8))
    w.release()

    cfg = CaptureConfig(window_seconds=0.4)
    stream = CameraStream(_camera(str(path)), cfg)

    def go():
        _wait_frame(stream)
        return capture_burst(stream, cfg)

    frames = _run(stream, go)
    assert len(frames) >= cfg.min_frames
    assert frames[0].shape == (48, 64, 3)


def test_url_not_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="ywhob.capture")
    stream = CameraStream(_camera(), FAST, opener=Opener(FakeCapture(opened=False)))
    _run(stream, lambda: capture_burst(stream, FAST))
    assert caplog.records
    assert "secret" not in caplog.text and "10.0.0.1" not in caplog.text


def test_open_video_webcam_uses_device_index(monkeypatch):
    import ywhob.capture as capture

    calls = []
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda *a, **k: calls.append(a) or FakeCapture())
    capture.open_video("webcam:1", FAST)
    assert calls == [(1,)]  # FFmpeg 백엔드가 아니라 장치 번호로 연다


def test_webcam_is_live_stream():
    stream = CameraStream(_camera("webcam:0"), FAST, opener=Opener(FakeCapture(fps=1.0)))
    _run(stream, lambda: _wait_frame(stream))
    assert stream._live  # 동영상 파일처럼 fps에 맞춰 늦춰 읽지 않는다
