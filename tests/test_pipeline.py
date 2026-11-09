import json
import threading
import time
from datetime import datetime

import fakeredis
import numpy as np

from yoba import pipeline as pl
from yoba.config import AggregateConfig, CameraConfig, CaptureConfig, ServiceConfig, ZoneConfig
from yoba.detect import FrameDetections
from yoba.pipeline import DetectItem, Pipeline
from yoba.publish import KST

ZONE_A = ZoneConfig("a", ((0, 0), (100, 0), (100, 100), (0, 100)), 10)
ZONE_B = ZoneConfig("b", ((100, 0), (200, 0), (200, 100), (100, 100)), 10)
CAM1 = CameraConfig("cam01", "rtsp://x", (ZONE_A,))
CAM2 = CameraConfig("cam02", "rtsp://y", (ZONE_B,))
T = datetime(2026, 9, 20, 14, 30, tzinfo=KST)


def _frames(n=5):
    # 노이즈 프레임: 블러 검증이 실제로 동작하게
    return [np.random.default_rng(i).integers(0, 256, (100, 200, 3), dtype=np.uint8) for i in range(n)]


def _dets(n_people, n=5):
    # 구역 a 안, 발 위치 (50, 90)
    boxes = np.array([[40, 30, 60, 90]] * n_people, dtype=np.float32).reshape(-1, 4)
    return [FrameDetections(boxes, np.full(len(boxes), 0.9, np.float32)) for _ in range(n)]


def _pipeline(cameras=(CAM1,), **service):
    r = fakeredis.FakeRedis()
    return Pipeline(ServiceConfig(**service), list(cameras), detector=object(), redis_client=r), r


def _zone(r, zid="a"):
    return json.loads(r.get(f"yoba:zone:{zid}"))


def test_postprocess_publishes_zone_and_image():
    p, r = _pipeline()
    frames = _frames()
    p.postprocess(DetectItem("cam01", T, frames, _dets(5)))
    z = _zone(r)
    assert z["status"] == "ok" and z["level"] == 1 and z["score"] == 0.5 and z["confidence"] == "high"
    assert z["image_url"].startswith("/snapshots/a.jpg?v=")
    assert r.get("yoba:img:a")[:2] == b"\xff\xd8"
    assert r.get("yoba:heartbeat") is not None
    assert frames == []  # 원본 프레임 폐기


def test_too_few_frames_is_error():
    p, r = _pipeline()
    p.postprocess(DetectItem("cam01", T, _frames(2), []))
    assert _zone(r)["status"] == "error"


def test_inference_failure_is_error():
    p, r = _pipeline()
    p.postprocess(DetectItem("cam01", T, _frames(), []))  # 추론 실패 시 detections는 빈 리스트
    assert _zone(r)["status"] == "error"


def test_blur_failure_sends_numbers_only(monkeypatch):
    monkeypatch.setattr(pl, "privacy_blur", lambda *a: None)
    p, r = _pipeline()
    p.postprocess(DetectItem("cam01", T, _frames(), _dets(3)))
    z = _zone(r)
    assert z["status"] == "ok" and z["image_url"] is None
    assert r.get("yoba:img:a") is None


def test_low_cycle_rewrites_previous():
    p, r = _pipeline()
    p.postprocess(DetectItem("cam01", T, _frames(), _dets(5)))
    unstable = [d for n in (0, 0, 9, 0, 1) for d in _dets(n, 1)]
    p.postprocess(DetectItem("cam01", datetime(2026, 9, 20, 14, 31, tzinfo=KST), _frames(), unstable))
    z = _zone(r)
    assert z["confidence"] == "low" and z["score"] == 0.5
    assert z["updated_at"] == "2026-09-20T14:30:00+09:00"


def test_full_queue_drops_oldest():
    p, _ = _pipeline(queue_size=2)
    for i in range(3):
        p._put_latest(p._frame_q, pl.CaptureItem(f"c{i}", T, []))
    assert [p._frame_q.get_nowait().camera_id for _ in range(2)] == ["c1", "c2"]


class FakeStream:
    def __init__(self, camera, cfg):
        self.camera = camera
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeDetector:
    def detect(self, frames):
        return _dets(5, len(frames))


def test_run_staggers_cameras_and_publishes(monkeypatch):
    calls = []

    def fake_burst(stream, cfg):
        calls.append((stream.camera.id, time.monotonic()))
        return _frames()

    monkeypatch.setattr(pl, "capture_burst", fake_burst)
    r = fakeredis.FakeRedis()
    service = ServiceConfig(cycle_seconds=1.0, capture=CaptureConfig(window_seconds=0.2))
    streams = []

    def factory(camera, cfg):
        streams.append(FakeStream(camera, cfg))
        return streams[-1]

    p = Pipeline(service, [CAM1, CAM2], detector=FakeDetector(), redis_client=r, stream_factory=factory)
    t = threading.Thread(target=p.run)
    t.start()
    time.sleep(2.3)
    p.stop()
    t.join(timeout=10)
    assert not t.is_alive()

    first = {cid: ts for cid, ts in reversed(calls)}  # 카메라별 첫 캡처 시각
    assert 0.4 <= first["cam02"] - first["cam01"] <= 0.6  # 10초/N 대신 1초/2대 = 0.5초 간격
    assert sum(cid == "cam01" for cid, _ in calls) >= 2  # 주기마다 반복
    assert all(s.started and s.stopped for s in streams)
    assert _zone(r, "a")["status"] == "ok" and _zone(r, "b")["status"] == "ok"
