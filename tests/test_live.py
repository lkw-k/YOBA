"""실시간 미리보기: 카메라·모델·창 없이 LiveView.process만 확인한다."""

import numpy as np

from yoba import blur
from yoba.config import CameraConfig, ServiceConfig, ZoneConfig
from yoba.detect import FrameDetections
from yoba.live import LiveView

ZONE = ZoneConfig("all", ((0, 0), (1280, 0), (1280, 720), (0, 720)), 2)
CAMERA = CameraConfig("webcam", "webcam:0", (ZONE,))
BOX = np.array([[400, 100, 600, 700]], dtype=np.float32)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def detect(self, frames):
        self.calls.append(len(frames))
        return [FrameDetections(self.boxes, np.ones(len(self.boxes))) for _ in frames]


def _noise(h=720, w=1280):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_process_counts_and_scales_view():
    det = FakeDetector(BOX)
    view, results, ok = LiveView(ServiceConfig(), CAMERA, det).process(_noise())
    assert det.calls == [1]  # 매 프레임 한 장씩 추론
    assert ok and view.shape[:2] == (540, 960)  # output_long_side 960으로 축소
    assert results[0].median == 1 and results[0].score == 0.5 and results[0].level == 1
    # 사람 박스 테두리(청록)가 960/1280 = 0.75 배 위치에 그려져야 한다
    assert tuple(view[300, 300]) == (255, 255, 0)


def test_tracker_state_carries_across_frames():
    live = LiveView(ServiceConfig(), CAMERA, FakeDetector(np.zeros((0, 4), np.float32)))
    live.process(_noise())
    live.detector.boxes = np.repeat(BOX, 2, axis=0)  # 2명 = capacity
    scores = [live.process(_noise())[1][0].score for _ in range(3)]
    assert scores[0] < scores[1] < scores[2] < 1  # 한 프레임 = 한 주기로 EMA가 따라 올라간다


def test_process_hides_image_when_blur_fails(monkeypatch):
    monkeypatch.setattr(blur, "_resize_long_side", lambda img, long_side: img)  # 블러가 안 되는 상황
    view, _, ok = LiveView(ServiceConfig(), CAMERA, FakeDetector(BOX)).process(_noise())
    assert not ok and view.shape[:2] == (540, 960) and not view.any()
