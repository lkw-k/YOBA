import numpy as np
import pytest

from ywhob.config import DetectorConfig
from ywhob.detect import YoloDetector


class _T:
    """torch 텐서 대역 (.cpu().numpy())."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self):
        return len(self.conf.a)


class _Result:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _Boxes(np.reshape(xyxy, (-1, 4)), conf, cls)


class FakeModel:
    def __init__(self, names=None, results=None):
        self.names = names or {0: "person", 1: "bicycle"}
        self.results = results or []
        self.kwargs = None
        self.n_frames = None

    def predict(self, frames, **kwargs):
        self.n_frames = len(frames)
        self.kwargs = kwargs
        return self.results


FRAMES = [np.zeros((8, 8, 3), np.uint8)] * 2


def test_batch_predict_with_service_settings():
    model = FakeModel(results=[_Result([], [], []), _Result([], [], [])])
    YoloDetector(DetectorConfig(person_conf=0.35, max_det=1000, imgsz=1280), model=model).detect(FRAMES)
    assert model.n_frames == 2  # 한 번에 배치로
    assert model.kwargs["classes"] == [0]
    assert model.kwargs["conf"] == 0.35 and model.kwargs["max_det"] == 1000 and model.kwargs["imgsz"] == 1280


def test_converts_boxes_and_keeps_only_person():
    r = _Result([[0, 0, 10, 20], [5, 5, 6, 6]], [0.9, 0.8], [0, 1])
    model = FakeModel(results=[r, _Result([], [], [])])
    out = YoloDetector(DetectorConfig(), model=model).detect(FRAMES)
    assert out[0].persons.tolist() == [[0, 0, 10, 20]]
    assert out[0].person_scores.tolist() == pytest.approx([0.9])
    assert out[1].persons.shape == (0, 4)


def test_rejects_model_without_person_class0():
    with pytest.raises(ValueError, match="person"):
        YoloDetector(DetectorConfig(), model=FakeModel(names={0: "head"}))
