import importlib.util
from pathlib import Path

import numpy as np

from ywhob.config import load_cameras

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("roi_editor", ROOT / "scripts" / "roi_editor.py")
roi_editor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi_editor)


def test_output_loads_as_cameras_yaml(tmp_path):
    zones = [{"zone_id": "hall_a", "roi": [[0, 0], [100, 0], [100, 80]], "capacity": 30.0}]
    masks = [[[0, 0], [100, 0], [100, 10]]]
    p = tmp_path / "cameras.yaml"
    p.write_text(roi_editor.camera_yaml("cam01", "rtsp://x", zones, masks), encoding="utf-8")
    cam = load_cameras(p)[0]
    assert cam.id == "cam01" and cam.zones[0].zone_id == "hall_a" and cam.zones[0].capacity == 30
    assert cam.zones[0].roi == ((0, 0), (100, 0), (100, 80))
    assert len(cam.mask) == 1


def test_pixelate_keeps_size_and_removes_detail():
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    out = roi_editor.pixelate(frame, 240)
    assert out.shape == frame.shape
    assert (out[:8, :8] == out[0, 0]).all()  # 8×8 블록이 한 색
