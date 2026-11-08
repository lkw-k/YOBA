"""뼈대 검증: 모듈 import, 설정 파일 로드, 설정 검증."""

import importlib
from pathlib import Path

import pytest

from ywhob.config import ConfigError, load_cameras, load_service
from ywhob.main import main

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "configs" / "service.yaml"
CAMERAS = ROOT / "configs" / "cameras.example.yaml"


@pytest.mark.parametrize(
    "module", ["aggregate", "blur", "capture", "detect", "matching", "pipeline", "publish"]
)
def test_modules_import(module):
    importlib.import_module(f"ywhob.{module}")


def test_service_yaml_matches_spec():
    cfg = load_service(SERVICE)
    assert cfg.cycle_seconds == 10
    assert cfg.capture.frames == 5 and cfg.capture.window_seconds == 3.0
    assert cfg.detector.max_det == 1000
    assert cfg.aggregate.up == (0.45, 0.75) and cfg.aggregate.down == (0.35, 0.65)


def test_cameras_example_loads():
    cams = load_cameras(CAMERAS)
    assert [z.zone_id for z in cams[0].zones] == ["hall_a_front", "hall_a_back"]


def test_hysteresis_must_have_gap(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("aggregate:\n  up: [0.45, 0.75]\n  down: [0.50, 0.65]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_service(p)


def test_unknown_key_reported(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("detector:\n  persn_conf: 0.3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="detector"):
        load_service(p)


def test_duplicate_zone_id_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    zone = "{zone_id: z1, roi: [[0,0],[1,0],[1,1]], capacity: 10}"
    p.write_text(
        f"cameras:\n  - {{id: a, url: x, zones: [{zone}]}}\n  - {{id: b, url: y, zones: [{zone}]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_cameras(p)


def test_cli_check_only():
    main(["--service", str(SERVICE), "--cameras", str(CAMERAS), "--check"])


def test_model_alias_resolved():
    det = load_service(SERVICE).detector
    assert det.model == "models/yolo26s-ywhob-v1.pt"  # 기본 s
    assert det.with_model("n").model == "models/yolo26n-ywhob-v1.pt"
    assert det.with_model("models/other.onnx").model == "models/other.onnx"  # 별칭이 아니면 경로


def test_cli_missing_files_exit(tmp_path):
    with pytest.raises(SystemExit):
        main(["--service", str(SERVICE), "--cameras", str(tmp_path / "none.yaml")])
    with pytest.raises(SystemExit):
        main(["--service", str(SERVICE), "--cameras", str(CAMERAS), "--model", str(tmp_path / "none.pt")])


@pytest.mark.parametrize(
    "capture", ["rtsp_transport: http", "read_timeout: 0", "open_timeout: -1"]
)
def test_invalid_capture_config_rejected(tmp_path, capture):
    p = tmp_path / "s.yaml"
    p.write_text(f"capture:\n  {capture}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_service(p)


@pytest.mark.parametrize(
    "blur",["box_pad: -0.1", "box_long_side: 0", "sharpness_ratio: 1.0", "sharpness_ratio: 0", "downscale: 8"]
)
def test_invalid_blur_config_rejected(tmp_path, blur):
    p = tmp_path / "s.yaml"
    p.write_text(f"blur:\n  {blur}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_service(p)
