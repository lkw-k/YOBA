"""뼈대 검증: 모듈 import, 설정 파일 로드, 설정 검증."""

import importlib
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from yoba.config import ConfigError, load_cameras, load_service
from yoba.main import download_model, main

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "configs" / "service.yaml"
CAMERAS = ROOT / "configs" / "cameras.example.yaml"


@pytest.mark.parametrize(
    "module", ["aggregate", "blur", "capture", "detect", "matching", "pipeline", "publish"]
)
def test_modules_import(module):
    importlib.import_module(f"yoba.{module}")


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
    assert det.model == "models/yolo26s-yoba-v1.pt"  # 기본 s
    assert det.with_model("n").model == "models/yolo26n-yoba-v1.pt"
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


def test_cli_webcam_uses_webcam_config(tmp_path, monkeypatch):
    import yoba.main as cli

    monkeypatch.chdir(ROOT)  # 기본 설정 경로는 저장소 루트 기준
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    seen = {}

    class FakePipeline:
        def __init__(self, service, cameras):
            seen["cameras"] = cameras

        def run(self):
            pass

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    main(["--service", str(SERVICE), "--webcam", "1", "--model", str(model)])
    cams = seen["cameras"]
    assert [c.url for c in cams] == ["webcam:1"]
    assert cams[0].zones[0].zone_id == "webcam_all"


def test_webcam_config_loads():
    cams = load_cameras(ROOT / "configs" / "cameras.webcam.yaml")
    assert cams[0].url == "webcam:0"


def test_invalid_webcam_url_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "cameras:\n  - id: w\n    url: webcam:abc\n    zones:\n"
        "      - {zone_id: z, roi: [[0,0],[1,0],[1,1]], capacity: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_cameras(p)


def test_cli_live_runs_preview_instead_of_pipeline(tmp_path, monkeypatch):
    import yoba.main as cli

    monkeypatch.chdir(ROOT)
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    seen = {}
    monkeypatch.setattr(cli, "run_live", lambda service, camera: seen.setdefault("camera", camera))
    monkeypatch.setattr(cli, "Pipeline", None)  # 호출되면 실패
    main(["--service", str(SERVICE), "--webcam", "--live", "--model", str(model)])
    assert seen["camera"].url == "webcam:0"


def test_cli_live_rejects_multiple_cameras(tmp_path):
    p = tmp_path / "c.yaml"
    zone = "      - {zone_id: %s, roi: [[0,0],[1,0],[1,1]], capacity: 1}\n"
    p.write_text(
        "cameras:\n"
        "  - id: a\n    url: webcam:0\n    zones:\n" + zone % "z1"
        + "  - id: b\n    url: webcam:1\n    zones:\n" + zone % "z2",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["--service", str(SERVICE), "--cameras", str(p), "--live"])


def test_cli_webcam_rejects_multiple_cameras(tmp_path):
    p = tmp_path / "c.yaml"
    zone = "      - {zone_id: %s, roi: [[0,0],[1,0],[1,1]], capacity: 1}\n"
    p.write_text(
        "cameras:\n"
        "  - id: a\n    url: rtsp://x/1\n    zones:\n" + zone % "z1"
        + "  - id: b\n    url: rtsp://x/2\n    zones:\n" + zone % "z2",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):  # 두 카메라가 같은 웹캠을 열게 되므로 막는다
        main(["--service", str(SERVICE), "--cameras", str(p), "--webcam", "--check"])


class _FakeResponse:
    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_model_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _FakeResponse(b"weights"))
    dest = tmp_path / "models" / "m.pt"
    assert download_model("https://example/m.pt", dest)
    assert dest.read_bytes() == b"weights"
    assert not (tmp_path / "models" / "m.pt.part").exists()


def test_download_model_failure_leaves_nothing(tmp_path, monkeypatch):
    def fail(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    dest = tmp_path / "m.pt"
    assert not download_model("https://example/m.pt", dest)
    assert list(tmp_path.iterdir()) == []


def test_cli_downloads_missing_alias_model(tmp_path, monkeypatch):
    model = tmp_path / "s.pt"
    service = tmp_path / "service.yaml"
    service.write_text(
        SERVICE.read_text(encoding="utf-8").replace("models/yolo26s-yoba-v1.pt", model.as_posix()),
        encoding="utf-8",
    )
    urls = []

    def fake_download(url, dest):
        urls.append(url)
        dest.write_bytes(b"w")
        return True

    monkeypatch.setattr("yoba.main.download_model", fake_download)
    monkeypatch.setattr("yoba.main.run_live", lambda *a: None)
    main(["--service", str(service), "--webcam", "--live"])
    assert urls == ["https://huggingface.co/illimax/YolPe/resolve/main/s.pt"]
