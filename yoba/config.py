"""설정 파일(YAML) 로딩과 검증."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class ZoneConfig:
    zone_id: str
    roi: Polygon
    capacity: float


@dataclass(frozen=True)
class CameraConfig:
    id: str
    url: str
    zones: tuple[ZoneConfig, ...]
    mask: tuple[Polygon, ...] = ()


@dataclass(frozen=True)
class CaptureConfig:
    frames: int = 5
    window_seconds: float = 3.0
    min_frames: int = 3
    reconnect_max_backoff: float = 30.0
    rtsp_transport: str = "tcp"
    open_timeout: float = 10.0
    read_timeout: float = 10.0


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "models/yolo26s-yoba-v1.pt"
    imgsz: int = 1280
    person_conf: float = 0.35
    max_det: int = 1000
    device: str | int | None = None
    models: dict[str, str] = field(default_factory=dict)  # 별칭 → 가중치 경로 (예: s, n)
    download_url: str | None = None  # 별칭 파일이 없을 때 받을 주소 (뒤에 파일 이름을 붙임)

    def with_model(self, name: str) -> DetectorConfig:
        """별칭(models의 키)이나 가중치 경로로 모델을 고른다."""
        return replace(self, model=self.models.get(name, name))


@dataclass(frozen=True)
class ConfidenceRule:
    """편차 허용치: (max - min) <= max(abs, rel * median)."""

    abs: float
    rel: float


@dataclass(frozen=True)
class AggregateConfig:
    high: ConfidenceRule = ConfidenceRule(2, 0.2)
    medium: ConfidenceRule = ConfidenceRule(4, 0.5)
    ema_alpha: float = 0.4
    up: tuple[float, float] = (0.45, 0.75)
    down: tuple[float, float] = (0.35, 0.65)
    confirm_cycles: int = 2


@dataclass(frozen=True)
class BlurConfig:
    box_pad: float = 0.1
    box_long_side: int = 16
    output_long_side: int = 960
    sharpness_max: float = 30.0
    sharpness_ratio: float = 0.25
    jpeg_quality: int = 80


@dataclass(frozen=True)
class RedisConfig:
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "yoba"
    image_ttl: int = 120
    image_url_base: str = "/snapshots"
    socket_timeout: float = 5.0
    live_frame_ttl_ms: int = 5000


@dataclass(frozen=True)
class ServiceConfig:
    cycle_seconds: float = 10.0
    queue_size: int = 4
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)


class ConfigError(ValueError):
    pass


def _read_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: 최상위가 매핑이 아님")
    return data


def _build(cls, raw: dict | None, where: str):
    """dataclass 생성. 오타 등 모르는 키는 ConfigError로 알려준다."""
    try:
        return cls(**(raw or {}))
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def _polygon(raw, where: str) -> Polygon:
    try:
        pts = tuple((float(x), float(y)) for x, y in raw)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: 다각형 좌표 형식 오류 ({e})") from e
    if len(pts) < 3:
        raise ConfigError(f"{where}: 다각형은 점이 3개 이상이어야 함")
    return pts


def _aggregate(raw: dict) -> AggregateConfig:
    conf = raw.get("confidence", {})
    cfg = AggregateConfig(
        high=_build(ConfidenceRule, conf["high"], "aggregate.confidence.high") if "high" in conf else AggregateConfig.high,
        medium=_build(ConfidenceRule, conf["medium"], "aggregate.confidence.medium") if "medium" in conf else AggregateConfig.medium,
        ema_alpha=float(raw.get("ema_alpha", AggregateConfig.ema_alpha)),
        up=tuple(raw.get("up", AggregateConfig.up)),
        down=tuple(raw.get("down", AggregateConfig.down)),
        confirm_cycles=int(raw.get("confirm_cycles", AggregateConfig.confirm_cycles)),
    )
    if len(cfg.up) != 2 or len(cfg.down) != 2:
        raise ConfigError("aggregate.up/down: 값이 2개여야 함")
    if not (cfg.up[0] < cfg.up[1] and cfg.down[0] < cfg.down[1]):
        raise ConfigError("aggregate.up/down: 오름차순이어야 함")
    if not all(d < u for d, u in zip(cfg.down, cfg.up)):
        raise ConfigError("aggregate: 하강 임계값은 같은 경계의 상승 임계값보다 작아야 함 (히스테리시스)")
    if not 0 < cfg.ema_alpha <= 1:
        raise ConfigError("aggregate.ema_alpha: 0 < α <= 1")
    if cfg.confirm_cycles < 1:
        raise ConfigError("aggregate.confirm_cycles: 1 이상")
    if not (cfg.high.abs <= cfg.medium.abs and cfg.high.rel <= cfg.medium.rel):
        raise ConfigError("aggregate.confidence: high 허용치가 medium보다 클 수 없음")
    return cfg


def _detector(raw: dict | None) -> DetectorConfig:
    cfg = _build(DetectorConfig, raw, "detector")
    models = {str(k): str(v) for k, v in (cfg.models or {}).items()}
    return replace(cfg, models=models).with_model(cfg.model)


def load_service(path: str | Path) -> ServiceConfig:
    raw = _read_yaml(path)
    cfg = ServiceConfig(
        cycle_seconds=float(raw.get("cycle_seconds", ServiceConfig.cycle_seconds)),
        queue_size=int(raw.get("queue_size", ServiceConfig.queue_size)),
        capture=_build(CaptureConfig, raw.get("capture"), "capture"),
        detector=_detector(raw.get("detector")),
        aggregate=_aggregate(raw.get("aggregate") or {}),
        blur=_build(BlurConfig, raw.get("blur"), "blur"),
        redis=_build(RedisConfig, raw.get("redis"), "redis"),
    )
    if os.environ.get("YOBA_REDIS_URL"):  # 비밀번호가 든 주소는 git에 올라가는 yaml 대신 .env로 받는다
        cfg = replace(cfg, redis=replace(cfg.redis, url=os.environ["YOBA_REDIS_URL"]))
    cap = cfg.capture
    if not 1 <= cap.min_frames <= cap.frames:
        raise ConfigError("capture.min_frames: 1 이상, frames 이하")
    if cap.window_seconds >= cfg.cycle_seconds:
        raise ConfigError("capture.window_seconds: 측정 주기보다 짧아야 함")
    if cap.rtsp_transport not in ("tcp", "udp"):
        raise ConfigError("capture.rtsp_transport: tcp 또는 udp")
    if cap.open_timeout <= 0 or cap.read_timeout <= 0:
        raise ConfigError("capture.open_timeout/read_timeout: 0보다 커야 함")
    if cfg.redis.socket_timeout <= 0:
        raise ConfigError("redis.socket_timeout: 0보다 커야 함")
    if cfg.redis.live_frame_ttl_ms <= 0:
        raise ConfigError("redis.live_frame_ttl_ms: 0보다 커야 함")
    if cfg.queue_size < 1:
        raise ConfigError("queue_size: 1 이상")
    if cfg.blur.box_pad < 0:
        raise ConfigError("blur.box_pad: 0 이상")
    if cfg.blur.box_long_side < 1:
        raise ConfigError("blur.box_long_side: 1 이상")
    if not 0 < cfg.blur.sharpness_ratio < 1:
        raise ConfigError("blur.sharpness_ratio: 0 < r < 1 (1 이상이면 블러 안 된 박스도 통과)")
    return cfg


def load_cameras(path: str | Path) -> list[CameraConfig]:
    raw = _read_yaml(path)
    cameras: list[CameraConfig] = []
    cam_ids: set[str] = set()
    zone_ids: set[str] = set()
    for i, c in enumerate(raw.get("cameras") or []):
        where = f"cameras[{i}]"
        cam_id = str(c["id"])
        if cam_id in cam_ids:
            raise ConfigError(f"{where}: 카메라 id 중복 ({cam_id})")
        cam_ids.add(cam_id)

        zones = []
        for j, z in enumerate(c.get("zones") or []):
            zid = str(z["zone_id"])
            if zid in zone_ids:
                raise ConfigError(f"{where}.zones[{j}]: zone_id 중복 ({zid})")
            zone_ids.add(zid)
            capacity = float(z["capacity"])
            if capacity <= 0:
                raise ConfigError(f"{where}.zones[{j}]: capacity는 0보다 커야 함")
            zones.append(ZoneConfig(zid, _polygon(z["roi"], f"{where}.zones[{j}].roi"), capacity))
        if not zones:
            raise ConfigError(f"{where}: 구역이 하나 이상 있어야 함")

        masks = tuple(_polygon(p, f"{where}.mask[{k}]") for k, p in enumerate(c.get("mask") or []))
        url = str(c["url"])
        if url.lower().startswith("webcam:") and not url[len("webcam:"):].isdigit():
            raise ConfigError(f"{where}.url: 웹캠은 webcam:번호 형식 (예: webcam:0)")
        cameras.append(CameraConfig(cam_id, url, tuple(zones), masks))
    if not cameras:
        raise ConfigError(f"{path}: 카메라가 없음")
    return cameras
