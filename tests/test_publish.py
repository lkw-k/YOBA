import json
from datetime import datetime

import fakeredis
import pytest

from yoba.config import RedisConfig
from yoba.publish import KST, Publisher

T1 = datetime(2026, 9, 20, 14, 30, 0, tzinfo=KST)
T2 = datetime(2026, 9, 20, 14, 30, 30, tzinfo=KST)


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def _zone(r, zid="hall_a"):
    return json.loads(r.get(f"yoba:zone:{zid}"))


def test_publish_writes_image_and_json(r):
    p = Publisher(r, RedisConfig())
    assert p.publish("hall_a", 1, 0.5234, "high", b"\xff\xd8jpeg", T1)
    z = _zone(r)
    assert z == {
        "zone_id": "hall_a",
        "level": 1,
        "score": 0.523,
        "confidence": "high",
        "image_url": f"/snapshots/hall_a.jpg?v={int(T1.timestamp())}",
        "updated_at": "2026-09-20T14:30:00+09:00",
        "status": "ok",
    }
    assert r.get("yoba:img:hall_a") == b"\xff\xd8jpeg"
    assert 0 < r.ttl("yoba:img:hall_a") <= 120
    assert r.smembers("yoba:zones") == {b"hall_a"}


def test_image_written_before_json(r):
    calls = []
    real = r.pipeline

    def spy():
        p = real()
        orig = p.set

        def set_(key, *a, **kw):
            calls.append(key)
            return orig(key, *a, **kw)

        p.set = set_
        return p

    r.pipeline = spy
    Publisher(r, RedisConfig()).publish("hall_a", 1, 0.5, "high", b"img", T1)
    assert calls == ["yoba:img:hall_a", "yoba:zone:hall_a"]


def test_no_image_gives_null_url(r):
    Publisher(r, RedisConfig()).publish("hall_a", 0, 0.1, "medium", None, T1)
    assert _zone(r)["image_url"] is None
    assert r.get("yoba:img:hall_a") is None


def test_low_rewrites_previous_with_old_timestamp(r):
    p = Publisher(r, RedisConfig())
    p.publish("hall_a", 2, 0.8, "high", b"img", T1)
    assert p.publish_low("hall_a")
    z = _zone(r)
    assert z["confidence"] == "low" and z["level"] == 2 and z["score"] == 0.8
    assert z["updated_at"] == "2026-09-20T14:30:00+09:00"  # 갱신하지 않음
    assert z["status"] == "ok"


def test_low_without_previous_writes_nothing(r):
    assert not Publisher(r, RedisConfig()).publish_low("hall_a")
    assert r.get("yoba:zone:hall_a") is None


def test_error_status(r):
    p = Publisher(r, RedisConfig())
    p.publish("hall_a", 1, 0.5, "high", b"img", T1)
    p.publish_error("hall_a", T2)
    z = _zone(r)
    assert z["status"] == "error" and z["image_url"] is None
    assert z["updated_at"] == "2026-09-20T14:30:30+09:00"
    # error 뒤 low는 마지막 정상 값을 다시 쓴다
    p.publish_low("hall_a")
    assert _zone(r)["status"] == "ok"


def test_redis_down_does_not_raise():
    server = fakeredis.FakeServer()
    server.connected = False
    p = Publisher(fakeredis.FakeRedis(server=server), RedisConfig())
    assert not p.publish("hall_a", 1, 0.5, "high", b"img", T1)
    assert not p.publish_error("hall_a", T1)
    p.heartbeat(T1)
    server.connected = True
    assert p.publish_low("hall_a")  # 실패한 주기의 측정값도 기억해 둬서 복구 후 low 재기록 가능
    assert _zone(fakeredis.FakeRedis(server=server))["level"] == 1


def test_heartbeat(r):
    Publisher(r, RedisConfig()).heartbeat(T1)
    assert r.get("yoba:heartbeat") == b"2026-09-20T14:30:00+09:00"


# --- 실시간 모드 (spec 9.4) ---

from yoba.config import CameraConfig, ZoneConfig  # noqa: E402
from yoba.publish import LivePublisher  # noqa: E402

LIVE_CAM = CameraConfig("cam01", "rtsp://x/1", (ZoneConfig("front", ((0, 0), (100.4, 0), (100, 50)), 5),))


def test_live_publish_writes_json_frame_and_camera_set(r):
    lp = LivePublisher(r, RedisConfig(), LIVE_CAM)
    assert lp.publish((1280, 720), {"front": 3.0}, b"\xff\xd8jpeg", 7.84, 180.4)
    live = json.loads(r.get("yoba:live:cam01"))
    assert {k: live[k] for k in ("camera_id", "frame_size", "fps", "latency_ms", "blur_ok",
                                  "blur_fail_total", "frames_total")} == {
        "camera_id": "cam01", "frame_size": [1280, 720], "fps": 7.8, "latency_ms": 180,
        "blur_ok": True, "blur_fail_total": 0, "frames_total": 1,
    }
    assert live["zones"] == [{"zone_id": "front", "count": 3, "roi": [[0, 0], [100, 0], [100, 50]]}]
    assert datetime.fromisoformat(live["updated_at"]).utcoffset() == KST.utcoffset(None)
    assert r.get("yoba:frame:cam01") == b"\xff\xd8jpeg"
    assert 0 < r.pttl("yoba:frame:cam01") <= 5000
    assert r.smembers("yoba:cameras") == {b"cam01"}
    assert r.get("yoba:heartbeat")
    assert r.get("yoba:zone:front") is None  # 10초 모드 키는 건드리지 않는다


def test_live_publish_blur_failure_sends_black_jpeg(r):
    lp = LivePublisher(r, RedisConfig(), LIVE_CAM)
    lp.publish((1280, 720), {"front": 2}, None, 5.0, 100)
    live = json.loads(r.get("yoba:live:cam01"))
    assert live["blur_ok"] is False and live["blur_fail_total"] == 1
    assert live["zones"][0]["count"] == 2  # 사람 수는 그대로 보낸다
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(r.get("yoba:frame:cam01"), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.max() < 10  # 검은 화면 (JPEG 오차 허용)


def test_live_publish_redis_failure_logs_once(caplog):
    import redis

    class Broken:
        def pipeline(self):
            raise redis.ConnectionError

    lp = LivePublisher(Broken(), RedisConfig(), LIVE_CAM)
    with caplog.at_level("WARNING"):
        assert not lp.publish((1280, 720), {"front": 1}, b"j", 5.0, 100)
        assert not lp.publish((1280, 720), {"front": 1}, b"j", 5.0, 100)
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    assert lp.frames_total == 2
