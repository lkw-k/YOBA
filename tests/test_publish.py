import json
from datetime import datetime

import fakeredis
import pytest

from ywhob.config import RedisConfig
from ywhob.publish import KST, Publisher

T1 = datetime(2026, 9, 20, 14, 30, 0, tzinfo=KST)
T2 = datetime(2026, 9, 20, 14, 30, 30, tzinfo=KST)


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def _zone(r, zid="hall_a"):
    return json.loads(r.get(f"ywhob:zone:{zid}"))


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
    assert r.get("ywhob:img:hall_a") == b"\xff\xd8jpeg"
    assert 0 < r.ttl("ywhob:img:hall_a") <= 120
    assert r.smembers("ywhob:zones") == {b"hall_a"}


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
    assert calls == ["ywhob:img:hall_a", "ywhob:zone:hall_a"]


def test_no_image_gives_null_url(r):
    Publisher(r, RedisConfig()).publish("hall_a", 0, 0.1, "medium", None, T1)
    assert _zone(r)["image_url"] is None
    assert r.get("ywhob:img:hall_a") is None


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
    assert r.get("ywhob:zone:hall_a") is None


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
    assert r.get("ywhob:heartbeat") == b"2026-09-20T14:30:00+09:00"
