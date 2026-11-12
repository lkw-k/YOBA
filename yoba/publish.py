"""Redis 전송 (spec 5.8, 9).

- 이미지(yoba:img:{zone_id}, TTL)를 먼저, 구역 JSON(yoba:zone:{zone_id})을 나중에 쓴다 (한 트랜잭션, 순서 유지)
- image_url에 ?v={unix_ts} 버전 쿼리
- low: 직전 값을 confidence low로, updated_at 유지 / error: status error
- Redis 장애는 로그만 남기고 False. 쌓아 두지 않고 다음 주기에 최신 값을 다시 쓴다
- LivePublisher: 실시간 모드(--live --publish)의 카메라별 매 프레임 전송 (spec 9.4)
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import redis

from .config import RedisConfig

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


class Publisher:
    def __init__(self, client, cfg: RedisConfig):
        self.r = client
        self.cfg = cfg
        self._last_ok: dict[str, dict] = {}  # 구역별 마지막 정상 JSON (low 재기록용)

    def _key(self, kind: str, zone_id: str) -> str:
        return f"{self.cfg.key_prefix}:{kind}:{zone_id}"

    def _write(self, zone_id: str, payload: dict, image: bytes | None = None) -> bool:
        try:
            p = self.r.pipeline()
            if image is not None:
                p.set(self._key("img", zone_id), image, ex=self.cfg.image_ttl)
            p.set(self._key("zone", zone_id), json.dumps(payload, ensure_ascii=False))
            p.sadd(f"{self.cfg.key_prefix}:zones", zone_id)
            p.execute()
            return True
        except redis.RedisError as e:
            log.warning("%s: Redis 전송 실패 (%s)", zone_id, type(e).__name__)
            return False

    def publish(
        self,
        zone_id: str,
        level: int,
        score: float,
        confidence: str,
        image: bytes | None,
        updated_at: datetime,
    ) -> bool:
        ts = updated_at.astimezone(KST)
        image_url = None
        if image is not None:
            image_url = f"{self.cfg.image_url_base}/{zone_id}.jpg?v={int(ts.timestamp())}"
        payload = {
            "zone_id": zone_id,
            "level": int(level),
            "score": round(float(score), 3),
            "confidence": confidence,
            "image_url": image_url,
            "updated_at": ts.isoformat(timespec="seconds"),
            "status": "ok",
        }
        ok = self._write(zone_id, payload, image)
        self._last_ok[zone_id] = payload  # Redis가 실패해도 측정값은 최신으로 둔다
        return ok

    def publish_low(self, zone_id: str) -> bool:
        """직전 정상 값을 confidence low로 다시 쓴다. updated_at은 그대로라 3주기 연속이면 웹에서 stale."""
        last = self._last_ok.get(zone_id)
        if last is None:
            return False  # 아직 유효 주기가 없으면 다시 쓸 값이 없다
        return self._write(zone_id, {**last, "confidence": "low"})

    def publish_error(self, zone_id: str, updated_at: datetime) -> bool:
        """카메라 장애나 캡처 부족. 직전 수치가 있으면 남기되 status error로 웹이 여유로 표시하지 않게 한다."""
        last = self._last_ok.get(zone_id, {})
        payload = {
            "zone_id": zone_id,
            "level": last.get("level"),
            "score": last.get("score"),
            "confidence": "low",
            "image_url": None,
            "updated_at": updated_at.astimezone(KST).isoformat(timespec="seconds"),
            "status": "error",
        }
        return self._write(zone_id, payload)

    def heartbeat(self, at: datetime) -> None:
        try:
            self.r.set(f"{self.cfg.key_prefix}:heartbeat", at.astimezone(KST).isoformat(timespec="seconds"))
        except redis.RedisError as e:
            log.warning("heartbeat 전송 실패 (%s)", type(e).__name__)


class LivePublisher:
    """실시간 모드 전송 (spec 9.4). 카메라 1대의 매 프레임 결과를 yoba:live / yoba:frame에 덮어쓴다.

    블러 실패 프레임은 검은 JPEG로 대신하고 blur_ok false. 기존 10초 모드 키는 건드리지 않는다.
    """

    def __init__(self, client, cfg: RedisConfig, camera):
        self.r = client
        self.cfg = cfg
        self.camera = camera
        self.frames_total = 0
        self.blur_fail_total = 0
        self._failing = False  # 초당 여러 번 실패해도 경고는 상태가 바뀔 때만 남긴다
        self._black = cv2.imencode(".jpg", np.zeros((540, 960, 3), np.uint8))[1].tobytes()

    def publish(self, frame_size: tuple[int, int], counts: dict[str, float], jpeg: bytes | None,
                fps: float, latency_ms: float) -> bool:
        self.frames_total += 1
        if jpeg is None:
            self.blur_fail_total += 1
        now = datetime.now(KST)
        payload = {
            "camera_id": self.camera.id,
            "updated_at": now.isoformat(timespec="milliseconds"),
            "frame_size": list(frame_size),
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms),
            "blur_ok": jpeg is not None,
            "blur_fail_total": self.blur_fail_total,
            "frames_total": self.frames_total,
            "zones": [
                {"zone_id": z.zone_id, "count": int(counts[z.zone_id]), "roi": [[round(x), round(y)] for x, y in z.roi]}
                for z in self.camera.zones
            ],
        }
        p = self.cfg.key_prefix
        try:
            pipe = self.r.pipeline()
            pipe.set(f"{p}:frame:{self.camera.id}", jpeg if jpeg is not None else self._black,
                     px=self.cfg.live_frame_ttl_ms)
            pipe.set(f"{p}:live:{self.camera.id}", json.dumps(payload, ensure_ascii=False))
            pipe.sadd(f"{p}:cameras", self.camera.id)
            pipe.set(f"{p}:heartbeat", now.isoformat(timespec="seconds"))
            pipe.execute()
        except redis.RedisError as e:
            if not self._failing:
                log.warning("%s: 실시간 Redis 전송 실패 (%s). 복구될 때까지 경고를 생략", self.camera.id, type(e).__name__)
            self._failing = True
            return False
        if self._failing:
            log.info("%s: 실시간 Redis 전송 복구", self.camera.id)
        self._failing = False
        return True


class LatestSender:
    """LivePublisher.publish를 별도 스레드에서 돌린다. 전송(SSH 터널 너머 Redis)이 느려도 검출·화면 루프를
    막지 않고, 전송 중에 쌓인 프레임은 버리고 가장 최근 것만 보낸다."""

    def __init__(self, publisher: LivePublisher):
        self._pub = publisher
        self._cond = threading.Condition()
        self._args: tuple | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="live-publish", daemon=True)
        self._thread.start()

    def publish(self, *args) -> None:
        with self._cond:
            self._args = args  # 아직 안 보낸 이전 프레임은 덮어쓴다
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._args is None and not self._stop:
                    self._cond.wait()
                if self._args is None:
                    return
                args, self._args = self._args, None
            self._pub.publish(*args)
