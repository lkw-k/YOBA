"""Redis 전송 (spec 5.8, 9).

- 이미지(ywhob:img:{zone_id}, TTL)를 먼저, 구역 JSON(ywhob:zone:{zone_id})을 나중에 쓴다 (한 트랜잭션, 순서 유지)
- image_url에 ?v={unix_ts} 버전 쿼리
- low: 직전 값을 confidence low로, updated_at 유지 / error: status error
- Redis 장애는 로그만 남기고 False. 쌓아 두지 않고 다음 주기에 최신 값을 다시 쓴다
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

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
