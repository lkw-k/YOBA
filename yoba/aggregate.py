"""구역 배정, 5장 집계, 신뢰도, EMA, 히스테리시스 판정 (spec 5.4 ~ 5.6).

- ZoneAssigner: 사람 박스의 발 위치(하단 중심)로 구역 배정, mask 영역 제외
- classify_confidence: 중앙값, 편차(max-min), high/medium/low
- ZoneTracker: EMA, score = EMA / capacity, 히스테리시스 + confirm_cycles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .config import AggregateConfig, Polygon, ZoneConfig

HIGH, MEDIUM, LOW = "high", "medium", "low"


def _contour(poly: Polygon) -> np.ndarray:
    return np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)


def _inside(contour: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0  # 경계 포함


class ZoneAssigner:
    def __init__(self, zones: Sequence[ZoneConfig], masks: Sequence[Polygon] = ()):
        self.zone_ids = [z.zone_id for z in zones]
        self._zones = [_contour(z.roi) for z in zones]
        self._masks = [_contour(m) for m in masks]

    def count(self, boxes: np.ndarray) -> dict[str, int]:
        """사람 박스 → 구역별 인원. 발 위치가 mask 안이거나 어느 구역에도 없으면 세지 않는다.
        구역이 겹치면 설정 순서상 앞 구역에 넣는다 (한 사람을 두 번 세지 않음)."""
        counts = dict.fromkeys(self.zone_ids, 0)
        for x0, _, x1, y1 in np.asarray(boxes).reshape(-1, 4):
            fx, fy = (x0 + x1) / 2, y1
            if any(_inside(m, fx, fy) for m in self._masks):
                continue
            for zid, zone in zip(self.zone_ids, self._zones):
                if _inside(zone, fx, fy):
                    counts[zid] += 1
                    break
        return counts


def classify_confidence(counts: Sequence[int], cfg: AggregateConfig) -> tuple[float, float, str]:
    """(중앙값, 편차, 신뢰도)."""
    arr = np.asarray(counts, dtype=float)
    median = float(np.median(arr))
    spread = float(arr.max() - arr.min())
    if spread <= max(cfg.high.abs, cfg.high.rel * median):
        return median, spread, HIGH
    if spread <= max(cfg.medium.abs, cfg.medium.rel * median):
        return median, spread, MEDIUM
    return median, spread, LOW


def median_frame_index(counts: Sequence[int]) -> int:
    """인원 수가 중앙값에 가장 가까운 프레임 (블러 대상). 같으면 앞 프레임."""
    arr = np.asarray(counts, dtype=float)
    return int(np.argmin(np.abs(arr - np.median(arr))))


@dataclass
class ZoneResult:
    zone_id: str
    accepted: bool  # False면 신뢰도 low로 폐기된 주기
    median: float
    spread: float
    confidence: str
    score: float | None  # 한 번도 유효 주기가 없었으면 None
    level: int | None


class ZoneTracker:
    """구역 하나의 EMA와 단계 상태."""

    def __init__(self, zone: ZoneConfig, cfg: AggregateConfig):
        self.zone = zone
        self.cfg = cfg
        self.ema: float | None = None
        self.score: float | None = None
        self.level: int | None = None
        self._pending: int | None = None  # 바뀌려는 단계
        self._pending_cycles = 0

    def _target(self, score: float) -> int:
        """현재 단계에서 score가 가리키는 단계. 올라갈 때는 up, 내려갈 때는 down만 본다."""
        up, down = self.cfg.up, self.cfg.down
        level = self.level
        if level < 2 and score >= up[1]:
            return 2
        if level < 1 and score >= up[0]:
            return 1
        if level > 0 and score < down[0]:
            return 0
        if level > 1 and score < down[1]:
            return 1
        return level

    def _initial_level(self, score: float) -> int:
        up = self.cfg.up
        return 2 if score >= up[1] else 1 if score >= up[0] else 0

    def update(self, counts: Sequence[int]) -> ZoneResult:
        median, spread, confidence = classify_confidence(counts, self.cfg)
        if confidence == LOW:  # 주기 폐기: EMA와 단계 상태를 건드리지 않는다
            return ZoneResult(self.zone.zone_id, False, median, spread, confidence, self.score, self.level)

        a = self.cfg.ema_alpha
        self.ema = median if self.ema is None else a * median + (1 - a) * self.ema
        self.score = float(np.clip(self.ema / self.zone.capacity, 0.0, 1.0))

        if self.level is None:
            # 첫 유효 주기: 비교할 이전 단계가 없으므로 상승 임계값으로 바로 정한다
            self.level = self._initial_level(self.score)
        else:
            target = self._target(self.score)
            if target == self.level:
                self._pending, self._pending_cycles = None, 0
            else:
                if target == self._pending:
                    self._pending_cycles += 1
                else:
                    self._pending, self._pending_cycles = target, 1
                if self._pending_cycles >= self.cfg.confirm_cycles:
                    self.level = target
                    self._pending, self._pending_cycles = None, 0

        return ZoneResult(self.zone.zone_id, True, median, spread, confidence, self.score, self.level)
