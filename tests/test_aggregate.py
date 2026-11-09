import numpy as np
import pytest

from ywhob.aggregate import HIGH, LOW, MEDIUM, ZoneAssigner, ZoneTracker, classify_confidence, median_frame_index
from ywhob.config import AggregateConfig, ZoneConfig

LEFT = ZoneConfig("left", ((0, 0), (100, 0), (100, 100), (0, 100)), 10)
RIGHT = ZoneConfig("right", ((100, 0), (200, 0), (200, 100), (100, 100)), 10)
CFG = AggregateConfig()


def _box(fx, fy, w=10, h=30):
    """발 위치(fx, fy)인 사람 박스."""
    return [fx - w / 2, fy - h, fx + w / 2, fy]


def test_assign_by_foot_position():
    a = ZoneAssigner([LEFT, RIGHT])
    # 몸통은 left에 걸쳐도 발이 right면 right
    boxes = np.array([_box(50, 50), _box(150, 50), _box(105, 90, w=40), _box(50, 150)])
    assert a.count(boxes) == {"left": 1, "right": 2}  # 마지막은 어느 구역에도 없음


def test_mask_excludes_people():
    mask = ((0, 0), (100, 0), (100, 40), (0, 40))
    a = ZoneAssigner([LEFT], masks=[mask])
    assert a.count(np.array([_box(50, 20), _box(50, 80)])) == {"left": 1}


def test_overlapping_zone_counts_once():
    a = ZoneAssigner([LEFT, ZoneConfig("all", ((0, 0), (200, 0), (200, 100), (0, 100)), 10)])
    assert a.count(np.array([_box(50, 50)])) == {"left": 1, "all": 0}


def test_empty_boxes():
    assert ZoneAssigner([LEFT]).count(np.zeros((0, 4))) == {"left": 0}


@pytest.mark.parametrize(
    "counts, expected",
    [
        ([10, 11, 12, 10, 11], HIGH),  # d=2 <= max(2, 2.2)
        ([0, 0, 1, 0, 2], HIGH),  # 빈 구역: 최소 허용치 2
        ([20, 24, 26, 22, 25], MEDIUM),  # d=6 > 4.8, <= max(4, 12)
        ([0, 0, 5, 0, 1], LOW),  # d=5 > max(4, 0)
    ],
)
def test_classify_confidence(counts, expected):
    median, spread, conf = classify_confidence(counts, CFG)
    assert conf == expected
    assert median == np.median(counts) and spread == max(counts) - min(counts)


def test_median_frame_index():
    assert median_frame_index([3, 9, 5, 1, 5]) == 2


def _zone(capacity=10):
    return ZoneConfig("z", ((0, 0), (1, 0), (1, 1)), capacity)


def test_first_cycle_sets_ema_and_level():
    t = ZoneTracker(_zone(), CFG)
    r = t.update([5] * 5)
    assert r.accepted and r.score == 0.5 and r.level == 1


def test_ema_and_score_clip():
    t = ZoneTracker(_zone(), CFG)
    t.update([10] * 5)
    r = t.update([20] * 5)
    assert t.ema == pytest.approx(0.4 * 20 + 0.6 * 10)
    assert r.score == 1.0  # 14/10 → 1로 자름


def test_low_cycle_discarded():
    t = ZoneTracker(_zone(), CFG)
    t.update([5] * 5)
    r = t.update([0, 0, 9, 0, 1])
    assert not r.accepted and r.confidence == LOW
    assert r.score == 0.5 and r.level == 1 and t.ema == 5


def test_low_before_any_valid_cycle():
    r = ZoneTracker(_zone(), CFG).update([0, 0, 9, 0, 1])
    assert r.score is None and r.level is None


def test_level_change_needs_confirm_cycles():
    t = ZoneTracker(_zone(capacity=100), AggregateConfig(ema_alpha=1.0))
    assert t.update([10] * 5).level == 0
    assert t.update([50] * 5).level == 0  # 1주기째: 아직
    assert t.update([50] * 5).level == 1  # 2주기 연속
    assert t.update([10] * 5).level == 1
    assert t.update([50] * 5).level == 1  # 연속이 끊기면 처음부터


def test_hysteresis_uses_down_threshold():
    t = ZoneTracker(_zone(capacity=100), AggregateConfig(ema_alpha=1.0, confirm_cycles=1))
    t.update([50] * 5)  # 1단계
    assert t.update([40] * 5).level == 1  # 0.40: 상승 0.45 미만이지만 하강 0.35 이상
    assert t.update([34] * 5).level == 0


def test_can_skip_a_level():
    t = ZoneTracker(_zone(capacity=100), AggregateConfig(ema_alpha=1.0, confirm_cycles=1))
    t.update([10] * 5)
    assert t.update([80] * 5).level == 2
    assert t.update([10] * 5).level == 0
