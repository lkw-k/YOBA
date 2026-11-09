"""사람 확정 규칙 (spec 5.3, 보류).

현재 파이프라인은 person 박스 하나를 한 명으로 세므로 이 모듈을 쓰지 않는다.
person 단독 검출의 오탐이 문제가 되어 머리 확인을 다시 도입할 때를 위해 남겨 둔다.

TODO(구현, 재도입 시):
- 머리 중심이 사람 박스 상단 top_ratio 영역 안에 있고, 머리/사람 너비 비율이 범위 안이면 후보
- 신뢰도 높은 사람부터 1:1 배정, 기대 위치(사람 상단 중앙)에 가까운 머리 우선
- 미배정 person/head 개수 반환 (로그용)
- 재도입하면 MatchParams를 configs/cameras.yaml의 카메라별 설정으로 되돌린다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchParams:
    top_ratio: float = 0.4
    head_width_ratio: tuple[float, float] = (0.15, 0.7)


@dataclass
class MatchResult:
    confirmed: np.ndarray  # 확정된 person 인덱스
    head_of: np.ndarray  # confirmed[i]에 배정된 head 인덱스
    unmatched_persons: int
    unmatched_heads: int


def confirm_people(
    persons: np.ndarray,
    person_scores: np.ndarray,
    heads: np.ndarray,
    params: MatchParams,
) -> MatchResult:
    raise NotImplementedError
