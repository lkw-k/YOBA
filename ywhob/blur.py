"""사람 박스 블러와 검증 (spec 5.7).

- 사람 박스를 box_pad만큼 넓혀 긴 변 box_long_side로 축소 → 원래 크기로 확대
- 박스별 검증: 블러 후 선명도 <= sharpness_max 또는 <= 블러 전 × sharpness_ratio
- 프레임 긴 변이 output_long_side보다 크면 줄여서 JPEG. 실패하면 None (호출 측은 이미지를 보내지 않음)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import BlurConfig

log = logging.getLogger(__name__)


def sharpness(img: np.ndarray) -> float:
    """라플라시안 분산. 값이 클수록 선명하다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _expand_box(box, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    """박스를 상하좌우 pad 비율만큼 넓히고 프레임 안으로 자른다."""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (
        max(0, int(x0 - px)),
        max(0, int(y0 - py)),
        min(w, int(np.ceil(x1 + px))),
        min(h, int(np.ceil(y1 + py))),
    )


def _resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))  # cv2는 (너비, 높이) 순서
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def privacy_blur(frame: np.ndarray, boxes: np.ndarray, cfg: BlurConfig) -> bytes | None:
    """frame의 사람 박스를 블러한 JPEG. 검증 실패나 예외면 None."""
    try:
        img = frame.copy()  # 원본 배열은 건드리지 않는다
        h, w = img.shape[:2]

        for box in boxes:
            x0, y0, x1, y1 = _expand_box(box, cfg.box_pad, w, h)
            rw, rh = x1 - x0, y1 - y0
            if rw <= 0 or rh <= 0 or max(rw, rh) <= cfg.box_long_side:
                continue  # 이미 충분히 작은 박스는 얼굴을 알아볼 수 없다

            roi = img[y0:y1, x0:x1]
            before = sharpness(roi)
            small = _resize_long_side(roi, cfg.box_long_side)
            blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)

            after = sharpness(blurred)
            if after > cfg.sharpness_max and after > cfg.sharpness_ratio * before:
                log.warning("블러 검증 실패: 선명도 %.1f (블러 전 %.1f)", after, before)
                return None
            img[y0:y1, x0:x1] = blurred

        # 출력 크기보다 작은 프레임은 확대하지 않는다
        out = img if max(h, w) <= cfg.output_long_side else _resize_long_side(img, cfg.output_long_side)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if not ok:
            log.warning("블러 이미지 JPEG 인코딩 실패")
            return None
        return buf.tobytes()
    except Exception:
        log.exception("블러 처리 중 예외")  # 프레임 내용은 로그에 넣지 않는다
        return None
