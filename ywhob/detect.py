"""사람 검출 (spec 5.2).

- Ultralytics YOLO로 .pt/.onnx/.engine 로드, 5장을 한 배치로 추론
- 모델의 0번 클래스가 person인지 확인하고 0번만 남긴다 (COCO 가중치도 0번이 person이라 그대로 쓸 수 있음)
- person_conf 임계값, max_det 반영
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import DetectorConfig

log = logging.getLogger(__name__)

PERSON = 0


@dataclass
class FrameDetections:
    """한 프레임의 검출 결과. 박스는 xyxy 픽셀 좌표."""

    persons: np.ndarray  # (N, 4)
    person_scores: np.ndarray  # (N,)


class Detector(Protocol):
    def detect(self, frames: Sequence[np.ndarray]) -> list[FrameDetections]: ...


class YoloDetector:
    """YOLO26s 래퍼. model을 주면 그것을 쓴다 (테스트용)."""

    def __init__(self, cfg: DetectorConfig, model=None):
        self.cfg = cfg
        if model is None:
            from ultralytics import YOLO  # 서비스 실행에만 필요 (extra: model)

            model = YOLO(cfg.model, task="detect")
        names = model.names or {}
        if names.get(PERSON) != "person":
            raise ValueError(f"{cfg.model}: 0번 클래스가 person이 아님 ({names.get(PERSON)!r})")
        self.model = model
        log.info("검출 모델 로드: %s (클래스 %d개)", cfg.model, len(names))

    def detect(self, frames: Sequence[np.ndarray]) -> list[FrameDetections]:
        results = self.model.predict(
            list(frames),
            imgsz=self.cfg.imgsz,
            conf=self.cfg.person_conf,
            max_det=self.cfg.max_det,
            classes=[PERSON],
            device=self.cfg.device,
            verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            keep = b.cls.cpu().numpy().astype(int) == PERSON  # classes 인자를 무시하는 export 모델 대비
            out.append(
                FrameDetections(
                    persons=b.xyxy.cpu().numpy().reshape(-1, 4)[keep].astype(np.float32),
                    person_scores=b.conf.cpu().numpy().reshape(-1)[keep].astype(np.float32),
                )
            )
            if len(b) >= self.cfg.max_det:
                log.warning("검출 수가 max_det(%d)에 도달: 인원이 잘렸을 수 있음", self.cfg.max_det)
        return out
