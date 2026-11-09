"""단계 연결 (spec 2, 6).

    [capture ×N] ─frame queue─▶ [inference] ─result queue─▶ [aggregate → blur → publish]

- 카메라별 캡처 스레드: 카메라 i는 i × (cycle / N) 오프셋으로 시작해 추론 부하를 주기 안에 나눈다
- 큐 크기 제한, 가득 차면 가장 오래된 주기를 버린다 (지연보다 최신성)
- 주기 처리 시간(캡처 시작 → 전송 끝)을 로깅하고, cycle_seconds를 넘으면 경고
- 원본 프레임은 후처리가 끝나면 즉시 버린다
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import redis

from .aggregate import ZoneAssigner, ZoneTracker, median_frame_index
from .blur import privacy_blur
from .capture import CameraStream, capture_burst
from .config import CameraConfig, ServiceConfig
from .detect import Detector, FrameDetections, YoloDetector
from .publish import KST, Publisher

log = logging.getLogger(__name__)


@dataclass
class CaptureItem:
    """캡처 → 추론. frames가 min_frames보다 적으면 error 주기."""

    camera_id: str
    captured_at: datetime
    frames: list[np.ndarray]
    started: float = 0.0  # time.monotonic(), 처리 시간 측정용


@dataclass
class DetectItem:
    """추론 → 후처리. 블러를 위해 원본 frames를 함께 넘기고, 후처리 후 폐기한다."""

    camera_id: str
    captured_at: datetime
    frames: list[np.ndarray]
    detections: list[FrameDetections]  # 캡처 부족이나 추론 실패면 빈 리스트
    started: float = 0.0


class Pipeline:
    def __init__(
        self,
        service: ServiceConfig,
        cameras: list[CameraConfig],
        detector: Detector | None = None,
        redis_client=None,
        stream_factory=CameraStream,
    ):
        self.service = service
        self.cameras = cameras
        self.detector = detector  # None이면 run()에서 모델을 불러온다
        if redis_client is None:
            rc = service.redis
            redis_client = redis.Redis.from_url(
                rc.url, socket_timeout=rc.socket_timeout, socket_connect_timeout=rc.socket_timeout
            )
        self.publisher = Publisher(redis_client, service.redis)
        self._stream_factory = stream_factory
        self._by_id = {c.id: c for c in cameras}
        self._assigners = {c.id: ZoneAssigner(c.zones, c.mask) for c in cameras}
        self._trackers = {c.id: [ZoneTracker(z, service.aggregate) for z in c.zones] for c in cameras}
        self._frame_q: queue.Queue[CaptureItem] = queue.Queue(service.queue_size)
        self._result_q: queue.Queue[DetectItem] = queue.Queue(service.queue_size)
        self._put_lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """캡처, 추론, 후처리 워커를 띄우고 종료 신호까지 대기."""
        if self.detector is None:
            self.detector = YoloDetector(self.service.detector)  # 모델 오류는 스트림 연결 전에 드러나게

        streams = [self._stream_factory(c, self.service.capture) for c in self.cameras]
        for s in streams:
            s.start()
        threads = [
            threading.Thread(target=self._capture_worker, args=(i, s), name=f"cycle-{s.camera.id}", daemon=True)
            for i, s in enumerate(streams)
        ]
        threads.append(threading.Thread(target=self._inference_worker, name="inference", daemon=True))
        threads.append(threading.Thread(target=self._postprocess_worker, name="postprocess", daemon=True))
        for t in threads:
            t.start()
        log.info("파이프라인 시작: 카메라 %d대, 주기 %.0f초", len(self.cameras), self.service.cycle_seconds)

        try:
            while not self._stop.wait(1.0):  # Windows에서도 Ctrl+C가 먹도록 짧게 깨어난다
                pass
        except KeyboardInterrupt:
            log.info("종료 신호 받음")
        finally:
            self._stop.set()
            for t in threads:
                t.join(timeout=self.service.capture.window_seconds + 5)
            for s in streams:
                s.stop()
            log.info("파이프라인 종료")

    def _put_latest(self, q: queue.Queue, item) -> None:
        """큐가 가득 차면 가장 오래된 주기를 버리고 넣는다."""
        with self._put_lock:
            while True:
                try:
                    q.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        old = q.get_nowait()
                        log.warning("%s: 처리가 밀려 오래된 주기를 버림", old.camera_id)
                    except queue.Empty:
                        pass

    def _capture_worker(self, index: int, stream: CameraStream) -> None:
        cycle = self.service.cycle_seconds
        offset = index * cycle / len(self.cameras)
        base = time.monotonic()
        k = 0
        while True:
            due = base + offset + k * cycle
            if self._stop.wait(max(0.0, due - time.monotonic())):
                return
            started = time.monotonic()
            captured_at = datetime.now(KST)
            frames = capture_burst(stream, self.service.capture)
            self._put_latest(self._frame_q, CaptureItem(stream.camera.id, captured_at, frames, started))
            # 늦어져 이미 지난 시점은 건너뛰고 다음 시점에 맞춘다 (시점 분산 유지)
            k = max(k + 1, int((time.monotonic() - base - offset) // cycle) + 1)

    def _inference_worker(self) -> None:
        min_frames = self.service.capture.min_frames
        while not self._stop.is_set():
            try:
                item = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            detections: list[FrameDetections] = []
            if len(item.frames) >= min_frames:
                try:
                    detections = self.detector.detect(item.frames)
                except Exception:
                    log.exception("%s: 추론 실패, error 처리", item.camera_id)
            self._put_latest(
                self._result_q,
                DetectItem(item.camera_id, item.captured_at, item.frames, detections, item.started),
            )

    def _postprocess_worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._result_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.postprocess(item)
            except Exception:
                log.exception("%s: 후처리 실패", item.camera_id)

    def postprocess(self, item: DetectItem) -> None:
        """구역 집계 → 판정 → 블러 → 전송."""
        cam = self._by_id[item.camera_id]
        frames, dets = item.frames, item.detections
        try:
            if len(frames) < self.service.capture.min_frames or len(dets) != len(frames):
                log.warning("%s: 캡처 %d장, 검출 %d장 → error", cam.id, len(frames), len(dets))
                for z in cam.zones:
                    self.publisher.publish_error(z.zone_id, item.captured_at)
                return

            per_frame = [self._assigners[cam.id].count(d.persons) for d in dets]
            results = [t.update([c[t.zone.zone_id] for c in per_frame]) for t in self._trackers[cam.id]]

            image = None
            if any(r.accepted for r in results):
                totals = [sum(c.values()) for c in per_frame]
                i = median_frame_index(totals)
                image = privacy_blur(frames[i], dets[i].persons, self.service.blur)
                if image is None:
                    log.warning("%s: 블러 실패, 이미지 없이 수치만 전송", cam.id)

            for r in results:
                if r.accepted:
                    self.publisher.publish(r.zone_id, r.level, r.score, r.confidence, image, item.captured_at)
                else:
                    self.publisher.publish_low(r.zone_id)

            log.info(
                "%s: %.1f초, %s",
                cam.id,
                time.monotonic() - item.started if item.started else 0.0,
                ", ".join(f"{r.zone_id}={r.median:g}명({r.confidence}, 단계 {r.level})" for r in results),
            )
        finally:
            frames.clear()  # 원본 프레임 폐기
            self.publisher.heartbeat(datetime.now(KST))
            if item.started and time.monotonic() - item.started > self.service.cycle_seconds:
                log.warning(
                    "%s: 주기 처리 %.1f초 (주기 %.0f초 초과)",
                    cam.id, time.monotonic() - item.started, self.service.cycle_seconds,
                )
