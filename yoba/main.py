"""실행 진입점: `uv run yoba --cameras configs/cameras.yaml [--model s|n]`

웹캠 테스트: `uv run yoba --webcam [번호]` (configs/cameras.webcam.yaml, 화면 전체를 한 구역으로)
RTSP 테스트: `uv run yoba --rtsp "rtsp://..."` (같은 설정, 화면 전체 한 구역)
실시간 모드: `--live` (매 프레임 처리해 PC 창에 띄움). `--publish`를 더하면 프레임마다 Redis에도 씀 (spec 9.4)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import urllib.request

import redis
from pathlib import Path

from .capture import WEBCAM_SCHEME
from .config import load_cameras, load_service
from .live import run_live
from .pipeline import Pipeline
from .publish import LatestSender, LivePublisher


def download_model(url: str, dest: Path) -> bool:
    """가중치를 받아 dest에 둔다. 받다 끊겨도 깨진 파일이 남지 않게 .part에 받은 뒤 옮긴다."""
    part = dest.with_name(dest.name + ".part")
    logging.info("모델 다운로드: %s → %s", url, dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        part.replace(dest)
        return True
    except OSError as e:  # URLError, HTTPError(404 포함) 모두 OSError
        part.unlink(missing_ok=True)
        logging.error("모델 다운로드 실패: %s (%s)", url, e)
        return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yoba", description="CCTV 혼잡도 측정 파이프라인")
    ap.add_argument("--service", default="configs/service.yaml")
    ap.add_argument("--cameras", help="카메라 설정. 기본 configs/cameras.yaml, --webcam/--rtsp면 configs/cameras.webcam.yaml")
    test_src = ap.add_mutually_exclusive_group()
    test_src.add_argument("--webcam", type=int, nargs="?", const=0, metavar="번호",
                          help="CCTV 대신 PC에 연결된 웹캠으로 실행 (기본 0번: 노트북 내장, USB 웹캠은 보통 1)")
    test_src.add_argument("--rtsp", metavar="URL",
                          help="CCTV 한 대를 RTSP 주소로 바로 열어 테스트 (화면 전체 한 구역, ROI 설정 불필요)")
    ap.add_argument("--live", action="store_true",
                    help="10초 주기 대신 매 프레임 처리해 블러 화면을 PC 창에 띄움 (카메라 1대)")
    ap.add_argument("--publish", action="store_true",
                    help="--live와 함께: 프레임마다 인원과 블러 이미지를 Redis(yoba:live, yoba:frame)에 씀")
    ap.add_argument("--model", help="검출 모델. service.yaml detector.models의 별칭(s: 스몰, n: 나노) 또는 가중치 경로")
    ap.add_argument("--check", action="store_true", help="설정만 검증하고 종료")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = load_service(args.service)
    if args.model:
        service = dataclasses.replace(service, detector=service.detector.with_model(args.model))
    if args.publish and not args.live:
        logging.error("--publish는 --live와 함께 쓴다 (10초 주기 모드는 원래 Redis로 보냄)")
        sys.exit(1)
    test_mode = args.webcam is not None or args.rtsp is not None
    if args.rtsp is not None and not args.rtsp.lower().startswith(("rtsp://", "rtsps://")):
        logging.error("--rtsp 주소는 rtsp:// 또는 rtsps://로 시작해야 함")  # 주소에 계정이 있을 수 있어 출력하지 않는다
        sys.exit(1)
    if args.cameras is None:
        args.cameras = "configs/cameras.webcam.yaml" if test_mode else "configs/cameras.yaml"
    if not Path(args.cameras).is_file():
        logging.error("카메라 설정 없음: %s (configs/cameras.example.yaml을 복사해 RTSP 주소와 ROI를 채운다)", args.cameras)
        sys.exit(1)
    cameras = load_cameras(args.cameras)
    if args.webcam is not None:
        cameras = [dataclasses.replace(c, url=f"{WEBCAM_SCHEME}{args.webcam}") for c in cameras]
        logging.info("웹캠 %d번으로 실행", args.webcam)
    if args.rtsp is not None:
        cameras = [dataclasses.replace(c, url=args.rtsp) for c in cameras]
        logging.info("RTSP 주소로 실행 (카메라 id: %s)", ", ".join(c.id for c in cameras))  # 주소(계정 포함)는 남기지 않는다
    if (args.live or test_mode) and len(cameras) != 1:
        # 웹캠/RTSP 한 대를 여러 카메라가 동시에 열 수 없다
        opt = "--live" if args.live else ("--webcam" if args.webcam is not None else "--rtsp")
        logging.error("%s는 카메라 1대만 지원: %s에 %d대 (1대짜리 설정을 준다)", opt, args.cameras, len(cameras))
        sys.exit(1)
    zones = sum(len(c.zones) for c in cameras)
    logging.info("설정 로드: 카메라 %d대, 구역 %d개, 모델 %s", len(cameras), zones, service.detector.model)

    det = service.detector
    model = Path(det.model)
    # 별칭 파일만 받는다. 임의 경로는 사용자가 둔 파일이라 받지 않는다
    if not model.is_file() and not args.check and det.download_url and det.model in det.models.values():
        download_model(det.download_url + model.name, model)
    if not model.is_file():
        aliases = ", ".join(f"{k}={v}" for k, v in service.detector.models.items())
        msg = "모델 파일 없음: %s (별칭: %s). models/README.md 참고"
        if args.check:
            logging.warning(msg, service.detector.model, aliases or "없음")
        else:
            logging.error(msg, service.detector.model, aliases or "없음")
            sys.exit(1)
    if args.check:
        return
    if args.live:
        publisher = None
        if args.publish:
            rc = service.redis
            client = redis.Redis.from_url(rc.url, socket_timeout=rc.socket_timeout, socket_connect_timeout=rc.socket_timeout)
            publisher = LatestSender(LivePublisher(client, rc, cameras[0]), rc.live_senders)
        try:
            run_live(service, cameras[0], publisher=publisher)
        finally:
            if publisher is not None:
                publisher.close()
        return
    Pipeline(service, cameras).run()


if __name__ == "__main__":
    main()
