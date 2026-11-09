"""실행 진입점: `uv run ywhob --cameras configs/cameras.yaml [--model s|n]`

웹캠 테스트: `uv run ywhob --webcam [번호]` (configs/cameras.webcam.yaml, 화면 전체를 한 구역으로)
실시간 미리보기: `uv run ywhob --webcam --live` (매 프레임 처리해 PC 창에 띄움, Redis 전송 없음)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from .capture import WEBCAM_SCHEME
from .config import load_cameras, load_service
from .live import run_live
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ywhob", description="CCTV 혼잡도 측정 파이프라인")
    ap.add_argument("--service", default="configs/service.yaml")
    ap.add_argument("--cameras", help="카메라 설정. 기본 configs/cameras.yaml, --webcam이면 configs/cameras.webcam.yaml")
    ap.add_argument("--webcam", type=int, nargs="?", const=0, metavar="번호",
                    help="CCTV 대신 PC에 연결된 웹캠으로 실행 (기본 0번: 노트북 내장, USB 웹캠은 보통 1)")
    ap.add_argument("--live", action="store_true",
                    help="10초 주기 대신 매 프레임 처리해 블러 화면을 PC 창에 띄움 (카메라 1대, Redis 전송 없음)")
    ap.add_argument("--model", help="검출 모델. service.yaml detector.models의 별칭(s: 스몰, n: 나노) 또는 가중치 경로")
    ap.add_argument("--check", action="store_true", help="설정만 검증하고 종료")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = load_service(args.service)
    if args.model:
        service = dataclasses.replace(service, detector=service.detector.with_model(args.model))
    if args.cameras is None:
        args.cameras = "configs/cameras.webcam.yaml" if args.webcam is not None else "configs/cameras.yaml"
    if not Path(args.cameras).is_file():
        logging.error("카메라 설정 없음: %s (configs/cameras.example.yaml을 복사해 RTSP 주소와 ROI를 채운다)", args.cameras)
        sys.exit(1)
    cameras = load_cameras(args.cameras)
    if args.webcam is not None:
        cameras = [dataclasses.replace(c, url=f"{WEBCAM_SCHEME}{args.webcam}") for c in cameras]
        logging.info("웹캠 %d번으로 실행", args.webcam)
    if (args.live or args.webcam is not None) and len(cameras) != 1:
        opt = "--live" if args.live else "--webcam"  # 웹캠 한 대를 여러 카메라가 동시에 열 수 없다
        logging.error("%s는 카메라 1대만 지원: %s에 %d대 (1대짜리 설정을 준다)", opt, args.cameras, len(cameras))
        sys.exit(1)
    zones = sum(len(c.zones) for c in cameras)
    logging.info("설정 로드: 카메라 %d대, 구역 %d개, 모델 %s", len(cameras), zones, service.detector.model)

    if not Path(service.detector.model).is_file():
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
        run_live(service, cameras[0])
        return
    Pipeline(service, cameras).run()


if __name__ == "__main__":
    main()
