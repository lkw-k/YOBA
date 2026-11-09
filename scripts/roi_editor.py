"""프레임 위에 ROI/mask 다각형을 그려 cameras.yaml 형식으로 출력 (spec 7.2)

- CCTV(RTSP)나 동영상 파일에서 프레임 1장을 받아 창에 띄운다
- 원본은 저장하지 않고, 화면에도 프레임 전체를 픽셀화해서 띄운다 (바닥·부스 배치는 보이고 얼굴은 알아볼 수 없는 정도)
- 좌표는 원본 해상도 기준으로 출력한다. 서브스트림을 쓰면 서비스에서 받는 것과 같은 스트림 주소로 그린다

사용:
    uv run python scripts/roi_editor.py --id cam01 --url rtsp://user:pass@192.168.0.10:554/stream2
    창에서
      왼쪽 클릭   점 추가
      Enter       다각형 닫기 → 콘솔에서 zone_id, capacity 입력 (mask 모드면 바로 저장)
      m           구역/mask 모드 전환
      u           마지막 점 취소 (점이 없으면 마지막 다각형 취소)
      q, Esc      종료 → 콘솔에 cameras.yaml용 YAML 출력
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ywhob.capture import open_video  # noqa: E402
from ywhob.config import load_service  # noqa: E402

ZONE_COLOR, MASK_COLOR, DRAFT_COLOR = (0, 200, 0), (0, 0, 220), (0, 220, 220)


def pixelate(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return frame.copy()
    small = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def camera_yaml(cam_id: str, url: str, zones: list[dict], masks: list[list]) -> str:
    """cameras.yaml의 cameras 항목 하나. URL에는 계정이 있을 수 있어 자리표시자로 둔다."""
    cam = {"id": cam_id, "url": url, "zones": zones}
    if masks:
        cam["mask"] = masks
    return yaml.safe_dump({"cameras": [cam]}, allow_unicode=True, sort_keys=False, default_flow_style=None)


def grab_frame(url: str, service: str) -> np.ndarray:
    cap = open_video(url, load_service(service).capture)
    last = None
    try:
        for _ in range(30):  # CCTV는 첫 몇 프레임이 깨져 들어오기도 해서 몇 장 넘긴다
            ok, frame = cap.read()
            if ok and frame is not None:
                last = frame
    finally:
        cap.release()
    if last is None:
        raise SystemExit("프레임을 받지 못함: 주소, 계정, 네트워크를 확인")
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="카메라 id (cameras.yaml의 id)")
    ap.add_argument("--url", required=True, help="RTSP 주소 또는 동영상 파일")
    ap.add_argument("--service", default="configs/service.yaml", help="RTSP 전송 방식·타임아웃을 읽을 설정")
    ap.add_argument("--pixelate", type=int, default=240, help="화면 표시용 픽셀화 긴 변(px). 작을수록 흐림")
    ap.add_argument("--max-width", type=int, default=1600, help="창 최대 너비(px). 좌표는 원본 기준으로 환산")
    args = ap.parse_args()

    frame = grab_frame(args.url, args.service)
    h, w = frame.shape[:2]
    base = pixelate(frame, args.pixelate)
    del frame  # 원본은 더 쓰지 않는다
    view_scale = min(1.0, args.max_width / w)

    zones: list[dict] = []
    masks: list[list] = []
    draft: list[list[int]] = []
    mode = ["zone"]

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            draft.append([round(x / view_scale), round(y / view_scale)])

    win = f"roi_editor - {args.id}"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(f"원본 해상도 {w}x{h}. 클릭으로 점 추가, Enter로 닫기, m 모드 전환, u 취소, q 종료")

    while True:
        img = base.copy()
        for z in zones:
            cv2.polylines(img, [np.int32(z["roi"])], True, ZONE_COLOR, 2)
            cv2.putText(img, z["zone_id"], tuple(z["roi"][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, ZONE_COLOR, 2)
        for m in masks:
            cv2.polylines(img, [np.int32(m)], True, MASK_COLOR, 2)
        if draft:
            cv2.polylines(img, [np.int32(draft)], False, DRAFT_COLOR, 2)
            for p in draft:
                cv2.circle(img, tuple(p), 4, DRAFT_COLOR, -1)
        cv2.putText(img, f"mode: {mode[0]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, DRAFT_COLOR, 2)
        if view_scale < 1:
            img = cv2.resize(img, (round(w * view_scale), round(h * view_scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow(win, img)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("m"):
            mode[0] = "mask" if mode[0] == "zone" else "zone"
        elif key == ord("u"):
            if draft:
                draft.pop()
            elif mode[0] == "zone" and zones:
                zones.pop()
            elif mode[0] == "mask" and masks:
                masks.pop()
        elif key == 13:  # Enter
            if len(draft) < 3:
                print("점이 3개 이상이어야 함")
                continue
            if mode[0] == "mask":
                masks.append(list(draft))
            else:
                zone_id = input("zone_id: ").strip()
                capacity = float(input("capacity (구역 면적 m² × 1.5): ").strip())
                zones.append({"zone_id": zone_id, "roi": list(draft), "capacity": capacity})
            draft.clear()

    cv2.destroyAllWindows()
    print("\n# configs/cameras.yaml에 붙여 넣는다 (url은 실제 주소로)")
    print(camera_yaml(args.id, "rtsp://<user>:<password>@<host>/<stream>", zones, masks))


if __name__ == "__main__":
    main()
