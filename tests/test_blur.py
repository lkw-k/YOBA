import cv2
import numpy as np

from ywhob import blur
from ywhob.blur import privacy_blur, sharpness
from ywhob.config import BlurConfig


def _noise(h=1080, w=1920):
    # 무작위 노이즈 = 가장 선명한 이미지
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _upscaled(img: np.ndarray) -> np.ndarray:
    """출력(긴 변 960)을 원본 크기(1920×1080)로 되돌려 같은 좌표로 비교한다."""
    return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_NEAREST)


BOX = np.array([[800, 200, 1000, 800]], dtype=np.float32)  # 200×600 사람


def test_returns_jpeg_at_output_size():
    out = privacy_blur(_noise(), BOX, BlurConfig())
    assert out is not None and out[:2] == b"\xff\xd8"  # JPEG 시그니처
    assert max(_decode(out).shape[:2]) == 960


def test_only_box_is_blurred():
    img = _upscaled(_decode(privacy_blur(_noise(), BOX, BlurConfig())))
    inside = img[300:700, 850:950]
    outside = img[300:700, 100:500]
    assert sharpness(inside) < 0.1 * sharpness(outside)


def test_padding_covers_box_edges():
    # 박스 경계 바로 바깥(pad 10% 안쪽)도 블러돼야 한다
    img = _upscaled(_decode(privacy_blur(_noise(), BOX, BlurConfig())))
    edge = img[300:700, 790:800]  # 박스 왼쪽 바깥 10px
    far = img[300:700, 100:110]
    assert sharpness(edge) < 0.1 * sharpness(far)


def test_tiny_box_is_left_as_is():
    frame = _noise()
    small_box = np.array([[100, 100, 110, 112]], dtype=np.float32)  # 패딩 후에도 16px 이하
    assert privacy_blur(frame, small_box, BlurConfig()) == privacy_blur(frame, np.empty((0, 4)), BlurConfig())


def test_box_outside_frame_is_clipped():
    box = np.array([[-50, -50, 300, 400]], dtype=np.float32)
    assert privacy_blur(_noise(), box, BlurConfig()) is not None


def test_no_boxes_still_returns_image():
    assert privacy_blur(_noise(), np.empty((0, 4)), BlurConfig()) is not None


def test_unblurred_box_is_rejected(monkeypatch):
    # 축소가 적용되지 않으면(버그 등) 선명도 검증에서 걸려야 한다
    monkeypatch.setattr(blur, "_resize_long_side", lambda img, long_side: img)
    assert privacy_blur(_noise(), BOX, BlurConfig()) is None


def test_low_contrast_box_passes_by_absolute_threshold(monkeypatch):
    # 원래 밋밋한 영역(선명도 약 20)은 비율이 줄지 않아도 sharpness_max 이하라 통과해야 한다
    monkeypatch.setattr(blur, "_resize_long_side", lambda img, long_side: img)
    flat = (128 + np.random.default_rng(0).integers(-2, 3, (1080, 1920, 3))).astype(np.uint8)
    assert privacy_blur(flat, BOX, BlurConfig()) is not None


def test_small_frame_is_not_upscaled():
    out = privacy_blur(_noise(360, 640), np.empty((0, 4)), BlurConfig())
    assert _decode(out).shape[:2] == (360, 640)


def test_input_frame_is_not_modified():
    frame = _noise()
    before = frame.copy()
    privacy_blur(frame, BOX, BlurConfig())
    assert np.array_equal(frame, before)


def test_exception_returns_none():
    assert privacy_blur(np.zeros((0, 0, 3), np.uint8), BOX, BlurConfig()) is None
