import numpy as np
import pytest

from aniqr_ai.components.qr_generation import (
    QRCodeGenerator,
    QRGenerationConfig,
    normalize_content,
)


def test_phone_number_uses_click_to_call_format():
    assert normalize_content("+1 (555) 123-4567", "phone") == "tel:+15551234567"


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        ("example.com/path", "url", "https://example.com/path"),
        ("hello AniQR", "text", "hello AniQR"),
        ("555-1234", "phone", "tel:5551234"),
    ],
)
def test_normalize_content(value, kind, expected):
    assert normalize_content(value, kind) == expected


def test_invalid_content_and_config_are_rejected():
    with pytest.raises(ValueError):
        normalize_content("", "text")
    with pytest.raises(ValueError):
        normalize_content("not a phone", "phone")
    with pytest.raises(ValueError):
        QRGenerationConfig(error_correction="X")


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        ("https://example.com/aniqr", "url", "https://example.com/aniqr"),
        ("AniQR scan check", "text", "AniQR scan check"),
        ("+15551234567", "phone", "tel:+15551234567"),
    ],
)
def test_generated_qr_scans_back_to_encoded_value(value, kind, expected):
    cv2 = pytest.importorskip("cv2", reason="OpenCV is needed for QR scan verification")
    image = QRCodeGenerator(QRGenerationConfig(error_correction="H", box_size=10)).generate(value, kind)
    pixels = np.asarray(image)
    decoded, points, _ = cv2.QRCodeDetector().detectAndDecode(pixels)
    assert points is not None
    assert decoded == expected
