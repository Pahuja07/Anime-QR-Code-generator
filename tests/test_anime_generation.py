from PIL import Image
import pytest

from aniqr_ai.components.anime_image_generation import (
    AnimeGenerationParams,
    AnimeQRGenerator,
    ANIME_STYLES,
)
from aniqr_ai.components.qr_validation import QRValidationResult
from aniqr_ai.components.qr_generation import QRCodeGenerator, QRGenerationConfig
from aniqr_ai.components.qr_validation import QRValidator


@pytest.mark.parametrize(
    "kwargs",
    [
        {"style": "unknown"},
        {"max_retries": 0},
        {"num_inference_steps": 100},
        {"control_guidance_start": 0.9, "control_guidance_end": 0.5},
    ],
)
def test_generation_parameters_reject_out_of_range_values(kwargs):
    with pytest.raises(ValueError):
        AnimeGenerationParams(**kwargs)


def test_anime_generator_retries_and_reports_scale_used():
    class Loader:
        def load_pipeline(self):
            return "MOCK_DIFFUSERS_PIPELINE"

    class Validator:
        calls = 0

        def validate(self, image, expected_content):
            self.calls += 1
            if self.calls == 1:
                return QRValidationResult(False, None, "first attempt unreadable")
            return QRValidationResult(True, expected_content)

    assert len(ANIME_STYLES) == 7
    result = AnimeQRGenerator(Loader(), Validator()).generate(
        Image.new("RGB", (128, 128), "white"),
        "https://example.com",
        AnimeGenerationParams(max_retries=2),
    )

    assert result.is_scannable
    assert result.attempts == 2
    assert result.effective_conditioning_scale == pytest.approx(1.35)


def test_artwork_blend_preserves_quiet_zone_and_qr_modules():
    qr = Image.new("RGB", (100, 100), "white")
    # Simple QR-like black modules inset from the white quiet zone.
    for y in range(30, 70):
        for x in range(30, 70):
            if (x // 10 + y // 10) % 2 == 0:
                qr.putpixel((x, y), (0, 0, 0))
    artwork = Image.new("RGB", qr.size, (200, 30, 20))

    result = AnimeQRGenerator._show_art_through_qr(artwork, qr, attempt=1)

    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((35, 35))[0] < 100
    assert result.getpixel((45, 35))[0] > 150


def test_uploaded_reference_image_is_blended_into_qr_art():
    qr = Image.new("RGB", (100, 100), "white")
    for y in range(30, 70):
        for x in range(30, 70):
            if (x // 10 + y // 10) % 2 == 0:
                qr.putpixel((x, y), (0, 0, 0))
    generated_art = Image.new("RGB", qr.size, (245, 245, 245))
    uploaded_art = Image.new("RGB", qr.size, (20, 20, 220))

    result = AnimeQRGenerator._show_art_through_qr(
        generated_art, qr, attempt=1, reference_image=uploaded_art
    )

    # A light module contains the uploaded image's blue contribution, while quiet zone stays white.
    red, green, blue = result.getpixel((45, 35))
    assert blue > red
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_iron_man_artwork_remains_scannable_in_local_pipeline():
    class DevModel:
        def load_pipeline(self):
            return "MOCK_DIFFUSERS_PIPELINE"

    payload = "https://example.com/iron-man-qr"
    qr = QRCodeGenerator(QRGenerationConfig(target_size=512)).generate(payload, "url")
    result = AnimeQRGenerator(model_loader=DevModel()).generate(
        qr,
        payload,
        AnimeGenerationParams(style="iron_man", max_retries=3, seed=42),
    )

    assert result.is_scannable
    assert QRValidator().validate(result.image, expected_content=payload).is_valid

