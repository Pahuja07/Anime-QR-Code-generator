from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from aniqr_ai.components.qr_generation import QRCodeGenerator, QRGenerationConfig
from aniqr_ai.components.qr_validation import QRValidator, validate_qr_code
from aniqr_ai.pipeline.anime_qr_pipeline import QRGenerationPipeline


def test_qr_validator_decodes_and_validates():
    generator = QRCodeGenerator(QRGenerationConfig(error_correction="H", box_size=10))
    validator = QRValidator()

    # URL test
    url_img = generator.generate("https://github.com", "url")
    result = validator.validate(url_img, expected_content="https://github.com")
    assert result.is_valid
    assert result.decoded_text == "https://github.com"
    assert result.points is not None

    # Text test
    text_img = generator.generate("AniQR Studio 2026", "text")
    result = validator.validate(text_img, expected_content="AniQR Studio 2026")
    assert result.is_valid
    assert result.decoded_text == "AniQR Studio 2026"

    # Phone click-to-call test
    phone_img = generator.generate("+1 (800) 555-0199", "phone")
    result = validator.validate(phone_img, expected_content="tel:+18005550199")
    assert result.is_valid
    assert result.decoded_text == "tel:+18005550199"


def test_qr_validator_detects_content_mismatch():
    generator = QRCodeGenerator()
    validator = QRValidator()

    img = generator.generate("hello world", "text")
    result = validator.validate(img, expected_content="something else")
    assert not result.is_valid
    assert result.decoded_text == "hello world"
    assert "did not match expected" in result.error


def test_qr_validator_handles_blank_image():
    validator = QRValidator()
    blank_img = Image.new("RGB", (200, 200), color="white")
    result = validator.decode(blank_img)
    assert not result.is_valid
    assert result.decoded_text is None
    assert result.error is not None


def test_qr_generation_with_target_size(tmp_path: Path):
    target_size = (512, 512)
    config = QRGenerationConfig(error_correction="H", box_size=8, target_size=target_size)
    generator = QRCodeGenerator(config)

    img = generator.generate("https://example.com/target-size-test", "url")
    assert img.size == target_size

    # Ensure scannability is retained after nearest-neighbor resize
    result = validate_qr_code(img, expected_content="https://example.com/target-size-test")
    assert result.is_valid
    assert result.decoded_text == "https://example.com/target-size-test"

    # Test save with target_size
    save_path = tmp_path / "test_512.png"
    generator.save("https://example.com/target-size-test", save_path, "url")
    assert save_path.exists()
    saved_img = Image.open(save_path)
    assert saved_img.size == target_size


def test_qr_generation_pipeline(tmp_path: Path):
    output_file = tmp_path / "pipeline_test_qr.png"
    config = QRGenerationConfig(
        error_correction="H",
        box_size=10,
        output_image_path=output_file,
    )
    pipeline = QRGenerationPipeline(config=config)
    output = pipeline.run("https://aniqr.ai", content_type="url")

    assert output.is_valid
    assert output.normalized_content == "https://aniqr.ai"
    assert output.output_path == output_file
    assert output_file.exists()

    # Validate saved file
    result = validate_qr_code(output_file, expected_content="https://aniqr.ai")
    assert result.is_valid
    assert result.decoded_text == "https://aniqr.ai"
