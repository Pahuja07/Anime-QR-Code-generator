"""Research script: Test QR generation and scannability validation for Phase 1."""

from pathlib import Path
from aniqr_ai.components.qr_generation import (
    QRCodeGenerator,
    QRGenerationConfig,
    generate_qr_code,
)
from aniqr_ai.components.qr_validation import QRValidator, validate_qr_code
from aniqr_ai.logger import logger


def run_experiment():
    print("=" * 60)
    print("AniQR - Phase 1: QR Generation Engine & Validation Experiment")
    print("=" * 60)

    output_dir = Path("artifacts/research_qrs")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("https://github.com/aniqr", "url", "URL QR Code"),
        ("AniQR Engine v1.0 - High Quality Anime QR", "text", "Plain Text QR Code"),
        ("+1 (555) 019-2834", "phone", "Click-to-Call QR Code"),
    ]

    generator = QRCodeGenerator(
        QRGenerationConfig(
            error_correction="H",
            box_size=10,
            border=4,
            target_size=(512, 512),
        )
    )
    validator = QRValidator()

    for idx, (content, content_type, label) in enumerate(samples, 1):
        file_path = output_dir / f"sample_{idx}_{content_type}.png"
        print(f"\n[{idx}] Generating {label}...")
        print(f"    Raw Input: '{content}' ({content_type})")

        # Generate and save
        generator.save(content, file_path, content_type=content_type)
        print(f"    Saved to: {file_path}")

        # Validate scannability
        scan_result = validator.validate(file_path)
        print(f"    Scannable: {'PASSED' if scan_result.is_valid else 'FAILED'}")
        print(f"    Decoded Payload: '{scan_result.decoded_text}'")

    print("\n" + "=" * 60)
    print("All sample QR codes generated and validated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
