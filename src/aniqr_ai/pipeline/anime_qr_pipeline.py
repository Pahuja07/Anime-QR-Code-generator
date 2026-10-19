"""Master pipeline orchestrating QR generation, anime stylization, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image

from aniqr_ai.config.configuration import ConfigurationManager
from aniqr_ai.components.qr_generation import (
    ContentType,
    QRCodeGenerator,
    QRGenerationConfig,
    normalize_content,
)
from aniqr_ai.components.qr_validation import QRValidator, QRValidationResult
from aniqr_ai.components.anime_image_generation import (
    AnimeGenerationParams,
    AnimeQRGenerationResult,
    AnimeQRGenerator,
    ANIME_STYLES,
)
from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger


@dataclass
class QRPipelineOutput:
    """Output results from the base QR generation pipeline."""

    image: Image.Image
    output_path: Path | None
    normalized_content: str
    validation_result: QRValidationResult

    @property
    def is_valid(self) -> bool:
        return self.validation_result.is_valid


@dataclass
class AnimeQRPipelineOutput:
    """Output results from the full Anime QR generation pipeline."""

    base_qr_image: Image.Image
    anime_qr_image: Image.Image
    output_path: Path | None
    normalized_content: str
    is_scannable: bool
    decoded_content: str | None
    attempts: int
    style: str
    effective_conditioning_scale: float
    validation_result: QRValidationResult


class QRGenerationPipeline:
    """End-to-end pipeline for Phase 1: generating and verifying scan-ready QR codes."""

    def __init__(
        self,
        config: QRGenerationConfig | None = None,
        config_path: str | Path = "config/config.yaml",
    ) -> None:
        if config is not None:
            self.config = config
        else:
            try:
                cm = ConfigurationManager(config_path)
                self.config = cm.get_qr_generation_config()
            except Exception as e:
                logger.warning(f"Could not load config from {config_path} ({e}), using default config.")
                self.config = QRGenerationConfig()

        self.generator = QRCodeGenerator(self.config)
        self.validator = QRValidator()

    def run(
        self,
        value: str,
        content_type: ContentType = "text",
        output_path: str | Path | None = None,
        target_size: tuple[int, int] | int | None = None,
    ) -> QRPipelineOutput:
        """Execute the base QR generation and validation pipeline."""
        try:
            logger.info(">>> Starting QR Generation Pipeline <<<")
            normalized_data = normalize_content(value, content_type)
            logger.info(f"Input: '{value}' ({content_type}) -> Normalized: '{normalized_data}'")

            # 1. Generate base QR Code image
            image = self.generator.generate(value, content_type, target_size=target_size)

            # 2. Validate QR Code scannability
            logger.info("Validating base QR code scannability...")
            validation_result = self.validator.validate(image, expected_content=normalized_data)
            if not validation_result.is_valid:
                logger.warning(f"Base QR code failed validation check: {validation_result.error}")
            else:
                logger.info("Base QR code passed validation successfully!")

            # 3. Save to disk if requested
            save_dest: Path | None = None
            dest = output_path or self.config.output_image_path
            if dest:
                save_dest = Path(dest)
                save_dest.parent.mkdir(parents=True, exist_ok=True)
                image.save(save_dest)
                logger.info(f"Saved base QR code image to: {save_dest.resolve()}")

            logger.info(">>> QR Generation Pipeline Completed Successfully <<<")
            return QRPipelineOutput(
                image=image,
                output_path=save_dest,
                normalized_content=normalized_data,
                validation_result=validation_result,
            )
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if isinstance(e, ValueError):
                raise
            raise AniQRException(f"Error in QRGenerationPipeline: {e}") from e


class AnimeQRPipeline:
    """Master pipeline orchestrating QR generation, anime stylization, and verification."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        self.base_pipeline = QRGenerationPipeline(config_path=config_path)
        self.anime_generator = AnimeQRGenerator()

    def generate(
        self,
        value: str,
        content_type: ContentType = "text",
        params: AnimeGenerationParams | None = None,
        output_path: str | Path | None = None,
        target_size: tuple[int, int] = (512, 512),
        reference_image: Image.Image | None = None,
    ) -> AnimeQRPipelineOutput:
        """Full Phase 1 + Phase 2 inference pipeline:
        1. Generate scannable base QR code (Error Correction H, target resolution).
        2. Condition ControlNet with the base QR code and prompt.
        3. Validate scannability and retry adaptively if unscannable.
        4. Save final output.
        """
        try:
            params = params or AnimeGenerationParams()
            logger.info(">>> Starting Full Anime QR Pipeline <<<")

            # Step 1: Generate high error-correction base QR image
            base_output = self.base_pipeline.run(
                value=value,
                content_type=content_type,
                target_size=target_size,
            )

            # Step 2: Generate stylized anime QR image with retry loop
            anime_result: AnimeQRGenerationResult = self.anime_generator.generate(
                qr_image=base_output.image,
                expected_content=base_output.normalized_content,
                params=params,
                reference_image=reference_image,
            )

            # Step 3: Save final output image if path specified
            save_dest: Path | None = None
            if output_path:
                save_dest = Path(output_path)
                save_dest.parent.mkdir(parents=True, exist_ok=True)
                anime_result.image.save(save_dest)
                logger.info(f"Saved anime QR image to: {save_dest.resolve()}")

            logger.info(">>> Full Anime QR Pipeline Completed <<<")
            return AnimeQRPipelineOutput(
                base_qr_image=base_output.image,
                anime_qr_image=anime_result.image,
                output_path=save_dest,
                normalized_content=base_output.normalized_content,
                is_scannable=anime_result.is_scannable,
                decoded_content=anime_result.decoded_content,
                attempts=anime_result.attempts,
                style=anime_result.style,
                effective_conditioning_scale=anime_result.effective_conditioning_scale,
                validation_result=anime_result.validation_result,
            )
        except Exception as e:
            logger.error(f"AnimeQRPipeline execution error: {e}")
            if isinstance(e, ValueError):
                raise
            raise AniQRException(f"Error in AnimeQRPipeline: {e}") from e
