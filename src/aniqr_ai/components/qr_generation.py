"""Reusable QR code generation utilities for AniQR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import qrcode
from PIL import Image
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger

ContentType = Literal["url", "text", "phone"]

_ERROR_CORRECTION = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}


@dataclass(frozen=True)
class QRGenerationConfig:
    """Rendering options. ``box_size`` controls pixel resolution per module."""

    error_correction: Literal["L", "M", "Q", "H"] = "H"
    box_size: int = 10
    border: int = 4
    fill_color: str = "black"
    back_color: str = "white"
    target_size: tuple[int, int] | int | None = None
    root_dir: Path | None = None
    output_image_path: Path | None = None

    def __post_init__(self) -> None:
        if self.error_correction not in _ERROR_CORRECTION:
            raise ValueError("error_correction must be one of: L, M, Q, H")
        if self.box_size < 1:
            raise ValueError("box_size must be at least 1")
        if self.border < 4:
            raise ValueError("border must be at least 4 modules for reliable scanning")
        if self.target_size is not None:
            width, height = (self.target_size, self.target_size) if isinstance(self.target_size, int) else self.target_size
            if width < 1 or height < 1:
                raise ValueError("target_size dimensions must be positive")
            if width != height:
                raise ValueError("target_size must be square to preserve QR module geometry")


def normalize_content(value: str, content_type: ContentType) -> str:
    """Validate and normalize a URL, text value, or phone number for encoding."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("QR content must be a non-empty string")
    value = value.strip()

    if content_type == "text":
        return value
    if content_type == "url":
        candidate = value if "://" in value else f"https://{value}"
        parsed = urlsplit(candidate)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not parsed.hostname
            or "." not in parsed.hostname
        ):
            raise ValueError("Provide a valid HTTP(S) URL or domain name")
        return candidate
    if content_type == "phone":
        # Keep a leading international +; remove common visual separators.
        number = re.sub(r"[\s().-]", "", value)
        if number.lower().startswith("tel:"):
            number = number[4:]
        if not re.fullmatch(r"\+?\d{3,15}", number):
            raise ValueError("Phone number must contain 3-15 digits, optionally starting with +")
        return f"tel:{number}"
    raise ValueError("content_type must be 'url', 'text', or 'phone'")


class QRCodeGenerator:
    """Create scan-ready QR images and optionally save them to disk."""

    def __init__(self, config: QRGenerationConfig | None = None) -> None:
        self.config = config or QRGenerationConfig()

    def generate(
        self,
        value: str,
        content_type: ContentType = "text",
        target_size: tuple[int, int] | int | None = None,
    ) -> Image.Image:
        """Return a Pillow RGB image encoding the supplied value.
        
        Optionally resizes to target_size (e.g., 512 or (512, 512)) using
        nearest-neighbor resampling to maintain crisp module edges for AI models.
        """
        try:
            data = normalize_content(value, content_type)
            logger.info(f"Generating QR code for content_type='{content_type}', normalized_data='{data}'")

            qr = qrcode.QRCode(
                version=None,
                error_correction=_ERROR_CORRECTION[self.config.error_correction],
                box_size=self.config.box_size,
                border=self.config.border,
            )
            qr.add_data(data)
            qr.make(fit=True)
            image = qr.make_image(
                fill_color=self.config.fill_color,
                back_color=self.config.back_color,
            ).convert("RGB")

            # Determine desired output dimensions
            desired_size = target_size if target_size is not None else self.config.target_size
            if desired_size is not None:
                if isinstance(desired_size, int):
                    desired_size = (desired_size, desired_size)
                if len(desired_size) != 2 or min(desired_size) < 1:
                    raise ValueError("target_size must contain two positive dimensions")
                if desired_size[0] != desired_size[1]:
                    raise ValueError("target_size must be square to preserve QR module geometry")
                image = image.resize(desired_size, resample=Image.Resampling.NEAREST)
                logger.info(f"Resized QR code image to target size: {desired_size}")

            return image
        except Exception as e:
            logger.error(f"Failed to generate QR code: {e}")
            if isinstance(e, ValueError):
                raise
            raise AniQRException(f"Error generating QR code: {e}") from e

    def save(
        self,
        value: str,
        path: str | Path,
        content_type: ContentType = "text",
        format: str | None = None,
        target_size: tuple[int, int] | int | None = None,
    ) -> Path:
        """Generate and save a QR image, creating parent directories if needed."""
        try:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image = self.generate(value, content_type, target_size=target_size)
            image.save(destination, format=format)
            logger.info(f"QR code successfully saved to: {destination.resolve()}")
            return destination
        except Exception as e:
            logger.error(f"Failed to save QR code to {path}: {e}")
            if isinstance(e, ValueError):
                raise
            raise AniQRException(f"Error saving QR code to {path}: {e}") from e


def generate_qr_code(
    value: str,
    content_type: ContentType = "text",
    config: QRGenerationConfig | None = None,
    target_size: tuple[int, int] | int | None = None,
) -> Image.Image:
    """Convenience function for callers that do not need a generator instance."""
    return QRCodeGenerator(config).generate(value, content_type, target_size=target_size)
