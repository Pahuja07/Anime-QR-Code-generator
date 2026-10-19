"""QR code validation and scanning module for AniQR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger


@dataclass
class QRValidationResult:
    """Holds the result of a QR code validation/scan check."""

    is_valid: bool
    decoded_text: str | None
    error: str | None = None
    points: Any = None

    def __bool__(self) -> bool:
        return self.is_valid


class QRValidator:
    """Scans and validates scannability of QR codes using computer vision."""

    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def _to_cv2_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        """Convert PIL Image, numpy array, or file path to OpenCV-compatible numpy image."""
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                raise ValueError(f"Failed to read image from path: {img_path}")
            return cv_img

        if isinstance(image, Image.Image):
            rgb_arr = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            return image

        raise TypeError(f"Unsupported image type: {type(image)}")

    def decode(self, image: Image.Image | np.ndarray | str | Path) -> QRValidationResult:
        """Decode a QR code from an image and return the decoded data."""
        try:
            cv_img = self._to_cv2_image(image)
            decoded_text, points, _ = self.detector.detectAndDecode(cv_img)

            if points is not None and decoded_text:
                logger.info(f"QR code decoded successfully: '{decoded_text}'")
                return QRValidationResult(
                    is_valid=True,
                    decoded_text=decoded_text,
                    points=points.tolist() if hasattr(points, "tolist") else points,
                )

            # Try grayscale thresholding if direct scan fails
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
            decoded_text, points, _ = self.detector.detectAndDecode(gray)
            if points is not None and decoded_text:
                logger.info(f"QR code decoded successfully via grayscale: '{decoded_text}'")
                return QRValidationResult(
                    is_valid=True,
                    decoded_text=decoded_text,
                    points=points.tolist() if hasattr(points, "tolist") else points,
                )

            logger.warning("QR code detection failed: No scannable QR code found.")
            return QRValidationResult(
                is_valid=False,
                decoded_text=None,
                error="No scannable QR code detected in the provided image.",
            )
        except Exception as e:
            logger.error(f"Error during QR code decoding: {e}")
            return QRValidationResult(
                is_valid=False,
                decoded_text=None,
                error=str(e),
            )

    def validate(
        self,
        image: Image.Image | np.ndarray | str | Path,
        expected_content: str | None = None,
    ) -> QRValidationResult:
        """Verify that a QR code is scannable and optionally matches expected content."""
        result = self.decode(image)
        if not result.is_valid:
            return result

        if expected_content is not None:
            if result.decoded_text != expected_content:
                logger.warning(
                    f"QR code content mismatch! Expected: '{expected_content}', Got: '{result.decoded_text}'"
                )
                return QRValidationResult(
                    is_valid=False,
                    decoded_text=result.decoded_text,
                    error=f"Decoded content '{result.decoded_text}' did not match expected '{expected_content}'",
                    points=result.points,
                )

        return result


def validate_qr_code(
    image: Image.Image | np.ndarray | str | Path,
    expected_content: str | None = None,
) -> QRValidationResult:
    """Convenience helper to validate a QR code image."""
    return QRValidator().validate(image, expected_content=expected_content)
