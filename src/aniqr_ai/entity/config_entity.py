from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class QRGenerationConfig:
    """Configuration options for QR code generation."""

    error_correction: Literal["L", "M", "Q", "H"] = "H"
    box_size: int = 10
    border: int = 4
    fill_color: str = "black"
    back_color: str = "white"
    target_size: tuple[int, int] | None = None
    root_dir: Path | None = None
    output_image_path: Path | None = None

    def __post_init__(self) -> None:
        if self.error_correction not in ("L", "M", "Q", "H"):
            raise ValueError("error_correction must be one of: L, M, Q, H")
        if self.box_size < 1:
            raise ValueError("box_size must be at least 1")
        if self.border < 4:
            raise ValueError("border must be at least 4 modules for reliable scanning")


@dataclass(frozen=True)
class QRValidationConfig:
    """Configuration options for QR code validation."""

    detector: str = "opencv"
    root_dir: Path | None = None
