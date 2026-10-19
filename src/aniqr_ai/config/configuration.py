from pathlib import Path
from typing import Any
import yaml

from aniqr_ai.entity.config_entity import QRGenerationConfig, QRValidationConfig
from aniqr_ai.logger import logger
from aniqr_ai.exception import AniQRException


def read_yaml(path_to_yaml: Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary."""
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Loaded YAML file successfully from: {path_to_yaml}")
            return content or {}
    except Exception as e:
        raise AniQRException(f"Error reading YAML file at {path_to_yaml}: {e}") from e


def create_directories(paths: list[Path | str]) -> None:
    """Create a list of directories if they do not exist."""
    for path in paths:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory verified/created: {p}")


class ConfigurationManager:
    """Manages reading project configuration and providing config objects."""

    def __init__(self, config_filepath: str | Path = "config/config.yaml"):
        self.config_filepath = Path(config_filepath)
        self.config = read_yaml(self.config_filepath)

        artifacts_root = self.config.get("artifacts_root", "artifacts")
        create_directories([artifacts_root])

    def get_qr_generation_config(self) -> QRGenerationConfig:
        """Get the QR generation configuration."""
        config_data = self.config.get("qr_generation", {})
        root_dir = Path(config_data.get("root_dir", "artifacts/qr_generation"))
        output_image_path = Path(config_data.get("output_image_path", "artifacts/qr_generation/generated_qr.png"))

        create_directories([root_dir, output_image_path.parent])

        target_size = config_data.get("target_size")
        if target_size and isinstance(target_size, list):
            target_size = tuple(target_size)

        return QRGenerationConfig(
            error_correction=config_data.get("error_correction", "H"),
            box_size=config_data.get("box_size", 10),
            border=config_data.get("border", 4),
            fill_color=config_data.get("fill_color", "black"),
            back_color=config_data.get("back_color", "white"),
            target_size=target_size,
            root_dir=root_dir,
            output_image_path=output_image_path,
        )

    def get_qr_validation_config(self) -> QRValidationConfig:
        """Get the QR validation configuration."""
        config_data = self.config.get("qr_validation", {})
        root_dir = Path(config_data.get("root_dir", "artifacts/qr_validation"))
        create_directories([root_dir])

        return QRValidationConfig(
            detector=config_data.get("detector", "opencv"),
            root_dir=root_dir,
        )
