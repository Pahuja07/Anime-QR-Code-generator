"""Model loading and caching utilities for Stable Diffusion + ControlNet."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    base_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_id: str = "monster-labs/control_v1p_sd15_qrcode_monster"
    device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    enable_xformers: bool = False
    cpu_offload: bool = False


class ModelLoader:
    """Singleton-style or cached loader for StableDiffusionControlNetPipeline."""

    _instance: ModelLoader | None = None
    _pipeline: Any = None
    _loaded_config: ModelConfig | None = None

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    @classmethod
    def get_instance(cls, config: ModelConfig | None = None) -> ModelLoader:
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _resolve_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        import torch

        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        if self.config.torch_dtype == "auto":
            dtype = torch.float16 if device.type == "cuda" else torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config.torch_dtype, torch.float32)

        return device, dtype

    def load_pipeline(self) -> Any:
        """Load and return the StableDiffusionControlNetPipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            device, dtype = self._resolve_device_and_dtype()
            logger.info(
                f"Loading models: base='{self.config.base_model_id}', "
                f"controlnet='{self.config.controlnet_id}' on {device} ({dtype})"
            )

            try:
                from diffusers import (
                    ControlNetModel,
                    DPMSolverMultistepScheduler,
                    StableDiffusionControlNetPipeline,
                )
            except ImportError as exc:
                raise AniQRException(
                    "diffusers is required for anime QR inference. Install the AI requirements."
                ) from exc

            controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_id,
                torch_dtype=dtype,
            )

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.config.base_model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )

            if device.type == "cuda" and self.config.cpu_offload:
                pipe.enable_model_cpu_offload()
            elif device.type == "cuda":
                if self.config.enable_xformers:
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception as ex:
                        logger.warning(f"Could not enable xformers: {ex}")
                pipe = pipe.to(device)
            else:
                pipe = pipe.to(device)

            self._pipeline = pipe
            self._loaded_config = self.config
            logger.info("Stable Diffusion + ControlNet pipeline loaded successfully!")
            return self._pipeline

        except Exception as e:
            logger.error(f"Failed to load AI pipeline: {e}")
            raise AniQRException(f"Model loading failed: {e}") from e
