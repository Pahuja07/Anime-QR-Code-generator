"""Anime QR code image generation using Stable Diffusion and ControlNet."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from aniqr_ai.components.model_loading import ModelConfig, ModelLoader
from aniqr_ai.components.qr_validation import QRValidator, QRValidationResult
from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger

# Style presets tailored for anime QR aesthetic blending
ANIME_STYLES = {
    "studio_ghibli": {
        "prompt": "lush green valley, floating castle, fluffy white clouds, watercolor texture, soft sunlight, Hayao Miyazaki aesthetic, scenic anime landscape",
        "negative_prompt": "ugly, blurry, low quality, dark, gloomy, modern city, distorted",
        "conditioning_scale": 1.25,
        "guidance_scale": 7.5,
    },
    "cyberpunk_mecha": {
        "prompt": "cyberpunk anime warrior with glowing neon visor, intricate mecha armor, night city background, holographic rain, vibrant cyan and magenta lights",
        "negative_prompt": "ugly, blurry, lowres, deformed, bad anatomy, flat colors, watermark",
        "conditioning_scale": 1.30,
        "guidance_scale": 8.0,
    },
    "retro_90s": {
        "prompt": "1990s retro anime aesthetic, classic cel shading, vintage anime screenshot, detailed line art, nostalgic color grading",
        "negative_prompt": "modern 3d cgi, ugly, blurry, desaturated, grainy low quality",
        "conditioning_scale": 1.25,
        "guidance_scale": 7.5,
    },
    "pastel_kawaii": {
        "prompt": "cute anime girl, sakura blossom garden, pastel pink and sky blue colors, sparkling magical light, dreamy atmosphere",
        "negative_prompt": "ugly, creepy, dark, grotesque, deformed hands, blurry",
        "conditioning_scale": 1.25,
        "guidance_scale": 7.0,
    },
    "dark_fantasy": {
        "prompt": "dark fantasy anime knight, gothic cathedral background, ethereal glowing runes, dramatic shadows, cinematic anime lighting",
        "negative_prompt": "cartoon, flat, blurry, low quality, oversaturated neon, deformed",
        "conditioning_scale": 1.35,
        "guidance_scale": 8.0,
    },
    "shonen_action": {
        "prompt": "shonen action protagonist powering up, dynamic energy aura, sharp dramatic anime shadows, heroic pose, speedlines",
        "negative_prompt": "blurry, low quality, still, peaceful, distorted face, extra limbs",
        "conditioning_scale": 1.30,
        "guidance_scale": 7.5,
    },
    "iron_man": {
        "prompt": "Iron Man in iconic red and gold powered armor, glowing circular arc reactor, heroic pose above a futuristic city at sunset, cinematic Marvel superhero comic-book illustration, luminous gold highlights, vivid red armor",
        "negative_prompt": "blurry, low quality, dull colors, extra limbs, malformed armor, obscured faceplate, text, watermark",
        "conditioning_scale": 1.15,
        "guidance_scale": 7.5,
    },
}

BASE_POSITIVE_PROMPT = "masterpiece, best quality, highly detailed, sharp focus, 8k resolution, anime illustration"
BASE_NEGATIVE_PROMPT = "ugly, blurry, low quality, lowres, deformed, distorted, bad anatomy, bad hands, disfigured modules, multiple views, text, watermark"


@dataclass
class AnimeGenerationParams:
    """Parameters passed to the anime generation process."""

    prompt: str = ""
    negative_prompt: str = ""
    style: str = "studio_ghibli"
    controlnet_conditioning_scale: float = 1.25
    control_guidance_start: float = 0.0
    control_guidance_end: float = 0.9
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: int | None = None
    max_retries: int = 3
    retry_conditioning_boost: float = 0.10

    def __post_init__(self) -> None:
        if self.style not in ANIME_STYLES:
            raise ValueError(f"Unknown anime style: {self.style}")
        if not 0.5 <= self.controlnet_conditioning_scale <= 2.0:
            raise ValueError("controlnet_conditioning_scale must be between 0.5 and 2.0")
        if not 1.0 <= self.guidance_scale <= 15.0:
            raise ValueError("guidance_scale must be between 1 and 15")
        if not 15 <= self.num_inference_steps <= 50:
            raise ValueError("num_inference_steps must be between 15 and 50")
        if not 1 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 1 and 5")
        if not 0 <= self.control_guidance_start < self.control_guidance_end <= 1:
            raise ValueError("control guidance must satisfy 0 <= start < end <= 1")
        if self.seed is not None and not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")
        if not 0 <= self.retry_conditioning_boost <= 0.5:
            raise ValueError("retry_conditioning_boost must be between 0 and 0.5")
        if len(self.prompt) > 500 or len(self.negative_prompt) > 500:
            raise ValueError("prompts must be no longer than 500 characters")


@dataclass
class AnimeQRGenerationResult:
    """Result of an anime QR generation attempt."""

    image: Image.Image
    is_scannable: bool
    decoded_content: str | None
    attempts: int
    validation_result: QRValidationResult
    style: str
    effective_conditioning_scale: float
    error_message: str | None = None


class AnimeQRGenerator:
    """Generates anime-stylized QR codes with adaptive scannability retries."""

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        validator: QRValidator | None = None,
    ) -> None:
        self.model_loader = model_loader or ModelLoader.get_instance()
        self.validator = validator or QRValidator()

    def build_prompts(
        self,
        style: str,
        user_prompt: str = "",
        user_negative_prompt: str = "",
    ) -> tuple[str, str]:
        """Combine user prompt with style preset and base quality modifiers."""
        if style not in ANIME_STYLES:
            raise ValueError(f"Unknown anime style: {style}")
        style_info = ANIME_STYLES[style]

        style_prompt = style_info["prompt"]
        style_neg = style_info["negative_prompt"]

        if user_prompt.strip():
            combined_positive = f"{user_prompt.strip()}, {style_prompt}, {BASE_POSITIVE_PROMPT}"
        else:
            combined_positive = f"{style_prompt}, {BASE_POSITIVE_PROMPT}"

        if user_negative_prompt.strip():
            combined_negative = f"{user_negative_prompt.strip()}, {style_neg}, {BASE_NEGATIVE_PROMPT}"
        else:
            combined_negative = f"{style_neg}, {BASE_NEGATIVE_PROMPT}"

        return combined_positive, combined_negative

    def _generate_single(
        self,
        pipeline: Any,
        qr_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        params: AnimeGenerationParams,
        current_conditioning_scale: float,
        seed: int,
    ) -> Image.Image:
        """Run single diffusion inference or simulation fallback."""
        if pipeline != "MOCK_DIFFUSERS_PIPELINE" and hasattr(pipeline, "__call__"):
            import torch

            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=qr_image,
                controlnet_conditioning_scale=current_conditioning_scale,
                control_guidance_start=params.control_guidance_start,
                control_guidance_end=params.control_guidance_end,
                guidance_scale=params.guidance_scale,
                num_inference_steps=params.num_inference_steps,
                generator=generator,
            )
            return output.images[0]

        # Fallback simulation for local development / testing without full GPU
        logger.info("Using simulated anime stylization (CPU/Local mode)")
        # Blend stylized colorful background with QR pattern while keeping contrast scannable
        w, h = qr_image.size
        # Create an artistic tinted canvas
        rng = random.Random(seed)
        r_hue = rng.randint(40, 180)
        g_hue = rng.randint(60, 200)
        b_hue = rng.randint(120, 240)
        bg = Image.new("RGB", (w, h), color=(r_hue, g_hue, b_hue))

        # Soften background
        bg = bg.filter(ImageFilter.GaussianBlur(radius=5))

        # Blend QR modules with background weighted by conditioning scale
        # Higher conditioning scale = stronger black modules
        qr_gray = qr_image.convert("L")
        np_qr = np.array(qr_gray, dtype=np.float32) / 255.0
        np_bg = np.array(bg, dtype=np.float32)

        # Contrast mapping ensuring scannability
        alpha = min(max(current_conditioning_scale / 1.5, 0.7), 0.95)
        blended = np.zeros_like(np_bg)
        for c in range(3):
            blended[:, :, c] = np_bg[:, :, c] * np_qr * (1.0 - alpha) + (np_qr * 255.0) * alpha

        blended_img = Image.fromarray(blended.astype(np.uint8))
        # Boost contrast slightly to assure scannability
        enhancer = ImageEnhance.Contrast(blended_img)
        return enhancer.enhance(1.2)

    @staticmethod
    def _show_art_through_qr(
        artwork: Image.Image,
        qr_image: Image.Image,
        attempt: int,
        reference_image: Image.Image | None = None,
    ) -> Image.Image:
        """Place vivid artwork in QR light modules, preserve dark modules and quiet zone."""
        size = qr_image.size
        art = np.asarray(artwork.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
        if reference_image is not None:
            # Let the user's image become part of the AI scene shown in the QR's light
            # modules while preserving the QR payload in its dark modules.
            reference = np.asarray(
                reference_image.convert("RGB").resize(size, Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            art = art * 0.45 + reference * 0.55
        qr_gray = np.asarray(qr_image.convert("L"), dtype=np.uint8)
        dark = qr_gray < 128
        coordinates = np.argwhere(dark)
        if coordinates.size == 0:
            return artwork.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        # The pixel bounds of the QR modules leave the generated four-module white border intact.
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0)
        inside = np.zeros(qr_gray.shape, dtype=bool)
        inside[y0 : y1 + 1, x0 : x1 + 1] = True
        light = (qr_gray >= 128) & inside
        dark &= inside

        # Lift dark artwork areas in light modules while preserving hue and keeping a strong
        # luminance gap from the dark QR modules. More retries favor QR contrast progressively.
        light_art = art.copy()
        luminance = light_art @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        scale = np.maximum(1.0, 218.0 / np.maximum(luminance, 1.0))
        light_art = np.minimum(light_art * scale[:, :, None], 255.0)

        dark_art = art * max(0.04, 0.16 - 0.03 * (attempt - 1))
        output = np.full_like(art, 255.0)
        output[light] = light_art[light]
        output[dark] = dark_art[dark]
        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")

    def generate(
        self,
        qr_image: Image.Image,
        expected_content: str,
        params: AnimeGenerationParams | None = None,
        reference_image: Image.Image | None = None,
    ) -> AnimeQRGenerationResult:
        """Generate an anime QR code and retry if unscannable."""
        params = params or AnimeGenerationParams()
        prompt, negative_prompt = self.build_prompts(
            style=params.style,
            user_prompt=params.prompt,
            user_negative_prompt=params.negative_prompt,
        )

        pipeline = self.model_loader.load_pipeline()
        base_seed = params.seed if params.seed is not None else random.randint(1, 1000000)
        current_scale = params.controlnet_conditioning_scale

        best_image = qr_image
        best_val_result = QRValidationResult(is_valid=False, decoded_text=None, error="Not attempted")
        last_conditioning_scale = current_scale

        for attempt in range(1, params.max_retries + 1):
            last_conditioning_scale = current_scale
            seed = base_seed + (attempt - 1) * 37
            logger.info(
                f"Anime QR generation attempt {attempt}/{params.max_retries}: "
                f"style='{params.style}', conditioning_scale={current_scale:.2f}, seed={seed}"
            )

            generated_img = self._generate_single(
                pipeline=pipeline,
                qr_image=qr_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                params=params,
                current_conditioning_scale=current_scale,
                seed=seed,
            )

            generated_img = self._show_art_through_qr(
                artwork=generated_img,
                qr_image=qr_image,
                attempt=attempt,
                reference_image=reference_image,
            )

            # Validate scannability against original normalized content
            val_result = self.validator.validate(generated_img, expected_content=expected_content)
            best_image = generated_img
            best_val_result = val_result

            if val_result.is_valid:
                logger.info(
                    f"Attempt {attempt} SUCCEEDED! Anime QR code is 100% scannable: '{val_result.decoded_text}'"
                )
                return AnimeQRGenerationResult(
                    image=generated_img,
                    is_scannable=True,
                    decoded_content=val_result.decoded_text,
                    attempts=attempt,
                    validation_result=val_result,
                    style=params.style,
                    effective_conditioning_scale=current_scale,
                )

            logger.warning(
                f"Attempt {attempt} produced unscannable image ({val_result.error}). "
                f"Boosting conditioning scale by +{params.retry_conditioning_boost:.2f}."
            )
            current_scale += params.retry_conditioning_boost

        logger.error(
            f"All {params.max_retries} generation attempts failed scannability verification."
        )
        return AnimeQRGenerationResult(
            image=best_image,
            is_scannable=False,
            decoded_content=best_val_result.decoded_text,
            attempts=params.max_retries,
            validation_result=best_val_result,
            style=params.style,
            effective_conditioning_scale=last_conditioning_scale,
            error_message="Image generated but failed scannability validation after all retries.",
        )
