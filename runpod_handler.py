"""RunPod Serverless Worker Handler for AniQR Anime QR Generation."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Any
import runpod

from aniqr_ai.components.anime_image_generation import (
    ANIME_STYLES,
    AnimeGenerationParams,
)
from aniqr_ai.pipeline.anime_qr_pipeline import AnimeQRPipeline
from aniqr_ai.logger import logger

# Initialize pipeline once on worker startup (cold start model loading)
_pipeline: AnimeQRPipeline | None = None
_inference_lock = threading.Lock()

def estimate_execution_cost(duration_seconds: float) -> float | None:
    """Estimate cost only when the endpoint operator supplies their actual GPU rate."""
    try:
        hourly_rate = float(os.getenv("RUNPOD_GPU_HOURLY_RATE_USD", ""))
    except ValueError:
        return None
    if hourly_rate <= 0:
        return None
    return round((duration_seconds / 3600.0) * hourly_rate, 6)


def get_pipeline() -> AnimeQRPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing AnimeQRPipeline inside RunPod worker...")
        _pipeline = AnimeQRPipeline()
    return _pipeline


def image_to_base64(img) -> str:
    """Encode PIL image to PNG base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_reference_image(encoded: Any):
    """Decode a bounded PNG/JPEG/WebP image sent with the RunPod job."""
    if encoded is None:
        return None
    if not isinstance(encoded, str) or len(encoded) > 5_600_000:
        raise ValueError("reference_image_base64 must be a base64 image no larger than 4 MB")
    raw_value = encoded.split(",", 1)[1] if encoded.startswith("data:") and "," in encoded else encoded
    raw = base64.b64decode(raw_value, validate=True)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("reference image must be 4 MB or smaller")
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    if image.format not in {"PNG", "JPEG", "WEBP"} or image.width > 4096 or image.height > 4096:
        raise ValueError("reference image must be PNG, JPEG, or WebP and at most 4096 x 4096")
    image.load()
    return image.convert("RGB")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod Serverless execution handler."""
    start_time = time.time()
    job_id = job.get("id", "local_job")
    logger.info(f"Received RunPod job [{job_id}]")

    job_input = job.get("input", {})
    if not job_input:
        return {
            "status": "FAILED",
            "error": "Missing 'input' dictionary in request payload.",
        }

    content = job_input.get("content")
    if not isinstance(content, str) or not content.strip():
        return {
            "status": "FAILED",
            "error": "The 'content' field is required and must not be empty.",
        }
    if len(content) > 1000:
        return {"status": "FAILED", "error": "Content must be no longer than 1000 characters."}

    content_type = job_input.get("content_type", "url")
    if content_type not in ("url", "text", "phone"):
        return {
            "status": "FAILED",
            "error": f"Invalid content_type '{content_type}'. Must be 'url', 'text', or 'phone'.",
        }

    style = job_input.get("style", "studio_ghibli")
    if style not in ANIME_STYLES:
        return {
            "status": "FAILED",
            "error": f"Invalid style '{style}'. Valid styles: {', '.join(ANIME_STYLES)}",
        }

    for field_name in ("prompt", "negative_prompt"):
        field_value = job_input.get(field_name, "")
        if not isinstance(field_value, str) or len(field_value) > 500:
            return {"status": "FAILED", "error": f"{field_name} must be a string of at most 500 characters."}

    try:
        # Model and generation hyperparameters
        params = AnimeGenerationParams(
            style=style,
            prompt=job_input.get("prompt", ""),
            negative_prompt=job_input.get("negative_prompt", ""),
            controlnet_conditioning_scale=float(job_input.get("controlnet_conditioning_scale", 1.25)),
            guidance_scale=float(job_input.get("guidance_scale", 7.5)),
            num_inference_steps=int(job_input.get("num_inference_steps", 30)),
            seed=job_input.get("seed"),
            max_retries=int(job_input.get("max_retries", 3)),
        )
        with _inference_lock:
            pipeline = get_pipeline()
            output = pipeline.generate(
                value=content,
                content_type=content_type,
                params=params,
                reference_image=decode_reference_image(job_input.get("reference_image_base64")),
            )

        if not output.is_scannable:
            execution_duration = time.time() - start_time
            return {
                "status": "FAILED",
                "is_scannable": False,
                "decoded_content": output.decoded_content,
                "attempts": output.attempts,
                "error": output.validation_result.error or "Generated QR did not pass scan validation.",
                "execution_time_ms": int(execution_duration * 1000),
                "estimated_cost_usd": estimate_execution_cost(execution_duration),
            }

        execution_duration = time.time() - start_time
        duration_ms = int(execution_duration * 1000)

        # Cost calculation based on GPU execution seconds
        cost_usd = estimate_execution_cost(execution_duration)

        b64_image = image_to_base64(output.anime_qr_image)

        return {
            "status": "COMPLETED",
            "is_scannable": output.is_scannable,
            "decoded_content": output.decoded_content,
            "normalized_content": output.normalized_content,
            "attempts": output.attempts,
            "style": output.style,
            "effective_conditioning_scale": output.effective_conditioning_scale,
            "image_base64": b64_image,
            "execution_time_ms": duration_ms,
            "estimated_cost_usd": cost_usd,
        }

    except Exception as e:
        logger.error(f"Job [{job_id}] failed: {e}")
        execution_duration = time.time() - start_time
        return {
            "status": "FAILED",
            "error": str(e),
            "execution_time_ms": int(execution_duration * 1000),
        }


if __name__ == "__main__":
    # Start the RunPod Serverless worker loop
    logger.info("Starting RunPod Serverless listener...")
    runpod.serverless.start({"handler": handler})
