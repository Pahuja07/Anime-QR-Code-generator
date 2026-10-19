"""FastAPI web service for AniQR: Generation, Credits, and Razorpay Checkout."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Literal
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from aniqr_ai.components.anime_image_generation import (
    ANIME_STYLES,
    AnimeGenerationParams,
)
from aniqr_ai.components.credit_management import (
    CreditManager,
    InsufficientCreditsError,
)
from aniqr_ai.components.payment_processing import (
    CREDIT_PACKAGES,
    RazorpayProcessor,
)
from aniqr_ai.components.qr_generation import normalize_content
from aniqr_ai.components.qr_validation import QRValidator
from aniqr_ai.components.runpod_inference import (
    RunPodInferenceClient,
    RunPodTimeoutError,
    RunPodUnscannableError,
)
from aniqr_ai.pipeline.anime_qr_pipeline import AnimeQRPipeline
from aniqr_ai.pipeline.payment_pipeline import PaymentPipeline
from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger

load_dotenv()

app = FastAPI(
    title="AniQR API",
    description="Production-ready Anime QR Code Generator with ControlNet and Razorpay Credits",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ANIQR_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
credit_manager = CreditManager.get_instance()
razorpay_processor = RazorpayProcessor()
payment_pipeline = PaymentPipeline(credit_manager=credit_manager, processor=razorpay_processor)
anime_pipeline = AnimeQRPipeline()
runpod_inference = RunPodInferenceClient()

# In-memory sliding window rate limiter
RATE_LIMIT_STORE: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60.0  # seconds
RATE_LIMIT_MAX_REQUESTS = 20  # requests per window
RATE_LIMIT_LOCK = threading.Lock()
LOCAL_INFERENCE_LOCK = threading.Lock()


class UnscannableQRCodeError(Exception):
    """Generation completed but the QR image failed its decode check."""


def check_rate_limit(client_id: str) -> None:
    with RATE_LIMIT_LOCK:
        now = time.time()
        # Remove stale clients so this in-process store does not grow without bound.
        for stale_client, stamps in list(RATE_LIMIT_STORE.items()):
            fresh = [stamp for stamp in stamps if now - stamp < RATE_LIMIT_WINDOW]
            if fresh:
                RATE_LIMIT_STORE[stale_client] = fresh
            else:
                del RATE_LIMIT_STORE[stale_client]
        timestamps = RATE_LIMIT_STORE.get(client_id, [])
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please wait a minute before making more requests.",
            )
        timestamps.append(now)
        RATE_LIMIT_STORE[client_id] = timestamps


def get_token_from_header(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Please obtain a guest token.",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use 'Bearer <guest_token>'.",
        )
    token = parts[1]
    if not credit_manager.get_account(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired guest token.",
        )
    return token


# --- Pydantic Request Models ---
class GenerateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000, description="URL, text, or phone number")
    content_type: Literal["url", "text", "phone"] = "url"
    style: Literal[
        "studio_ghibli", "cyberpunk_mecha", "retro_90s", "pastel_kawaii", "dark_fantasy", "shonen_action", "iron_man"
    ] = "studio_ghibli"
    prompt: str = Field("", max_length=500)
    negative_prompt: str = Field("", max_length=500)
    controlnet_conditioning_scale: float = Field(1.25, ge=0.5, le=2.0)
    guidance_scale: float = Field(7.5, ge=1.0, le=15.0)
    num_inference_steps: int = Field(30, ge=15, le=50)
    seed: int | None = None
    reference_image_base64: str | None = Field(None, max_length=5_600_000)


def decode_reference_image(encoded: str | None) -> Image.Image | None:
    """Decode and bound a user uploaded reference image before inference."""
    if not encoded:
        return None
    raw_value = encoded.split(",", 1)[1] if encoded.startswith("data:") and "," in encoded else encoded
    try:
        raw = base64.b64decode(raw_value, validate=True)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Uploaded image must be 4 MB or smaller.")
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Upload a PNG, JPEG, or WebP image.")
        if image.width > 4096 or image.height > 4096:
            raise ValueError("Uploaded image dimensions must be 4096 x 4096 or smaller.")
        image.load()
        return image.convert("RGB")
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid reference image: {exc}") from exc


class CreateOrderRequest(BaseModel):
    package_id: str


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    package_id: str


# --- API Routes ---
@app.get("/api/v1/health")
def healthcheck():
    return {"status": "healthy", "service": "AniQR AI", "timestamp": time.time()}


@app.post("/api/v1/auth/guest")
def issue_guest_token(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)
    account = credit_manager.create_guest_account(initial_credits=3)
    return {
        "guest_token": account.token,
        "credits": account.credits,
        "message": "Welcome to AniQR! 3 complimentary credits have been added to your trial session.",
    }


@app.get("/api/v1/credits")
def get_credit_balance(authorization: str | None = Header(None)):
    token = get_token_from_header(authorization)
    balance = credit_manager.get_balance(token)
    return {"guest_token": token, "credits": balance}


@app.get("/api/v1/payments/packages")
def list_packages():
    return {"packages": CREDIT_PACKAGES}


@app.post("/api/v1/payments/create-order")
def create_order(req: CreateOrderRequest, authorization: str | None = Header(None)):
    token = get_token_from_header(authorization)
    try:
        order = payment_pipeline.create_checkout_order(token=token, package_id=req.package_id)
        return {
            "order_id": order.order_id,
            "amount": order.amount_subunits,
            "currency": order.currency,
            "amount_inr": order.amount_inr,
            "credits": order.credits,
            "key_id": order.key_id,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AniQRException as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/payments/verify")
def verify_payment(req: VerifyPaymentRequest, authorization: str | None = Header(None)):
    token = get_token_from_header(authorization)
    if req.package_id not in CREDIT_PACKAGES:
        raise HTTPException(status_code=400, detail="Invalid package_id")
    credits_to_add = CREDIT_PACKAGES[req.package_id]["credits"]

    try:
        new_balance = payment_pipeline.fulfill_payment(
            token=token,
            order_id=req.razorpay_order_id,
            payment_id=req.razorpay_payment_id,
            signature=req.razorpay_signature,
            credits=credits_to_add,
            package_id=req.package_id,
        )
        return {
            "status": "success",
            "message": f"Payment verified! Added {credits_to_add} credits.",
            "credits": new_balance,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/payments/webhook")
async def razorpay_webhook(request: Request):
    signature = request.headers.get("X-Razorpay-Signature", "")
    raw_body = await request.body()
    try:
        event_payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    success = payment_pipeline.fulfill_webhook(
        raw_body=raw_body,
        signature=signature,
        event_payload=event_payload,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Webhook signature verification failed")
    return {"status": "ok"}


@app.post("/api/v1/generate")
def generate_anime_qr(
    req: GenerateRequest,
    request: Request,
    authorization: str | None = Header(None),
):
    token = get_token_from_header(authorization)
    check_rate_limit(token)

    params = AnimeGenerationParams(
        style=req.style,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        controlnet_conditioning_scale=req.controlnet_conditioning_scale,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        seed=req.seed,
    )

    try:
        reference_image = decode_reference_image(req.reference_image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        def run_generation():
            if runpod_inference.enabled:
                remote = runpod_inference.generate(
                    {
                        "content": req.content,
                        "content_type": req.content_type,
                        "style": req.style,
                        "prompt": req.prompt,
                        "negative_prompt": req.negative_prompt,
                        "controlnet_conditioning_scale": req.controlnet_conditioning_scale,
                        "guidance_scale": req.guidance_scale,
                        "num_inference_steps": req.num_inference_steps,
                        "seed": req.seed,
                        "reference_image_base64": req.reference_image_base64,
                    }
                )
                normalized = normalize_content(req.content, req.content_type)
                image_data = remote.get("image_base64")
                if not isinstance(image_data, str):
                    raise AniQRException("RunPod response did not include an image")
                try:
                    raw_image = base64.b64decode(image_data, validate=True)
                    image = Image.open(io.BytesIO(raw_image)).convert("RGB")
                except Exception as exc:
                    raise AniQRException("RunPod returned an invalid image") from exc
                validation = QRValidator().validate(image, expected_content=normalized)
                if not remote.get("is_scannable") or not validation.is_valid:
                    raise UnscannableQRCodeError(
                        validation.error or "RunPod image failed local scan validation."
                    )
                return SimpleNamespace(
                    anime_qr_image=image,
                    is_scannable=True,
                    decoded_content=validation.decoded_text,
                    normalized_content=normalized,
                    attempts=remote.get("attempts", 1),
                    style=remote.get("style", req.style),
                    effective_conditioning_scale=remote.get(
                        "effective_conditioning_scale", req.controlnet_conditioning_scale
                    ),
                    estimated_cost_usd=remote.get("estimated_cost_usd"),
                    execution_time_ms=remote.get("execution_time_ms"),
                )

            with LOCAL_INFERENCE_LOCK:
                result = anime_pipeline.generate(
                    value=req.content,
                    content_type=req.content_type,
                    params=params,
                    reference_image=reference_image,
                )
            if not result.is_scannable:
                raise UnscannableQRCodeError(
                    result.validation_result.error or "Generated QR failed scan validation."
                )
            return result

        output, remaining_credits = payment_pipeline.execute_with_credits(
            token=token,
            cost=1,
            generation_func=run_generation,
        )

        buffered = io.BytesIO()
        output.anime_qr_image.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "is_scannable": output.is_scannable,
            "decoded_content": output.decoded_content,
            "normalized_content": output.normalized_content,
            "attempts": output.attempts,
            "style": output.style,
            "effective_conditioning_scale": output.effective_conditioning_scale,
            "estimated_cost_usd": getattr(output, "estimated_cost_usd", None),
            "execution_time_ms": getattr(output, "execution_time_ms", None),
            "credits_remaining": remaining_credits,
            "image_base64": f"data:image/png;base64,{b64_image}",
        }

    except InsufficientCreditsError as ice:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(ice),
        )
    except UnscannableQRCodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RunPodUnscannableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RunPodTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/benchmarks")
def get_benchmarks():
    """Report benchmark availability without presenting sample estimates as measurements."""
    return {
        "benchmarks": {
            "status": "not_measured",
            "successful_generation_samples": 0,
            "cost_method": "The worker reports a per-job estimate only when RUNPOD_GPU_HOURLY_RATE_USD is configured.",
        },
        "licensing": {
            "stable_diffusion_v1_5": "CreativeML Open RAIL-M; review the model license and use restrictions before commercial deployment.",
            "controlnet_qrcode_monster": "OpenRAIL++ as listed on the model card; review the exact checkpoint and license terms before commercial deployment.",
            "generated_output_ownership": "Output rights and restrictions depend on the source model licenses and applicable law.",
        },
    }


# Minimal clean Checkout & Interactive Web UI (HTML + TailwindCSS)
@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AniQR - Anime QR Code Generator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <div class="max-w-4xl mx-auto px-4 py-8">
    <header class="flex justify-between items-center pb-6 border-b border-slate-800">
      <div>
        <h1 class="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
          ✨ AniQR AI Studio
        </h1>
        <p class="text-slate-400 text-sm mt-1">Transform URLs & Numbers into Artistic Anime QR Codes</p>
      </div>
      <div class="flex items-center gap-3">
        <span class="bg-purple-950 border border-purple-700 text-purple-300 text-sm px-3 py-1 rounded-full font-medium" id="creditBadge">
          🪙 Credits: <span id="creditCount">...</span>
        </span>
        <button onclick="openTopUpModal()" class="bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white text-xs font-semibold px-3 py-1.5 rounded-lg shadow">
          + Add Credits
        </button>
      </div>
    </header>

    <main class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
      <!-- Input Panel -->
      <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 class="text-lg font-semibold mb-4 text-purple-300">1. QR Details & Style</h2>

        <div class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Content Type</label>
            <div class="grid grid-cols-3 gap-2">
              <button type="button" onclick="setType('url')" id="type_url" class="type-btn active bg-purple-600 text-white text-xs py-2 rounded-lg font-medium">URL</button>
              <button type="button" onclick="setType('text')" id="type_text" class="type-btn bg-slate-800 text-slate-300 text-xs py-2 rounded-lg font-medium">Text</button>
              <button type="button" onclick="setType('phone')" id="type_phone" class="type-btn bg-slate-800 text-slate-300 text-xs py-2 rounded-lg font-medium">Call (tel:)</button>
            </div>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Payload Content</label>
            <input id="qrContent" type="text" placeholder="https://example.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-purple-500" value="https://aniqr.ai">
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Your image (optional, PNG/JPEG/WebP, up to 4 MB)</label>
            <input id="referenceImage" type="file" accept="image/png,image/jpeg,image/webp" class="w-full text-xs text-slate-300 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-700 file:px-3 file:py-2 file:text-slate-100">
            <p class="text-xs text-slate-500 mt-1">Your image is blended into the generated anime QR art. The QR is newly generated from the content above.</p>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Anime Art Style</label>
            <select id="qrStyle" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-purple-500">
              <option value="studio_ghibli">Studio Ghibli (Watercolor & Scenic)</option>
              <option value="cyberpunk_mecha">Cyberpunk Mecha (Neon & Hologram)</option>
              <option value="retro_90s">Retro 90s (Classic Cel Shading)</option>
              <option value="pastel_kawaii">Pastel Kawaii (Sakura & Soft Glow)</option>
              <option value="dark_fantasy">Dark Fantasy (Gothic & Ethereal)</option>
              <option value="shonen_action">Shonen Action (Speedlines & Energy)</option>
              <option value="iron_man">Iron Man (Red & Gold Superhero)</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Custom Prompt (Optional)</label>
            <input id="qrPrompt" type="text" placeholder="floating islands, giant cherry blossom tree" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-purple-500">
          </div>

          <button id="genBtn" onclick="generateQR()" class="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold py-3 rounded-xl shadow-lg mt-4 transition flex items-center justify-center gap-2">
            🎨 Generate Anime QR (1 Credit)
          </button>
        </div>
      </section>

      <!-- Output Panel -->
      <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col items-center justify-center text-center">
        <h2 class="text-lg font-semibold mb-4 text-pink-300 w-full text-left">2. Scan Verification & Preview</h2>

        <div id="loadingBox" class="hidden my-8">
          <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto"></div>
          <p class="text-sm text-slate-400 mt-3 animate-pulse">Stylizing with ControlNet & verifying scannability...</p>
        </div>

        <div id="outputBox" class="hidden flex flex-col items-center">
          <div class="p-3 bg-white rounded-xl shadow-2xl mb-4">
            <img id="resultImg" src="" alt="Generated Anime QR" class="w-64 h-64 object-contain rounded-lg">
          </div>
          <div id="scanBadge" class="mb-3 px-3 py-1 text-xs font-semibold rounded-full bg-green-950 text-green-300 border border-green-700">
            ✓ 100% Scannable
          </div>
          <p class="text-xs text-slate-400 mb-3" id="decodedTextDisplay"></p>
          <a id="downloadLink" download="anime_qr.png" class="bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium px-4 py-2 rounded-lg border border-slate-700">
            ⬇️ Download PNG
          </a>
        </div>

        <div id="placeholderBox" class="text-slate-600 my-16">
          <svg class="w-16 h-16 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 4v16m8-8H4"></path></svg>
          <p class="text-sm">Configure parameters and click Generate</p>
        </div>
      </section>
    </main>
  </div>

  <!-- Modal for purchasing credits -->
  <div id="topUpModal" class="hidden fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 text-slate-100 shadow-2xl">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-lg font-bold text-purple-300">Top Up Generation Credits</h3>
        <button onclick="closeTopUpModal()" class="text-slate-400 hover:text-white">&times;</button>
      </div>
      <p class="text-xs text-slate-400 mb-4">Secured by Razorpay. Credits are added to your account instantly.</p>

      <div class="space-y-3">
        <div onclick="buyPackage('pack_starter')" class="p-3 border border-slate-700 hover:border-purple-500 rounded-xl cursor-pointer bg-slate-800 flex justify-between items-center">
          <div>
            <div class="font-semibold text-sm">Starter Pack</div>
            <div class="text-xs text-slate-400">20 Anime QR Generations</div>
          </div>
          <div class="text-right font-bold text-purple-300">₹49</div>
        </div>

        <div onclick="buyPackage('pack_pro')" class="p-3 border-2 border-purple-600 rounded-xl cursor-pointer bg-purple-950/40 flex justify-between items-center">
          <div>
            <div class="font-semibold text-sm text-pink-300">Pro Creator ⭐</div>
            <div class="text-xs text-slate-400">100 Anime QR Generations</div>
          </div>
          <div class="text-right font-bold text-pink-300">₹199</div>
        </div>

        <div onclick="buyPackage('pack_studio')" class="p-3 border border-slate-700 hover:border-purple-500 rounded-xl cursor-pointer bg-slate-800 flex justify-between items-center">
          <div>
            <div class="font-semibold text-sm">Studio Enterprise</div>
            <div class="text-xs text-slate-400">500 Anime QR Generations</div>
          </div>
          <div class="text-right font-bold text-purple-300">₹799</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentType = 'url';
    let guestToken = localStorage.getItem('aniqr_token');

    async function initSession() {
      if (!guestToken) {
        const res = await fetch('/api/v1/auth/guest', { method: 'POST' });
        const data = await res.json();
        guestToken = data.guest_token;
        localStorage.setItem('aniqr_token', guestToken);
      }
      refreshCredits();
    }

    async function refreshCredits() {
      try {
        const res = await fetch('/api/v1/credits', {
          headers: { 'Authorization': `Bearer ${guestToken}` }
        });
        if (res.ok) {
          const data = await res.json();
          document.getElementById('creditCount').innerText = data.credits;
        } else if (res.status === 401) {
          localStorage.removeItem('aniqr_token');
          guestToken = null;
          initSession();
        }
      } catch(e) { console.error(e); }
    }

    function setType(type) {
      currentType = type;
      document.querySelectorAll('.type-btn').forEach(b => {
        b.classList.remove('bg-purple-600', 'text-white');
        b.classList.add('bg-slate-800', 'text-slate-300');
      });
      const target = document.getElementById(`type_${type}`);
      target.classList.add('bg-purple-600', 'text-white');
      target.classList.remove('bg-slate-800', 'text-slate-300');
      
      const input = document.getElementById('qrContent');
      if (type === 'url') input.placeholder = 'https://example.com';
      else if (type === 'phone') input.placeholder = '+1 (555) 019-2834';
      else input.placeholder = 'Plain text message';
    }

    async function generateQR() {
      const content = document.getElementById('qrContent').value.trim();
      if (!content) return alert('Please enter QR content');
      
      const style = document.getElementById('qrStyle').value;
      const prompt = document.getElementById('qrPrompt').value.trim();
      const imageFile = document.getElementById('referenceImage').files[0];
      if (imageFile && imageFile.size > 4 * 1024 * 1024) return alert('Image must be 4 MB or smaller');
      const referenceImageBase64 = imageFile ? await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Could not read the uploaded image'));
        reader.readAsDataURL(imageFile);
      }) : null;

      document.getElementById('placeholderBox').classList.add('hidden');
      document.getElementById('outputBox').classList.add('hidden');
      document.getElementById('loadingBox').classList.remove('hidden');

      try {
        const res = await fetch('/api/v1/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${guestToken}`
          },
          body: JSON.stringify({
            content: content,
            content_type: currentType,
            style: style,
            prompt: prompt,
            reference_image_base64: referenceImageBase64
          })
        });

        const data = await res.json();
        document.getElementById('loadingBox').classList.add('hidden');

        if (!res.ok) {
          if (res.status === 402) {
            openTopUpModal();
          }
          alert(data.detail || 'Generation failed');
          document.getElementById('placeholderBox').classList.remove('hidden');
          return;
        }

        document.getElementById('resultImg').src = data.image_base64;
        document.getElementById('downloadLink').href = data.image_base64;
        document.getElementById('decodedTextDisplay').innerText = `Payload: ${data.decoded_content}`;
        document.getElementById('outputBox').classList.remove('hidden');
        refreshCredits();

      } catch(e) {
        alert('Request failed: ' + e);
        document.getElementById('loadingBox').classList.add('hidden');
        document.getElementById('placeholderBox').classList.remove('hidden');
      }
    }

    function openTopUpModal() { document.getElementById('topUpModal').classList.remove('hidden'); }
    function closeTopUpModal() { document.getElementById('topUpModal').classList.add('hidden'); }

    async function buyPackage(pkgId) {
      try {
        const res = await fetch('/api/v1/payments/create-order', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${guestToken}`
          },
          body: JSON.stringify({ package_id: pkgId })
        });
        const orderData = await res.json();
        if (!res.ok) throw new Error(orderData.detail);

        // If in simulation mode or test keys
        if (!window.Razorpay || orderData.key_id.includes('placeholder')) {
          // Complete simulation
          const verifyRes = await fetch('/api/v1/payments/verify', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${guestToken}`
            },
            body: JSON.stringify({
              razorpay_order_id: orderData.order_id,
              razorpay_payment_id: 'pay_sim_' + Date.now(),
              razorpay_signature: 'sim_sig',
              package_id: pkgId
            })
          });
          alert('Test Payment simulated successfully! Credits credited.');
          closeTopUpModal();
          refreshCredits();
          return;
        }

        const options = {
          key: orderData.key_id,
          amount: orderData.amount,
          currency: orderData.currency,
          name: 'AniQR Studio',
          description: `Purchase Credits`,
          order_id: orderData.order_id,
          handler: async function (response) {
            const verifyRes = await fetch('/api/v1/payments/verify', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${guestToken}`
              },
              body: JSON.stringify({
                razorpay_order_id: response.razorpay_order_id,
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_signature: response.razorpay_signature,
                package_id: pkgId
              })
            });
            const vData = await verifyRes.json();
            alert(vData.message || 'Payment confirmed!');
            closeTopUpModal();
            refreshCredits();
          }
        };
        const rzp = new Razorpay(options);
        rzp.open();
      } catch(e) {
        alert('Order creation failed: ' + e.message);
      }
    }

    initSession();
  </script>
</body>
</html>
"""
