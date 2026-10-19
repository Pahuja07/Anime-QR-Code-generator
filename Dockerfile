# Base image with CUDA 12.1 support for RunPod Serverless GPU workers
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface \
    TORCH_HOME=/workspace/cache/torch

# Install system dependencies including OpenGL/GLib for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace Diffusers and AI inference libraries
RUN pip install --no-cache-dir \
    'diffusers>=0.28.0' \
    'transformers>=4.40.0' \
    'accelerate>=0.30.0' \
    'safetensors>=0.4.3' \
    'opencv-python-headless>=4.9.0' \
    'qrcode[pil]>=7.4.2' \
    'Pillow>=10.2.0' \
    'pyyaml>=6.0.1' \
    'runpod>=1.6.0' \
    'razorpay>=2.0.0'

# Copy application source code
COPY config /workspace/config
COPY src /workspace/src
COPY runpod_handler.py /workspace/runpod_handler.py
COPY setup.py /workspace/setup.py

# Install AniQR package in editable mode
RUN pip install --no-cache-dir -e .

# Cache directory volume
VOLUME /workspace/cache

# RunPod Serverless worker entrypoint (default target)
CMD ["python3", "-u", "runpod_handler.py"]
