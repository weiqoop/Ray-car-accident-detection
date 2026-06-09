# syntax=docker/dockerfile:1.7
# Single image: CUDA 12.1 runtime + Python 3.10 venv + Ray + PyTorch (cu121) + ultralytics.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        libgl1 libglib2.0-0 ffmpeg \
        git curl ca-certificates \
        tzdata \
 && rm -rf /var/lib/apt/lists/*

# venv at /opt/venv (matches user's preference for venv-isolated installs)
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

WORKDIR /workspace

# Install torch first in its own layer for cache reuse.
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

# Source is bind-mounted at runtime by docker-compose so edits don't need a
# rebuild. COPY here too so the image is self-contained for bare `docker run`.
# (Add project-specific dirs like ./reference/ ./data/ etc. as you need them.)
COPY src/     ./src/
COPY scripts/ ./scripts/

EXPOSE 8265

# Default command is a no-op. Override via docker-compose `command:` or
# `docker compose run --rm train python scripts/your_script.py`.
CMD ["python", "-c", "print('ray-framework image: override CMD to run a script')"]
