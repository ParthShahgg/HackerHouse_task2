# ============================================================================
# Voice RAG over MSMARCO-XI
#
# Two-stage build. The heavy layer is torch + the ~4.9GB of model weights, so
# weights are baked in at build time: a cold container that must download 4.9GB
# before serving its first request is not a deployable service.
#
# Default target is CPU. For GPU, build with:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
#                --build-arg DEVICE=cuda -f deploy/Dockerfile .
# ============================================================================

FROM python:3.12-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG DEVICE=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false \
    DEVICE=${DEVICE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch first, from the hardware-specific index. On CPU this avoids pulling
# ~2.5GB of CUDA libraries that would never be used.
RUN pip install --index-url ${TORCH_INDEX} torch==2.9.1

COPY requirements.txt .
# torch is already installed above; strip it so pip does not re-resolve it from
# the default index and undo the CPU/CUDA choice.
RUN grep -v '^torch==' requirements.txt > /tmp/reqs.txt \
    && pip install -r /tmp/reqs.txt

# ---- bake model weights into the image -------------------------------------
COPY app/config.py app/languages.py app/__init__.py ./app/
COPY scripts/download_models.py ./scripts/
RUN mkdir -p /models && python scripts/download_models.py

# ---- application ------------------------------------------------------------
COPY app ./app
COPY configs ./configs
COPY frontend ./frontend
COPY scripts ./scripts
COPY pyproject.toml ./

# Non-root. The app only needs to write /models cache metadata and /app/data.
RUN mkdir -p /app/data /app/reports \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# Reports degradation (empty index, missing secrets) rather than only liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker on purpose: the models are held in-process and each worker would
# be another ~4.9GB of resident weights. Scale with replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
