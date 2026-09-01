# whisper-blaze — Hopper-native Whisper serving for H100
#
# Build:  docker build -t whisper-blaze:0.1.12 .
# Run:    docker run --gpus all -p 8000:8000 whisper-blaze:0.1.12
#
# The kernels are compiled for sm_90a (H100/H200 only).

ARG CUDA_VERSION=12.4.1
ARG TORCH_VERSION=2.6.0

# ---------- build stage: compile the CUDA extension ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build
ARG TORCH_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /src
COPY setup.py pyproject.toml MANIFEST.in README.md LICENSE ./
COPY csrc ./csrc
COPY whisper_blaze ./whisper_blaze

RUN pip3 wheel . --no-build-isolation --no-deps -w /wheels

# ---------- runtime stage ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
ARG TORCH_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cu124

COPY --from=build /wheels /wheels
RUN pip3 install --no-cache-dir /wheels/*.whl \
        "transformers>=4.36.0" numpy \
        fastapi "uvicorn[standard]" python-multipart soundfile librosa \
    && rm -rf /wheels

LABEL org.opencontainers.image.source="https://github.com/techbysaurabh/whisper-blaze" \
      org.opencontainers.image.description="Hopper-native CUDA kernels for Whisper large-v3 on H100 — WGMMA, TMA, FP8, FA3, dynamic batching" \
      org.opencontainers.image.licenses="MIT"

COPY serve.py /app/serve.py
WORKDIR /app

ENV MODEL_ID=openai/whisper-large-v3 \
    PRECISION=mixed_fp8 \
    BATCH_WAIT_MS=200 \
    PORT=8000 \
    HF_HOME=/data/hf

VOLUME /data/hf
EXPOSE 8000

CMD ["python3", "serve.py"]
