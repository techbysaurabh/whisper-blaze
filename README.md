# whisper-blaze

High-throughput batched serving for [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) on NVIDIA H100, with a fused Hopper LayerNorm kernel.

- **Dynamic cross-request batching** — concurrent requests are fused into a
  single `model.generate()` pass instead of being decoded one chunk at a time.
- **VRAM capping for shared GPUs** — `VRAM_LIMIT_GB=24` runs transcription in
  24 GB of an 80 GB card and sizes each GPU batch to fit the budget.
- **Two long-form modes, selectable per request** — `fast` (default) stitches
  chunks by matching transcript text across a short overlap; `accurate`
  stitches on Whisper's own segment timestamps for materially lower word error
  rate, at roughly half the throughput.
- **OpenAI-compatible server in one command** — point existing OpenAI audio
  clients at it by changing the base URL.
- **Fused residual + LayerNorm CUDA kernel** — a hand-written Hopper kernel,
  used for all 162 LayerNorms in the model.

## Requirements

| Component | Version |
|---|---|
| GPU | NVIDIA H100 (Hopper, SM90) |
| CUDA toolkit | 12.2+ (12.6 recommended) |
| PyTorch | 2.1.0+ with matching CUDA |
| Python | 3.9+ |
| OS | Linux x86_64 |

## Installation

**Step 1 — Install PyTorch with CUDA support** (if you haven't already):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Step 2 — Install whisper-blaze:**

```bash
pip install whisper-blaze --no-build-isolation
```

> `--no-build-isolation` is **required** — it tells pip to use your existing PyTorch
> instead of fetching it into an isolated build environment.

**From source:**

```bash
git clone https://github.com/techbysaurabh/whisper-blaze.git
cd whisper-blaze
pip install -e . --no-build-isolation
```

If your CUDA toolkit isn't at `/usr/local/cuda`, set `CUDA_HOME` first:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
```

## Run with Docker

The fastest way to serve whisper-blaze — no local CUDA toolkit needed:

```bash
docker run --gpus all -p 8000:8000 -v hf-cache:/data/hf \
  ghcr.io/techbysaurabh/whisper-blaze:latest
```

Also on Docker Hub: `techbysaurabh/whisper-blaze`. Model weights download from
Hugging Face on first start and are cached in the volume.

The container exposes an OpenAI-compatible transcription API with dynamic
cross-request batching (concurrent requests fuse into a single GPU pass):

```bash
curl -F file=@audio.mp3 -F language=en localhost:8000/v1/audio/transcriptions
```

Configure via env vars: `MODEL_ID` (any HF Whisper checkpoint), `PRECISION`
`BATCH_WAIT_MS`, `MODE` (`fast` / `accurate`), `PORT`,
`HF_TOKEN`. Health check at `GET /health`. See `serve.py` and `Dockerfile`
for details.

**Sharing the GPU?** Set `VRAM_LIMIT_GB` to cap how much VRAM the server uses —
e.g. `-e VRAM_LIMIT_GB=30` runs transcription in 30 GB of an 80 GB H100,
leaving the rest for other workloads. This applies a hard allocator cap
(`torch.cuda.set_per_process_memory_fraction`) and automatically sizes GPU
batches to fit the budget.

## Quick Start

```python
from whisper_blaze import WhisperBlaze
from whisper_blaze.precision import full_fp16

model = WhisperBlaze.from_pretrained(
    "openai/whisper-large-v3",
    precision=full_fp16(),
)

# Single file — numpy array or torch tensor, float32, 16 kHz
# 1D [samples] or 2D [channels, samples] both accepted
result = model.transcribe(audio, language="en")
print(result["text"])
```

## Batch Transcription

`transcribe_batch()` accepts multiple audio files and fuses all their 30-second
chunks into a **single `model.generate()` call**, maximising VRAM utilisation on
an 80 GB H100.

```python
# results is a list of dicts, one per input audio
results = model.transcribe_batch(
    [audio1, audio2, audio3],
    language="en",
    task="transcribe",
)
for r in results:
    print(r["text"])
```

**Why it matters:** a single 15-minute file uses ~40 GB VRAM. With
`transcribe_batch()` you can process a second 15-minute file in the same GPU
pass, using ~78 GB — the remaining 40 GB that would otherwise sit idle.

Longer audio produces more internal chunks and uses more VRAM; shorter audio
batches more requests into the same GPU pass. The batcher automatically caps
batch size to stay within the available VRAM budget.

## Serving at Scale

For production deployments, pair whisper-blaze with a dynamic batching API server
that keeps a pool of concurrent requests in-flight and automatically groups them
into GPU batches:

```
Client pool (10 concurrent)
        │
        ▼
  FastAPI server              ← collect requests for 400 ms
        │
        ▼
  transcribe_batch()          ← one model.generate() for the whole batch
        │
        ▼
  Results returned individually
```

Dynamic batching delivers near-linear throughput scaling as concurrent requests
increase, with idle VRAM automatically absorbed by larger batch sizes.

## GPU Mel Spectrogram

```python
from whisper_blaze import WhisperBlazeProcessor

proc = WhisperBlazeProcessor(device="cuda")
mel = proc(audio_tensor, sampling_rate=16000)   # [1, 128, T] fp16 on GPU

# Long audio with overlapping chunks
mels = proc.process_chunks(long_audio, sampling_rate=16000, overlap_s=1.0)
```

## Direct Kernel API

```python
import torch
import whisper_blaze_kernels as k

# FP8 quantize / dequantize
x = torch.randn(512, 512, dtype=torch.float16, device="cuda")
fp8, scale = k.quantise_e4m3(x)
x_back = k.dequantise_e4m3(fp8, scale, [512, 512])

# Fused residual + LayerNorm
out = k.layernorm_fused(hidden, residual, gamma, beta, 1e-5)

# Fused RMSNorm
out = k.rmsnorm_fused(hidden, residual, gamma, 1e-5)

# Flash Attention 3
out = k.encoder_self_attn(Q, K, V)    # no causal mask
out = k.decoder_self_attn(Q, K, V)    # causal mask
out = k.decoder_cross_attn(Q, K, V)   # no causal mask

# GPU mel spectrogram
mel = k.mel_spectrogram(audio_cpu_float32)  # → [1, 128, T] fp16 on GPU
```

## Troubleshooting

**`RuntimeError: CUDA version mismatch`** — Your PyTorch was compiled against a different CUDA version than your system toolkit. Reinstall PyTorch from the correct index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**`ninja not found`** — Install ninja for faster builds:

```bash
pip install ninja
```

**`nvcc does not support sm_90a`** — Upgrade your CUDA toolkit to 12.2+. The H100 Hopper architecture requires `sm_90a`.

## License

MIT
