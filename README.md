# whisper-blaze

Hopper-native CUDA kernels for [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) on NVIDIA H100 GPUs.

Replaces standard PyTorch operations with hand-tuned CUDA kernels that exploit H100-specific hardware:

- **WGMMA** (Warpgroup MMA) GEMM in FP16 and FP8 (E4M3 / E5M2)
- **TMA** (Tensor Memory Accelerator) async bulk copy
- **Flash Attention 3** for encoder self-attention, decoder self/cross-attention
- **Fused residual + LayerNorm / RMSNorm**
- **GPU mel spectrogram** (replaces CPU librosa/HuggingFace preprocessor)
- **FP8 quantize/dequantize** with per-tensor scaling

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
git clone https://github.com/YOUR_USERNAME/whisper-blaze.git
cd whisper-blaze
pip install -e . --no-build-isolation
```

If your CUDA toolkit isn't at `/usr/local/cuda`, set `CUDA_HOME` first:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
```

## Quick Start

```python
from whisper_blaze import WhisperBlaze
from whisper_blaze.precision import mixed_fp8

model = WhisperBlaze.from_pretrained(
    "openai/whisper-large-v3",
    precision=mixed_fp8(),
)

result = model.transcribe(audio_tensor, language="en")
print(result["text"])
```

## Precision Presets

| Preset | When to use |
|---|---|
| `full_fp16()` | Maximum quality, no quantization |
| `mixed_fp8()` | **Recommended** — FP8 on FFN/QKV, FP16 on attention |
| `aggressive_fp8()` | Maximum throughput, FP8 everywhere |

```python
from whisper_blaze.precision import full_fp16, mixed_fp8, aggressive_fp8

model = WhisperBlaze.from_pretrained(precision=aggressive_fp8())
```

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
