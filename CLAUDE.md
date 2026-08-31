# whisper_blaze — Build Debugging Guide

## Project Overview
Custom Hopper-native CUDA kernel library for Whisper large-v3 on H100 GPUs.
Located at: `~/whisper_kernel_optimized/whisper_blaze/`

## Current Build Error

### Error
```
RuntimeError: The detected CUDA version (12.0) mismatches the version that was
used to compile PyTorch (13.0).
```

### Root Cause
The PyTorch installed in the venv was compiled against **CUDA 13.0** (a nightly
build), but the system CUDA toolkit is **12.0**. They must match for CUDA
extensions to compile.

Additional warnings observed:
- `NVIDIA driver too old (found version 12070)` — driver may not support CUDA 13.0
- `No ninja found` — build falls back to slow distutils
- `CUDA_HOME='/usr'` — CUDA is installed system-wide under `/usr`, not `/usr/local/cuda`

---

## Diagnosis Steps

Run these first to understand the environment:

```bash
# 1. Check system CUDA version
nvcc --version
cat /usr/local/cuda/version.json 2>/dev/null
cat /usr/local/cuda/version.txt  2>/dev/null
ls /usr/local/cuda-*/

# 2. Check CUDA under /usr (since CUDA_HOME='/usr')
ls /usr/lib/cuda/
dpkg -l | grep -i "cuda\|nvcc"

# 3. Check NVIDIA driver version
nvidia-smi

# 4. Check currently installed PyTorch CUDA version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built with:', torch.version.cuda)"

# 5. Check if ninja is installed
which ninja || echo "ninja not found"
```

---

## Fix

### Step 1 — Install ninja (speeds up build significantly)
```bash
uv pip install ninja
# or: apt-get install ninja-build
```

### Step 2 — Reinstall PyTorch matching system CUDA

Pick the right index URL based on your CUDA version:

| System CUDA | PyTorch index URL |
|---|---|
| 12.0 or 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |
| 11.8 | `https://download.pytorch.org/whl/cu118` |

```bash
# Example for CUDA 12.1 (covers CUDA 12.0 too):
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-build-isolation

# Verify it matches now:
python -c "import torch; print(torch.version.cuda)"
```

### Step 3 — Set CUDA_HOME correctly (if nvcc is not at /usr)
```bash
# Find where nvcc actually is:
which nvcc

# If it's at /usr/local/cuda-12.x/bin/nvcc, set:
export CUDA_HOME=/usr/local/cuda-12.x

# Add to ~/.bashrc to persist:
echo 'export CUDA_HOME=/usr/local/cuda-12.x' >> ~/.bashrc
```

### Step 4 — Build the extension
```bash
cd ~/whisper_kernel_optimized/whisper_blaze
uv pip install -e . --no-build-isolation
```

`--no-build-isolation` is required for all PyTorch CUDA extensions — it tells
uv to use the venv's torch instead of fetching it into an isolated build env.

---

## If CUDA 12.0 + H100 Hopper

H100 requires **CUDA 11.8 minimum**, but **CUDA 12.x is strongly recommended**
for full Hopper support (WGMMA, TMA, FP8). CUDA 12.0 should work for all
kernels in this library.

The kernels are compiled with `-arch=sm_90a` (in `setup.py`). This flag
requires **CUDA 11.8+** and enables:
- `wgmma.mma_async` (Warpgroup MMA)
- `cp.async.bulk.tensor` (TMA)
- FP8 types (`__nv_fp8_e4m3`, `__nv_fp8_e5m2`)

If `nvcc` does not support `sm_90a`, upgrade CUDA toolkit to 12.x.

---

## Environment the library was built for

| Component | Version |
|---|---|
| GPU | NVIDIA H100 (Hopper, SM90) |
| CUDA toolkit | 12.0+ |
| PyTorch | 2.1.0+ |
| Python | 3.9+ |
| NVCC flag | `-arch=sm_90a` |

---

## Project Structure (for reference)
```
whisper_blaze/
├── csrc/
│   ├── include/
│   │   ├── compute_traits.cuh   # FP16 / FP8 E4M3 / FP8 E5M2 type traits
│   │   ├── hopper_utils.cuh     # TMA, mbarrier, cluster, warpgroup helpers
│   │   └── wgmma_utils.cuh      # WGMMA PTX wrappers (all shapes)
│   ├── kernels/
│   │   ├── fp8_quantize.cu      # FP16 ↔ FP8 cast + scale
│   │   ├── layernorm_fused.cu   # Fused residual + LayerNorm / RMSNorm
│   │   ├── mel_spectrogram.cu   # GPU mel spectrogram (cuFFT-based)
│   │   ├── gemm_hopper.cu       # WGMMA + TMA GEMM (FP16 + FP8)
│   │   └── flash_attn_hopper.cu # Flash Attention 3 (encoder + decoder)
│   └── ops/
│       └── bindings.cpp         # PyTorch custom op registration
├── whisper_blaze/
│   ├── __init__.py
│   ├── model.py                 # WhisperBlaze.from_pretrained()
│   ├── precision.py             # PrecisionConfig: full_fp16 / mixed_fp8 / aggressive_fp8
│   └── processor.py             # GPU mel preprocessor (replaces HF AutoProcessor)
├── benchmarks/
│   ├── bench_gemm.py            # GEMM throughput vs torch.matmul
│   └── bench_mel.py             # Mel latency vs librosa CPU
├── tests/
│   ├── test_kernels.py          # CUDA kernel correctness tests
│   └── test_precision.py        # PrecisionConfig unit tests (no CUDA needed)
├── setup.py
└── pyproject.toml
```
