"""
whisper_blaze
============
Hopper-native CUDA kernel library for Whisper large-v3 on H100 GPUs.

Provides:
  - Custom WGMMA + TMA GEMM kernels (FP16 and FP8 E4M3/E5M2)
  - Flash Attention 3 for encoder self-attn, decoder cross-attn, decoder self-attn
  - Fused residual + LayerNorm / RMSNorm
  - GPU-native mel spectrogram (replaces HuggingFace CPU preprocessor)
  - FP8 quantise / dequantise with per-tensor scaling

Quick start:
    from whisper_blaze import WhisperBlaze
    from whisper_blaze.precision import mixed_fp8

    model = WhisperBlaze.from_pretrained(
        "openai/whisper-large-v3",
        precision=mixed_fp8(),
    )
    result = model.transcribe(audio_tensor)
    print(result["text"])
"""

from .model     import WhisperBlaze
from .precision import (
    Precision,
    PrecisionConfig,
    LayerPrecision,
    full_fp16,
    mixed_fp8,
    aggressive_fp8,
    from_name as precision_from_name,
)
from .processor import WhisperBlazeProcessor

__all__ = [
    "WhisperBlaze",
    "WhisperBlazeProcessor",
    "Precision",
    "PrecisionConfig",
    "LayerPrecision",
    "full_fp16",
    "mixed_fp8",
    "aggressive_fp8",
    "precision_from_name",
]

__version__ = "0.1.17"
