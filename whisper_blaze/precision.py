"""
precision.py
Precision configuration for whisper_blaze.

Defines which compute precision (FP16 / FP8-E4M3 / FP8-E5M2) to use
for each layer type in the Whisper model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Precision(str, Enum):
    FP16     = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"

    def is_fp8(self) -> bool:
        return self in (Precision.FP8_E4M3, Precision.FP8_E5M2)

    def max_val(self) -> float:
        return {
            Precision.FP16:     65504.0,
            Precision.FP8_E4M3: 448.0,
            Precision.FP8_E5M2: 57344.0,
        }[self]


@dataclass
class LayerPrecision:
    """Per-operation precision for one transformer layer."""
    weight: Precision = Precision.FP16
    activation: Precision = Precision.FP16

    def is_fp8(self) -> bool:
        return self.weight.is_fp8() or self.activation.is_fp8()


@dataclass
class PrecisionConfig:
    """
    Full model precision configuration.

    Attributes
    ----------
    encoder_ffn          : FFN layers in encoder
    encoder_attn_qkv     : QKV projection in encoder attention
    encoder_attn_out     : Output projection in encoder attention
    decoder_self_attn    : Decoder masked self-attention
    decoder_cross_attn   : Decoder cross-attention (wide range → E5M2 acts)
    decoder_ffn          : FFN layers in decoder
    """
    encoder_ffn:       LayerPrecision = field(default_factory=LayerPrecision)
    encoder_attn_qkv:  LayerPrecision = field(default_factory=LayerPrecision)
    encoder_attn_out:  LayerPrecision = field(default_factory=LayerPrecision)
    decoder_self_attn: LayerPrecision = field(default_factory=LayerPrecision)
    decoder_cross_attn:LayerPrecision = field(default_factory=LayerPrecision)
    decoder_ffn:       LayerPrecision = field(default_factory=LayerPrecision)

    def summary(self) -> str:
        lines = ["PrecisionConfig:"]
        for name, lp in self.__dict__.items():
            lines.append(f"  {name:<22} weight={lp.weight.value:<10} act={lp.activation.value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def full_fp16() -> PrecisionConfig:
    """All FP16. Maximum quality, no quantisation."""
    fp16 = LayerPrecision(Precision.FP16, Precision.FP16)
    return PrecisionConfig(
        encoder_ffn       = fp16,
        encoder_attn_qkv  = fp16,
        encoder_attn_out  = fp16,
        decoder_self_attn = fp16,
        decoder_cross_attn= fp16,
        decoder_ffn       = fp16,
    )


def mixed_fp8() -> PrecisionConfig:
    """
    FP8 E4M3 on all large matmuls (FFN + QKV), FP16 where precision matters.
    Recommended starting point — best quality/throughput trade-off.
    """
    e4m3 = LayerPrecision(Precision.FP8_E4M3, Precision.FP8_E4M3)
    fp16 = LayerPrecision(Precision.FP16,     Precision.FP16)
    return PrecisionConfig(
        encoder_ffn        = e4m3,
        encoder_attn_qkv   = e4m3,
        encoder_attn_out   = fp16,  # post-softmax activations vary widely
        decoder_self_attn  = fp16,  # causal attention is precision-sensitive
        decoder_cross_attn = fp16,  # cross-attn scores span large range
        decoder_ffn        = e4m3,
    )


def aggressive_fp8() -> PrecisionConfig:
    """
    FP8 everywhere possible.
    Decoder cross-attention uses E5M2 for activations (wider dynamic range).
    Maximum throughput, slight quality trade-off.
    """
    e4m3       = LayerPrecision(Precision.FP8_E4M3, Precision.FP8_E4M3)
    e4m3_e5m2  = LayerPrecision(Precision.FP8_E4M3, Precision.FP8_E5M2)
    return PrecisionConfig(
        encoder_ffn        = e4m3,
        encoder_attn_qkv   = e4m3,
        encoder_attn_out   = e4m3,
        decoder_self_attn  = e4m3,
        decoder_cross_attn = e4m3_e5m2,  # wide range acts → E5M2
        decoder_ffn        = e4m3,
    )


# Name → factory map for CLI / config-file use
PRESETS: dict[str, callable] = {
    "fp16":           full_fp16,
    "mixed_fp8":      mixed_fp8,
    "aggressive_fp8": aggressive_fp8,
}


def from_name(name: str) -> PrecisionConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown precision preset '{name}'. "
                         f"Available: {list(PRESETS.keys())}")
    return PRESETS[name]()
