"""
test_precision.py
Unit tests for the PrecisionConfig system (pure Python, no CUDA required).
"""

import pytest
from whisper_blaze.precision import (
    Precision,
    LayerPrecision,
    PrecisionConfig,
    full_fp16,
    mixed_fp8,
    aggressive_fp8,
    from_name,
)


class TestPrecision:
    def test_is_fp8(self):
        assert not Precision.FP16.is_fp8()
        assert Precision.FP8_E4M3.is_fp8()
        assert Precision.FP8_E5M2.is_fp8()

    def test_max_val(self):
        assert Precision.FP16.max_val()     == 65504.0
        assert Precision.FP8_E4M3.max_val() == 448.0
        assert Precision.FP8_E5M2.max_val() == 57344.0


class TestLayerPrecision:
    def test_default_is_fp16(self):
        lp = LayerPrecision()
        assert lp.weight     == Precision.FP16
        assert lp.activation == Precision.FP16
        assert not lp.is_fp8()

    def test_fp8_layer(self):
        lp = LayerPrecision(Precision.FP8_E4M3, Precision.FP8_E4M3)
        assert lp.is_fp8()

    def test_mixed_layer(self):
        lp = LayerPrecision(Precision.FP8_E4M3, Precision.FP8_E5M2)
        assert lp.is_fp8()


class TestPresets:
    def test_full_fp16_all_fp16(self):
        cfg = full_fp16()
        for name, lp in cfg.__dict__.items():
            assert lp.weight     == Precision.FP16, f"{name} weight should be FP16"
            assert lp.activation == Precision.FP16, f"{name} act should be FP16"

    def test_mixed_fp8_ffn_is_fp8(self):
        cfg = mixed_fp8()
        assert cfg.encoder_ffn.weight.is_fp8()
        assert cfg.decoder_ffn.weight.is_fp8()

    def test_mixed_fp8_cross_attn_is_fp16(self):
        cfg = mixed_fp8()
        # Cross attention should stay FP16 in mixed preset
        assert not cfg.decoder_cross_attn.is_fp8()

    def test_aggressive_fp8_cross_attn_has_e5m2(self):
        cfg = aggressive_fp8()
        assert cfg.decoder_cross_attn.activation == Precision.FP8_E5M2

    def test_from_name_fp16(self):
        cfg = from_name("fp16")
        assert isinstance(cfg, PrecisionConfig)
        assert cfg.encoder_ffn.weight == Precision.FP16

    def test_from_name_mixed_fp8(self):
        cfg = from_name("mixed_fp8")
        assert cfg.encoder_ffn.weight.is_fp8()

    def test_from_name_invalid(self):
        with pytest.raises(ValueError, match="Unknown precision preset"):
            from_name("nonexistent_preset")

    def test_summary_str(self):
        cfg = mixed_fp8()
        s   = cfg.summary()
        assert "PrecisionConfig" in s
        assert "encoder_ffn" in s
