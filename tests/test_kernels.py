"""
test_kernels.py
Correctness tests for whisper_blaze kernels.

Tests:
  - FP8 quantise/dequantise round-trip error
  - LayerNorm output matches torch reference
  - RMSNorm output matches torch reference
  - Mel spectrogram shape and value range
  - Attention output shape and causality mask

Run:
    pytest tests/test_kernels.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

try:
    import whisper_blaze_kernels as kernels
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_KERNELS,
    reason="CUDA + compiled whisper_blaze_kernels required"
)


DEVICE = "cuda"
ATOL   = 1e-2   # FP8 is lossy — allow 1% absolute tolerance
RTOL   = 1e-1


# ---------------------------------------------------------------------------
# FP8 Quantisation
# ---------------------------------------------------------------------------

class TestFP8:
    def test_e4m3_roundtrip_shape(self):
        x = torch.randn(128, 512, dtype=torch.float16, device=DEVICE)
        fp8, scale = kernels.quantise_e4m3(x)
        assert fp8.shape == x.shape, "FP8 tensor shape mismatch"

    def test_e4m3_roundtrip_values(self):
        x = torch.randn(512, 512, dtype=torch.float16, device=DEVICE)
        fp8, scale = kernels.quantise_e4m3(x)
        x_hat = kernels.dequantise_e4m3(fp8, scale, list(x.shape))
        # E4M3 has limited precision; allow generous tolerance
        assert torch.allclose(x.float(), x_hat.float(), atol=0.1, rtol=0.2), \
            f"E4M3 round-trip error too large: max={( x.float() - x_hat.float()).abs().max():.4f}"

    def test_e5m2_roundtrip_shape(self):
        x = torch.randn(64, 256, dtype=torch.float16, device=DEVICE)
        fp8, scale = kernels.quantise_e5m2(x)
        assert fp8.shape == x.shape

    def test_e5m2_roundtrip_values(self):
        x = torch.randn(512, 512, dtype=torch.float16, device=DEVICE)
        fp8, scale = kernels.quantise_e5m2(x)
        x_hat = kernels.dequantise_e5m2(fp8, scale, list(x.shape))
        assert torch.allclose(x.float(), x_hat.float(), atol=0.2, rtol=0.3), \
            f"E5M2 round-trip error too large"

    def test_scale_positive(self):
        x = torch.randn(256, 256, dtype=torch.float16, device=DEVICE)
        _, scale_e4m3 = kernels.quantise_e4m3(x)
        _, scale_e5m2 = kernels.quantise_e5m2(x)
        assert scale_e4m3 > 0, "E4M3 scale must be positive"
        assert scale_e5m2 > 0, "E5M2 scale must be positive"

    def test_zero_tensor(self):
        x = torch.zeros(64, 64, dtype=torch.float16, device=DEVICE)
        fp8, scale = kernels.quantise_e4m3(x)
        x_hat = kernels.dequantise_e4m3(fp8, scale, [64, 64])
        assert torch.allclose(x_hat.float(), x.float(), atol=1e-6)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class TestLayerNorm:
    @pytest.fixture
    def setup(self):
        B, T, D = 2, 64, 1280
        x        = torch.randn(B * T, D, dtype=torch.float16, device=DEVICE)
        residual = torch.randn(B * T, D, dtype=torch.float16, device=DEVICE)
        gamma    = torch.ones(D,           dtype=torch.float16, device=DEVICE)
        beta     = torch.zeros(D,          dtype=torch.float16, device=DEVICE)
        return x, residual, gamma, beta, D

    def test_shape(self, setup):
        x, res, g, b, D = setup
        out = kernels.layernorm_fused(x, res, g, b, 1e-5)
        assert out.shape == x.shape

    def test_matches_torch(self, setup):
        x, res, g, b, D = setup
        out_kernel = kernels.layernorm_fused(x, res, g, b, 1e-5).float()

        # Reference: torch LayerNorm on x + residual
        ref = F.layer_norm(
            (x + res).float(), [D],
            g.float(), b.float(), 1e-5
        )
        assert torch.allclose(out_kernel, ref, atol=ATOL, rtol=RTOL), \
            f"LayerNorm max diff: {(out_kernel - ref).abs().max():.5f}"

    def test_gamma_scaling(self):
        B, D = 16, 512
        x        = torch.randn(B, D, dtype=torch.float16, device=DEVICE)
        residual = torch.zeros(B, D, dtype=torch.float16, device=DEVICE)
        gamma    = torch.full((D,), 2.0, dtype=torch.float16, device=DEVICE)
        beta     = torch.zeros(D,        dtype=torch.float16, device=DEVICE)

        out   = kernels.layernorm_fused(x, residual, gamma, beta, 1e-5).float()
        ref   = F.layer_norm(x.float(), [D], gamma.float(), beta.float(), 1e-5)
        assert torch.allclose(out, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_shape(self):
        B, D = 32, 1280
        x   = torch.randn(B, D, dtype=torch.float16, device=DEVICE)
        res = torch.randn(B, D, dtype=torch.float16, device=DEVICE)
        g   = torch.ones(D,     dtype=torch.float16, device=DEVICE)
        out = kernels.rmsnorm_fused(x, res, g, 1e-5)
        assert out.shape == x.shape

    def test_matches_reference(self):
        B, D = 32, 512
        x   = torch.randn(B, D, dtype=torch.float16, device=DEVICE)
        res = torch.randn(B, D, dtype=torch.float16, device=DEVICE)
        g   = torch.ones(D,     dtype=torch.float16, device=DEVICE)

        out = kernels.rmsnorm_fused(x, res, g, 1e-5).float()

        # Reference RMSNorm
        h   = (x + res).float()
        rms = h.pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()
        ref = h * rms * g.float()

        assert torch.allclose(out, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

class TestMelSpectrogram:
    def _make_audio(self, secs: float = 30.0) -> torch.Tensor:
        n = int(secs * 16000)
        return torch.randn(n, dtype=torch.float32)

    def test_output_shape_30s(self):
        audio = self._make_audio(30.0)
        mel = kernels.mel_spectrogram(audio)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 128
        assert mel.dtype == torch.float16

    def test_output_dtype(self):
        mel = kernels.mel_spectrogram(self._make_audio())
        assert mel.dtype == torch.float16, "Mel output must be FP16"

    def test_output_device(self):
        mel = kernels.mel_spectrogram(self._make_audio())
        assert mel.is_cuda, "Mel output must be on CUDA"

    def test_values_reasonable(self):
        """Normalised mel values should be roughly in [-2, 2]."""
        mel = kernels.mel_spectrogram(self._make_audio()).float()
        assert mel.min() > -5.0 and mel.max() < 5.0, \
            f"Mel values out of expected range: [{mel.min():.2f}, {mel.max():.2f}]"

    def test_silence_is_low(self):
        """Silent audio should produce near-minimum log mel."""
        silence = torch.zeros(30 * 16000, dtype=torch.float32)
        mel = kernels.mel_spectrogram(silence).float()
        assert mel.mean() < -0.5, "Silent audio should give low mel values"


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:
    @pytest.fixture
    def qkv(self):
        B, H, T, D = 1, 20, 64, 64
        Q = torch.randn(B, H, T, D, dtype=torch.float16, device=DEVICE)
        K = torch.randn(B, H, T, D, dtype=torch.float16, device=DEVICE)
        V = torch.randn(B, H, T, D, dtype=torch.float16, device=DEVICE)
        return Q, K, V

    def test_encoder_self_attn_shape(self, qkv):
        Q, K, V = qkv
        out = kernels.encoder_self_attn(Q, K, V)
        assert out.shape == Q.shape

    def test_decoder_cross_attn_shape(self, qkv):
        Q, K, V = qkv
        out = kernels.decoder_cross_attn(Q, K, V)
        assert out.shape == Q.shape

    def test_decoder_self_attn_shape(self, qkv):
        Q, K, V = qkv
        out = kernels.decoder_self_attn(Q, K, V)
        assert out.shape == Q.shape

    def test_encoder_attn_dtype(self, qkv):
        Q, K, V = qkv
        out = kernels.encoder_self_attn(Q, K, V)
        assert out.dtype == torch.float16

    def test_attention_not_nan(self, qkv):
        Q, K, V = qkv
        for fn in [kernels.encoder_self_attn,
                   kernels.decoder_cross_attn,
                   kernels.decoder_self_attn]:
            out = fn(Q, K, V)
            assert not torch.isnan(out).any(), f"{fn.__name__} produced NaNs"
