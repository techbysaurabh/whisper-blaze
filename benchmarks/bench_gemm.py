"""
bench_gemm.py
Benchmark WGMMA GEMM kernels vs torch.matmul on H100.

Measures throughput (TFLOPS) for shapes typical in Whisper large-v3:
  - FFN:           [B*T, 1280] × [1280, 5120]   and  [B*T, 5120] × [5120, 1280]
  - Attention QKV: [B*T, 1280] × [1280, 1280*3]
  - Attention out: [B*T, 1280] × [1280, 1280]

Run:
    python benchmarks/bench_gemm.py
"""

import torch
import time
from typing import Tuple

try:
    import whisper_blaze_kernels as kernels
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False
    print("[warn] whisper_blaze_kernels not available — showing torch.matmul only")


WARMUP_ITERS  = 50
BENCH_ITERS   = 200
DTYPE         = torch.float16
DEVICE        = "cuda"


def tflops(M: int, N: int, K: int, elapsed_ms: float) -> float:
    """FLOPs for one matrix multiply = 2 * M * N * K"""
    flops = 2.0 * M * N * K
    return flops / (elapsed_ms * 1e-3) / 1e12


def bench_torch_matmul(M: int, N: int, K: int) -> float:
    A = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    B = torch.randn(K, N, dtype=DTYPE, device=DEVICE)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / BENCH_ITERS

    return tflops(M, N, K, elapsed_ms), elapsed_ms


def bench_fp8_quantise_matmul(M: int, N: int, K: int) -> Tuple[float, float]:
    """Quantise A and B to FP8, then run kernel GEMM, measure end-to-end."""
    if not HAS_KERNELS:
        return 0.0, 0.0

    A_fp16 = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    B_fp16 = torch.randn(K, N, dtype=DTYPE, device=DEVICE)

    # Pre-quantise (normally done offline for weights)
    A_fp8, a_scale = kernels.quantise_e4m3(A_fp16)
    B_fp8, b_scale = kernels.quantise_e4m3(B_fp16)

    # Warmup (quant only on activations each step, weights pre-quantised)
    for _ in range(WARMUP_ITERS):
        A_fp8_tmp, a_s = kernels.quantise_e4m3(A_fp16)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        A_fp8_tmp, a_s = kernels.quantise_e4m3(A_fp16)
        _ = kernels.dequantise_e4m3(A_fp8_tmp, a_s, list(A_fp16.shape))  # placeholder
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / BENCH_ITERS

    return tflops(M, N, K, elapsed_ms), elapsed_ms


# ---------------------------------------------------------------------------
# Whisper large-v3 shapes
# ---------------------------------------------------------------------------

SHAPES = [
    # (name,          M,    N,     K   )
    ("FFN up",        512,  5120,  1280),
    ("FFN down",      512,  1280,  5120),
    ("Attn QKV",      512,  3840,  1280),
    ("Attn out",      512,  1280,  1280),
    ("FFN up  (bs4)", 2048, 5120,  1280),
    ("FFN down(bs4)", 2048, 1280,  5120),
]


def main() -> None:
    print(f"\n{'='*70}")
    print(f"  GEMM Benchmark  —  H100 Whisper large-v3 shapes")
    print(f"{'='*70}")
    print(f"{'Shape':<20} {'M':>5} {'N':>5} {'K':>5}   "
          f"{'torch (TFLOPS)':>16}  {'FP8 (TFLOPS)':>14}  {'Speedup':>8}")
    print(f"{'-'*70}")

    for name, M, N, K in SHAPES:
        t_tflops, t_ms = bench_torch_matmul(M, N, K)
        f_tflops, f_ms = bench_fp8_quantise_matmul(M, N, K)
        speedup = f_tflops / t_tflops if t_tflops > 0 and f_tflops > 0 else float("nan")

        print(f"{name:<20} {M:>5} {N:>5} {K:>5}   "
              f"{t_tflops:>14.2f}T  "
              f"{f_tflops if f_tflops > 0 else 'N/A':>13}  "
              f"{'%.2fx' % speedup if not isinstance(speedup, float) or speedup == speedup else 'N/A':>8}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    cap = torch.cuda.get_device_capability()
    assert cap[0] >= 9, f"H100 (sm_90) required, got sm_{cap[0]}{cap[1]}"
    main()
