"""
bench_kernels.py — measure each exported whisper-blaze kernel against its
PyTorch equivalent, at Whisper large-v3 shapes, on H100.

Run from the repo root so the compiled extension is importable:
    PYTHONPATH=. python benchmarks/bench_kernels.py

Timing: CUDA events, warmup then median of N runs (median, not mean — GPU
timings have outliers).
"""
import json
import statistics
import sys

import torch

sys.path.insert(0, ".")
import whisper_blaze_kernels as K  # noqa: E402
from whisper_blaze.model import BlazeLinear  # noqa: E402
from whisper_blaze.precision import mixed_fp8, full_fp16  # noqa: E402

DEV = "cuda"
WARMUP, ITERS = 20, 100


def timed(fn, warmup=WARMUP, iters=ITERS):
    """Median milliseconds per call, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


results = {}


def report(group, name, blaze_ms, torch_ms, note=""):
    speedup = torch_ms / blaze_ms if blaze_ms > 0 else float("nan")
    results.setdefault(group, []).append({
        "case": name, "blaze_ms": round(blaze_ms, 4),
        "torch_ms": round(torch_ms, 4), "speedup": round(speedup, 3), "note": note,
    })
    flag = "faster" if speedup > 1.05 else ("SLOWER" if speedup < 0.95 else "parity")
    print(f"  {name:<28} blaze {blaze_ms:8.3f} ms   torch {torch_ms:8.3f} ms   "
          f"{speedup:5.2f}x  {flag}  {note}")


# --------------------------------------------------------------------------
print("\n=== 1. FP8Linear (the mixed_fp8/aggressive_fp8 path) vs nn.Linear ===")
# Whisper large-v3: d_model 1280, ffn 5120
for label, (M, K_in, N) in {
    "FFN up   [512,1280]x[1280,5120]":  (512, 1280, 5120),
    "FFN down [512,5120]x[5120,1280]":  (512, 5120, 1280),
    "QKV      [512,1280]x[1280,3840]":  (512, 1280, 3840),
    "FFN up   bs4 [2048,1280]":         (2048, 1280, 5120),
}.items():
    x = torch.randn(M, K_in, device=DEV, dtype=torch.float16)
    base = torch.nn.Linear(K_in, N, bias=False).to(DEV).half()

    fp8 = BlazeLinear(base.weight.data.clone(), None,
                      mixed_fp8().encoder_ffn).to(DEV)

    t_torch = timed(lambda: base(x))
    t_fp8 = timed(lambda: fp8(x))
    report("fp8_linear", label, t_fp8, t_torch)

# --------------------------------------------------------------------------
print("\n=== 2. Fused LayerNorm (residual + LN) vs torch ===")
for M in (512, 2048):
    h = torch.randn(M, 1280, device=DEV, dtype=torch.float16)
    r = torch.randn(M, 1280, device=DEV, dtype=torch.float16)
    g = torch.ones(1280, device=DEV, dtype=torch.float16)
    b = torch.zeros(1280, device=DEV, dtype=torch.float16)
    ln = torch.nn.LayerNorm(1280, eps=1e-5).to(DEV).half()

    t_blaze = timed(lambda: K.layernorm_fused(h, r, g, b, 1e-5))
    t_torch = timed(lambda: ln(h + r))
    report("layernorm", f"residual+LN [{M},1280]", t_blaze, t_torch)

# --------------------------------------------------------------------------
print("\n=== 3. Flash Attention 3 kernels vs torch SDPA ===")
# Whisper large-v3: 20 heads x 64 dim; encoder seq 1500, decoder short
for label, (B, H, S, D, fn, causal) in {
    "encoder self  [1,20,1500,64]": (1, 20, 1500, 64, K.encoder_self_attn, False),
    "decoder cross [1,20,448,64]":  (1, 20, 448, 64, K.decoder_cross_attn, False),
    "decoder self  [1,20,448,64]":  (1, 20, 448, 64, K.decoder_self_attn, True),
}.items():
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=DEV, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=DEV, dtype=torch.float16)
    try:
        t_blaze = timed(lambda: fn(q, k, v))
        t_torch = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal))
        report("attention", label, t_blaze, t_torch)
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<28} FAILED: {type(exc).__name__}: {exc}")
        results.setdefault("attention", []).append({"case": label, "error": str(exc)})

# --------------------------------------------------------------------------
print("\n=== 4. Quantise / dequantise round trip (pure overhead) ===")
for M in (512, 2048):
    x = torch.randn(M, 1280, device=DEV, dtype=torch.float16)
    t_q = timed(lambda: K.quantise_e4m3(x))
    q, s = K.quantise_e4m3(x)
    t_dq = timed(lambda: K.dequantise_e4m3(q, s, [M, 1280]))
    print(f"  quantise   [{M},1280]        {t_q:8.3f} ms")
    print(f"  dequantise [{M},1280]        {t_dq:8.3f} ms")
    results.setdefault("quant", []).append(
        {"case": f"[{M},1280]", "quantise_ms": round(t_q, 4),
         "dequantise_ms": round(t_dq, 4)})

print("\n" + json.dumps(results, indent=2))
with open("benchmarks/results_kernels.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote benchmarks/results_kernels.json")
