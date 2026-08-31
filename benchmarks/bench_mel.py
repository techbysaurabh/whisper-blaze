"""
bench_mel.py
Benchmark GPU mel spectrogram vs librosa CPU baseline.

Measures latency for 30-second audio chunks.

Run:
    python benchmarks/bench_mel.py
"""

import time
import torch
import numpy as np

try:
    import whisper_blaze_kernels as kernels
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


SAMPLE_RATE   = 16_000
AUDIO_SECS    = 30
N_SAMPLES     = AUDIO_SECS * SAMPLE_RATE
WARMUP_ITERS  = 10
BENCH_ITERS   = 50


def bench_gpu_mel(audio_cpu: torch.Tensor) -> float:
    if not HAS_KERNELS:
        return float("nan")

    for _ in range(WARMUP_ITERS):
        _ = kernels.mel_spectrogram(audio_cpu)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = kernels.mel_spectrogram(audio_cpu)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / BENCH_ITERS
    return elapsed_ms


def bench_librosa_mel(audio_np: np.ndarray) -> float:
    if not HAS_LIBROSA:
        return float("nan")

    # Warmup
    for _ in range(3):
        _ = librosa.feature.melspectrogram(
            y=audio_np, sr=SAMPLE_RATE,
            n_fft=400, hop_length=160, n_mels=128,
        )

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = librosa.feature.melspectrogram(
            y=audio_np, sr=SAMPLE_RATE,
            n_fft=400, hop_length=160, n_mels=128,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000 / BENCH_ITERS
    return elapsed_ms


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  Mel Spectrogram Benchmark  —  {AUDIO_SECS}s audio at 16kHz")
    print(f"{'='*55}")

    # Synthetic audio: white noise
    audio_np  = np.random.randn(N_SAMPLES).astype(np.float32)
    audio_cpu = torch.from_numpy(audio_np)

    gpu_ms     = bench_gpu_mel(audio_cpu)
    cpu_ms     = bench_librosa_mel(audio_np)
    speedup    = cpu_ms / gpu_ms if gpu_ms > 0 and cpu_ms > 0 else float("nan")

    print(f"  {'Method':<22} {'Latency (ms)':>14}")
    print(f"  {'-'*37}")
    print(f"  {'GPU (whisper_blaze)':<22} {gpu_ms:>14.2f}" if HAS_KERNELS else
          f"  {'GPU (whisper_blaze)':<22} {'N/A':>14}")
    print(f"  {'CPU (librosa)':<22} {cpu_ms:>14.2f}" if HAS_LIBROSA else
          f"  {'CPU (librosa)':<22} {'N/A':>14}")
    if not (gpu_ms != gpu_ms) and not (cpu_ms != cpu_ms):
        print(f"\n  Speedup: {speedup:.1f}x")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
