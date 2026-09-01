# whisper-blaze benchmarks — measured results

All figures measured 2026-09-01 on a **dedicated, idle NVIDIA H100 80GB HBM3**
(no other GPU tenants), SM clocks locked at 1980 MHz, persistence mode on.
torch 2.6.0+cu124, transformers 5.5.0, ctranslate2 4.5.0, whisper-large-v3.

Reproduce:

```bash
PYTHONPATH=. python benchmarks/bench_kernels.py          # kernel microbenchmarks
PYTHONPATH=. python benchmarks/bench_e2e.py <audio-dir>  # throughput
PYTHONPATH=. python benchmarks/bench_wer.py 100          # accuracy
```

Raw numbers: `benchmarks/results_kernels.json`, `results_e2e.json`,
`results_wer.json`.

---

## Headline finding: the CUDA kernels do not run

`WhisperBlaze._patch_layers()` (`whisper_blaze/model.py:263`) has a body of
`pass`. `BlazeLinear`, `BlazeAttention`, and `BlazeLayerNorm` are defined but
never instantiated anywhere in the package. `WhisperBlaze.from_pretrained()`
therefore returns a stock HuggingFace FP16 model, and the `precision=` argument
changes nothing at runtime.

This is confirmed end-to-end, not just by reading the source: across 100
LibriSpeech utterances, `full_fp16`, `mixed_fp8`, and `aggressive_fp8` produced
**character-identical transcripts, 100/100**, at identical WER and speed.

What the package actually provides today is HuggingFace `transformers` plus
30-second chunking and cross-request batching.

## Kernel microbenchmarks

CUDA events, 20 warmup + median of 100 runs, Whisper large-v3 shapes.
Correctness measured as relative L2 error against a float32 PyTorch reference.

| Kernel | Correct? | vs PyTorch | Reached by the model? |
|---|---|---|---|
| `layernorm_fused` | yes — `rel_l2 3.1e-04` | **2.48–2.52× faster** | no |
| `encoder_self_attn` | **no — `rel_l2 1.00`** | 0.37× (2.7× slower) | no |
| `decoder_cross_attn` | **no — `rel_l2 9.06`** | 0.81× | no |
| `decoder_self_attn` | **no — `rel_l2 4.50`** | 0.99× | no |
| `mel_spectrogram` | **no — emits 2998 frames, Whisper requires 3000** | 4.9× faster | no |
| FP8 `BlazeLinear` | numerically fine (2.6% quantisation error) | **0.39–0.53× (≈2.4× slower)** | no |
| WGMMA/TMA GEMM | — | — | **no Python binding exists** |

Notes:

- A relative L2 error of 1.0 means the attention output is essentially
  uncorrelated with the reference. A layout mismatch was ruled out first:
  `bindings.cpp` documents `[B,H,Sq,D]`, which is what both the test and
  `model.py` pass.
- The FP8 path is slower because it quantises to FP8 and then **dequantises
  back to FP16** before calling `nn.functional.linear`. The source comment
  reads "TMA descriptors would be pre-built in production; simplified here."
  Quantise costs 0.035 ms and dequantise 0.010 ms per call, on top of the
  unchanged matmul.
- `gemm_hopper.cu` compiles but is not exported in `csrc/ops/bindings.cpp`, so
  the headline WGMMA GEMM cannot be called from Python at all.
- `layernorm_fused` is genuinely correct and genuinely fast. It is the one
  kernel worth keeping as-is — it simply is not wired to anything.

## End-to-end throughput

Three real call-centre recordings (385.5 s + 318.5 s + 613.9 s = **22.0 min**),
Hindi → English translation. RTFx = seconds of audio per second of wall clock;
higher is better.

| Engine | RTFx | Wall | Peak VRAM |
|---|---|---|---|
| **stock HF, batched** | **510.2×** | 2.58 s | 14.32 GB |
| whisper-blaze `transcribe_batch()` `mixed_fp8` | 379.1× | 3.48 s | 16.70 GB |
| whisper-blaze `transcribe_batch()` `full_fp16` | 372.2× | 3.54 s | 16.70 GB |
| whisper-blaze `transcribe()` serial `mixed_fp8` | 208.5× | 6.32 s | 9.21 GB |
| whisper-blaze `transcribe()` serial `full_fp16` | 203.8× | 6.47 s | 9.21 GB |
| faster-whisper batched(16) | 155.4× | 8.48 s | not measurable¹ |
| stock HF, chunk-at-a-time (batch=1) | 33.5× | 39.32 s | 3.16 GB |
| faster-whisper serial | 14.3× | 92.41 s | not measurable¹ |

¹ CTranslate2 allocates outside PyTorch's caching allocator, so
`torch.cuda.max_memory_allocated()` cannot see it.

**whisper-blaze's batching is real** — 1.8× over its own serial path, and 11×
over naive chunk-at-a-time HuggingFace.

**But an equivalent ~20-line HuggingFace batching loop is 37% faster and uses
14% less VRAM.** The likely cause is chunk overlap: `transcribe_batch()` walks
the audio with a 25 s stride over 30 s windows (5 s overlap), producing 54
chunks for this corpus where non-overlapping 30 s windows produce 45 — about
20% more work. That overlap may buy accuracy at chunk boundaries; this corpus
does not isolate that effect, so it is untested either way.

## Accuracy — LibriSpeech test-clean, 100 utterances (11.2 min)

| Engine | WER | RTFx |
|---|---|---|
| faster-whisper fp16 | **1.76%** | 26.2× |
| whisper-blaze `full_fp16` | 2.83% | 16.6× |
| whisper-blaze `mixed_fp8` | 2.83% | 16.8× |
| whisper-blaze `aggressive_fp8` | 2.83% | 16.7× |

Preset outputs identical to `full_fp16`: **100/100** for both `mixed_fp8` and
`aggressive_fp8`.

RTFx is low for every engine here because these are ~7-second utterances, where
per-call overhead dominates. On this set faster-whisper is both more accurate
and faster.

## Environment issues found while benchmarking

- **ctranslate2 4.8.2 fails on H100** with `parallel_for failed:
  cudaErrorInvalidDevice: invalid device ordinal` on any GPU call. 4.5.0 works.
- **Installing `faster-whisper` pulls `nvidia-cudnn-cu13`** alongside torch's
  `nvidia-cudnn-cu12`, which breaks torch with `CUDNN_STATUS_NOT_INITIALIZED`
  on the first conv. Uninstalling the cu13 package restores both.
- `python benchmarks/foo.py` puts `benchmarks/` on `sys.path`, not the repo
  root, so the compiled extension is invisible. Run with `PYTHONPATH=.` — the
  pre-existing `bench_gemm.py` silently reported "kernels not available" and
  fell back to torch-only numbers because of this.

## What the numbers support

Honest claims available today:

- Batched transcription is **11× faster than naive chunk-at-a-time
  HuggingFace**, and fuses concurrent requests into one GPU pass.
- Memory is predictable and linear: `peak_gb = 2.92 + 0.249 × chunks`
  (max residual 0.03 GB), which is what makes `VRAM_LIMIT_GB` capping possible.
- The fused LayerNorm kernel is correct and 2.5× faster than PyTorch.

Claims **not** supported by the code as it stands: any FP8 speedup, any WGMMA
or TMA GEMM benefit, any Flash-Attention-3 benefit, and "replaces standard
PyTorch operations with hand-tuned CUDA kernels".
