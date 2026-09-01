"""
bench_e2e.py — end-to-end transcription throughput on H100.

Compares, on identical audio:
  1. whisper-blaze  transcribe()        (one file at a time)
  2. whisper-blaze  transcribe_batch()  (all files fused into one GPU pass)
  3. whisper-blaze  full_fp16 vs mixed_fp8 presets
  4. stock HuggingFace transformers     (the baseline blaze wraps)
  5. faster-whisper (CTranslate2)       (what most people actually use)

Metric: RTFx = seconds of audio transcribed per second of wall clock.
Higher is better. Peak VRAM is reported per engine.

    PYTHONPATH=. python benchmarks/bench_e2e.py <audio-dir>
"""
import gc
import glob
import json
import os
import sys
import time

import librosa
import numpy as np
import torch

sys.path.insert(0, ".")

AUDIO_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
MODEL_ID = "openai/whisper-large-v3"
LANG, TASK = "hi", "translate"
SR = 16000

files = sorted(glob.glob(os.path.join(AUDIO_DIR, "audio_*")))
audios, total_s = [], 0.0
print("=== corpus ===")
for f in files:
    try:
        a, _ = librosa.load(f, sr=SR)
    except Exception as exc:  # unreadable/corrupt file
        print(f"  skip {os.path.basename(f)} ({type(exc).__name__})")
        continue
    if len(a) / SR < 5:
        print(f"  skip {os.path.basename(f)} (too short)")
        continue
    audios.append(a.astype(np.float32))
    total_s += len(a) / SR
    print(f"  {os.path.basename(f):<48} {len(a)/SR:7.1f}s")
print(f"  {'TOTAL':<48} {total_s:7.1f}s ({total_s/60:.1f} min)\n")

results = {}


def record(name, wall_s, peak_gb, extra=""):
    rtfx = total_s / wall_s
    results[name] = {"wall_s": round(wall_s, 2), "rtfx": round(rtfx, 1),
                     "peak_vram_gb": round(peak_gb, 2), "note": extra}
    print(f"  {name:<38} {wall_s:7.2f}s  RTFx {rtfx:7.1f}x  peak {peak_gb:5.2f} GB  {extra}")


def fresh():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak():
    return torch.cuda.max_memory_allocated() / 2**30


# ---------------------------------------------------------------- whisper-blaze
print("=== whisper-blaze ===")
from whisper_blaze import WhisperBlaze          # noqa: E402
from whisper_blaze import precision as wp       # noqa: E402

for preset_name in ("full_fp16", "mixed_fp8"):
    fresh()
    t0 = time.perf_counter()
    model = WhisperBlaze.from_pretrained(MODEL_ID, precision=getattr(wp, preset_name)())
    load_s = time.perf_counter() - t0
    print(f"  [{preset_name}] model load {load_s:.1f}s")

    # warmup
    model.transcribe(audios[0][:SR * 30], language=LANG, task=TASK)

    fresh()
    t0 = time.perf_counter()
    for a in audios:
        model.transcribe(a, language=LANG, task=TASK)
    record(f"blaze {preset_name} transcribe() serial", time.perf_counter() - t0, peak())

    fresh()
    t0 = time.perf_counter()
    outs = model.transcribe_batch(audios, language=LANG, task=TASK)
    record(f"blaze {preset_name} transcribe_batch()", time.perf_counter() - t0, peak(),
           f"{len(outs)} results")

    if preset_name == "full_fp16":
        results["_sample_text"] = outs[0]["text"][:160]
    del model
    fresh()

# ------------------------------------------------------- stock HF transformers
print("\n=== stock HuggingFace transformers ===")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: E402

fresh()
hf = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    use_safetensors=True).to("cuda")
proc = AutoProcessor.from_pretrained(MODEL_ID)


def hf_transcribe(audio):
    """Chunk to 30s, one generate() per chunk — the naive baseline."""
    out = []
    for i in range(0, len(audio), SR * 30):
        chunk = audio[i:i + SR * 30]
        if len(chunk) < SR * 30:
            chunk = np.pad(chunk, (0, SR * 30 - len(chunk)))
        feats = proc(chunk, sampling_rate=SR, return_tensors="pt"
                     ).input_features.to("cuda", torch.float16)
        with torch.inference_mode():
            ids = hf.generate(feats, language=LANG, task=TASK)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    return " ".join(out)


def hf_transcribe_batched(audio_list):
    """The FAIR baseline: chunk everything to 30s, stack into ONE batch, one
    generate() call. This is exactly what transcribe_batch() does — if it
    matches blaze, the package adds packaging, not speed."""
    chunks, counts = [], []
    for a in audio_list:
        n = 0
        for i in range(0, len(a), SR * 30):
            c = a[i:i + SR * 30]
            if len(c) < SR * 30:
                c = np.pad(c, (0, SR * 30 - len(c)))
            chunks.append(c)
            n += 1
        counts.append(n)
    feats = proc(chunks, sampling_rate=SR, return_tensors="pt"
                 ).input_features.to("cuda", torch.float16)
    with torch.inference_mode():
        ids = hf.generate(feats, language=LANG, task=TASK)
    texts = proc.batch_decode(ids, skip_special_tokens=True)
    out, k = [], 0
    for n in counts:
        out.append(" ".join(texts[k:k + n]))
        k += n
    return out


hf_transcribe(audios[0][:SR * 30])  # warmup
fresh()
t0 = time.perf_counter()
for a in audios:
    hf_transcribe(a)
record("stock HF chunked batch=1 (naive)", time.perf_counter() - t0, peak())

fresh()
t0 = time.perf_counter()
hf_out = hf_transcribe_batched(audios)
record("stock HF batched (fair baseline)", time.perf_counter() - t0, peak(),
       f"{len(hf_out)} results")
del hf
fresh()

# ----------------------------------------------------------- faster-whisper
print("\n=== faster-whisper (CTranslate2) ===")
try:
    from faster_whisper import WhisperModel  # noqa: E402
    fresh()
    t0 = time.perf_counter()
    fwm = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print(f"  model load {time.perf_counter() - t0:.1f}s")

    list(fwm.transcribe(audios[0][:SR * 30], language=LANG, task=TASK)[0])  # warmup
    fresh()
    t0 = time.perf_counter()
    for a in audios:
        segs, _ = fwm.transcribe(a, language=LANG, task=TASK)
        _ = " ".join(s.text for s in segs)          # generator — must drain
    record("faster-whisper fp16", time.perf_counter() - t0, peak())

    # batched, if this version supports it
    try:
        from faster_whisper import BatchedInferencePipeline
        bp = BatchedInferencePipeline(model=fwm)
        fresh()
        t0 = time.perf_counter()
        for a in audios:
            segs, _ = bp.transcribe(a, language=LANG, task=TASK, batch_size=16)
            _ = " ".join(s.text for s in segs)
        record("faster-whisper batched(16)", time.perf_counter() - t0, peak())
    except ImportError:
        print("  (BatchedInferencePipeline not available in this version)")
except Exception as exc:  # noqa: BLE001
    print(f"  faster-whisper FAILED: {type(exc).__name__}: {exc}")
    results["faster_whisper_error"] = str(exc)

# ---------------------------------------------------------------------------
print("\n=== summary (RTFx: higher is better) ===")
ranked = sorted(((v["rtfx"], k) for k, v in results.items()
                 if isinstance(v, dict) and "rtfx" in v), reverse=True)
for rtfx, name in ranked:
    print(f"  {name:<38} {rtfx:8.1f}x")

results["_meta"] = {"total_audio_s": round(total_s, 1), "n_files": len(audios),
                    "gpu": torch.cuda.get_device_name(0), "model": MODEL_ID,
                    "language": LANG, "task": TASK}
with open("benchmarks/results_e2e.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote benchmarks/results_e2e.json")
