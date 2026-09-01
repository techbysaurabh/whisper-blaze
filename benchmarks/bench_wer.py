"""
bench_wer.py — transcription accuracy on LibriSpeech test-clean.

Answers two questions:
  1. How does whisper-blaze's WER compare to faster-whisper and stock HF?
  2. Do the precision presets (full_fp16 / mixed_fp8 / aggressive_fp8)
     actually change the output? Outputs are compared character-for-character,
     not just by WER — identical text across presets means the preset is a
     no-op at runtime.

    PYTHONPATH=. python benchmarks/bench_wer.py [n_utterances]
"""
import json
import sys
import time

import numpy as np
import torch
from jiwer import wer
from datasets import Audio, load_dataset

sys.path.insert(0, ".")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL_ID = "openai/whisper-large-v3"
SR = 16000

print(f"loading LibriSpeech test-clean ({N} utterances) ...")
ds = load_dataset("librispeech_asr", "clean", split=f"test[:{N}]")
# Decode with soundfile rather than torchcodec: installing torchcodec risks
# pulling a CUDA runtime that conflicts with the installed torch build.
ds = ds.cast_column("audio", Audio(decode=False))
import io
import soundfile as sf
audios = []
for rec in ds["audio"]:
    data = rec["bytes"] if rec.get("bytes") else open(rec["path"], "rb").read()
    a, sr_in = sf.read(io.BytesIO(data), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr_in != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr_in, target_sr=SR)
    audios.append(np.asarray(a, dtype=np.float32))
refs = [t.lower().strip() for t in ds["text"]]
total_s = sum(len(a) / SR for a in audios)
print(f"  {len(audios)} utterances, {total_s/60:.1f} min of audio\n")


def norm(s):
    """Light normalisation so punctuation/case don't dominate WER."""
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


results, transcripts = {}, {}

# ------------------------------------------------------------ whisper-blaze
from whisper_blaze import WhisperBlaze          # noqa: E402
from whisper_blaze import precision as wp       # noqa: E402

for preset in ("full_fp16", "mixed_fp8", "aggressive_fp8"):
    print(f"=== whisper-blaze {preset} ===")
    model = WhisperBlaze.from_pretrained(MODEL_ID, precision=getattr(wp, preset)())
    t0 = time.perf_counter()
    hyps = [model.transcribe(a, language="en", task="transcribe")["text"]
            for a in audios]
    dt = time.perf_counter() - t0
    w = wer([norm(r) for r in refs], [norm(h) for h in hyps])
    results[f"blaze_{preset}"] = {"wer": round(w * 100, 2), "wall_s": round(dt, 1),
                                  "rtfx": round(total_s / dt, 1)}
    transcripts[f"blaze_{preset}"] = hyps
    print(f"  WER {w*100:.2f}%   {dt:.1f}s   RTFx {total_s/dt:.1f}x\n")
    del model
    torch.cuda.empty_cache()

# Are the presets actually different? Compare text exactly.
base = transcripts.get("blaze_full_fp16", [])
for preset in ("mixed_fp8", "aggressive_fp8"):
    other = transcripts.get(f"blaze_{preset}", [])
    if base and other:
        identical = sum(a == b for a, b in zip(base, other))
        results[f"identical_to_fp16_{preset}"] = f"{identical}/{len(base)}"
        verdict = ("IDENTICAL — preset has no runtime effect"
                   if identical == len(base) else "outputs differ")
        print(f"  full_fp16 vs {preset}: {identical}/{len(base)} identical — {verdict}")

# ---------------------------------------------------------- faster-whisper
print("\n=== faster-whisper fp16 ===")
try:
    from faster_whisper import WhisperModel
    fwm = WhisperModel("large-v3", device="cuda", compute_type="float16")
    t0 = time.perf_counter()
    hyps = []
    for a in audios:
        segs, _ = fwm.transcribe(a, language="en", task="transcribe")
        hyps.append(" ".join(s.text for s in segs))
    dt = time.perf_counter() - t0
    w = wer([norm(r) for r in refs], [norm(h) for h in hyps])
    results["faster_whisper_fp16"] = {"wer": round(w * 100, 2), "wall_s": round(dt, 1),
                                      "rtfx": round(total_s / dt, 1)}
    print(f"  WER {w*100:.2f}%   {dt:.1f}s   RTFx {total_s/dt:.1f}x")
except Exception as exc:  # noqa: BLE001
    print(f"  FAILED: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
print("\n=== summary ===")
for k, v in results.items():
    if isinstance(v, dict):
        print(f"  {k:<28} WER {v['wer']:6.2f}%   RTFx {v['rtfx']:7.1f}x")
    else:
        print(f"  {k:<28} {v}")

results["_meta"] = {"dataset": "librispeech_asr test-clean", "n": len(audios),
                    "audio_min": round(total_s / 60, 1),
                    "gpu": torch.cuda.get_device_name(0)}
with open("benchmarks/results_wer.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote benchmarks/results_wer.json")
