"""
bench_overlap.py — does the 5-second chunk overlap earn its cost?

transcribe_batch() walks audio with a 25 s stride over 30 s windows, so every
chunk is re-decoded 5 s deep and _merge_chunks() strips the duplicate text by
string matching. That is ~20% more chunks than non-overlapping windows.

LibriSpeech utterances are ~7 s, so they never exercise chunking. This builds
long-form audio by concatenating utterances (reference = concatenated
transcripts) and measures WER and speed across stride settings.

    PYTHONPATH=. python benchmarks/bench_overlap.py [n_utterances]
"""
import io
import json
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from jiwer import wer

sys.path.insert(0, ".")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SR = 16000
MODEL_ID = "openai/whisper-large-v3"

print(f"building long-form audio from {N} LibriSpeech utterances ...")
ds = load_dataset("librispeech_asr", "clean", split=f"test[:{N}]")
ds = ds.cast_column("audio", Audio(decode=False))

pieces, refs = [], []
for rec, text in zip(ds["audio"], ds["text"]):
    data = rec["bytes"] if rec.get("bytes") else open(rec["path"], "rb").read()
    a, sr_in = sf.read(io.BytesIO(data), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    pieces.append(np.asarray(a, dtype=np.float32))
    refs.append(text)
    # a short gap between utterances, as in natural speech
    pieces.append(np.zeros(int(0.3 * SR), dtype=np.float32))

audio = np.concatenate(pieces)
reference = " ".join(refs)
total_s = len(audio) / SR
print(f"  {total_s/60:.1f} min of audio, {len(reference.split())} reference words\n")


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


from whisper_blaze import WhisperBlaze          # noqa: E402
from whisper_blaze.precision import full_fp16   # noqa: E402

model = WhisperBlaze.from_pretrained(MODEL_ID, precision=full_fp16())
model.transcribe_batch([audio[:SR * 60]], language="en", task="transcribe")  # warmup

results = {}
# 30 = no overlap; 25 = the shipped default (5 s overlap); 28 = 2 s overlap
import itertools
for ts_merge, stride in itertools.product((True, False), (30, 28, 25)):
    model.USE_TIMESTAMP_MERGE = ts_merge
    model.STRIDE_SEC = stride
    n_chunks = len(range(0, len(audio), stride * SR))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    out = model.transcribe_batch([audio], language="en", task="transcribe")
    dt = time.perf_counter() - t0

    w = wer(norm(reference), norm(out[0]["text"]))
    peak = torch.cuda.max_memory_allocated() / 2**30
    tag = f"{'timestamp' if ts_merge else 'string':>9}_merge_stride_{stride}s"
    results[tag] = {
        "overlap_s": 30 - stride, "chunks": n_chunks,
        "wer": round(w * 100, 2), "wall_s": round(dt, 2),
        "rtfx": round(total_s / dt, 1), "peak_vram_gb": round(peak, 2),
    }
    print(f"  {tag:<28} chunks {n_chunks:3d}  "
          f"WER {w*100:6.2f}%  {dt:6.2f}s  RTFx {total_s/dt:6.1f}x  peak {peak:5.2f} GB")

print("\n=== verdict ===")
best = min(results.items(), key=lambda kv: kv[1]["wer"])
print(f"  lowest WER: {best[0]}  WER {best[1]['wer']}%  RTFx {best[1]['rtfx']}x  "
      f"chunks {best[1]['chunks']}")

with open("benchmarks/results_overlap.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote benchmarks/results_overlap.json")
