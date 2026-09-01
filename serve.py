"""
whisper-blaze API server.

OpenAI-compatible transcription endpoint backed by WhisperBlaze with
dynamic cross-request batching: concurrent requests collected within
BATCH_WAIT_MS are fused into a single transcribe_batch() GPU pass.

Environment
-----------
MODEL_ID       HF model id            (default: openai/whisper-large-v3)
PRECISION      full_fp16              (default: full_fp16)
BATCH_WAIT_MS  batching window in ms  (default: 200)
MODE           fast | accurate        (default: fast)
               fast: stitches long audio by matching transcript strings over a
                 2s chunk overlap - maximum throughput (~420 RTFx, 11.8% WER)
               accurate: stitches on Whisper's own timestamps - roughly half
                 the word error rate (6.8%), about 2x slower (~215 RTFx)
               Selectable per request with the `mode` form field.
VRAM_LIMIT_GB  cap GPU memory use, e.g. 30 to use 30 GB of an 80 GB H100
               (default: 0 = use all available VRAM)
PORT           listen port            (default: 8000)
"""

import asyncio
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper-blaze-serve")

MODEL_ID      = os.environ.get("MODEL_ID", "openai/whisper-large-v3")
PRECISION     = os.environ.get("PRECISION", "full_fp16")
BATCH_WAIT_MS = float(os.environ.get("BATCH_WAIT_MS", "200"))
MODE          = os.environ.get("MODE", "fast")
VRAM_LIMIT_GB = float(os.environ.get("VRAM_LIMIT_GB", "0"))  # 0 = no limit
SAMPLE_RATE   = 16000
CHUNK_SEC     = 30

# Batch sizing under a VRAM limit, measured on H100 with large-v3 and real
# speech (linear fit, max residual 0.03 GB; identical across precision
# presets since FP8 quantization is in-kernel and weights stay FP16):
#   peak_gb = 2.92 + 0.249 * chunks
# _SAFETY derates for decode-length variance across workloads.
_MODEL_OVERHEAD_GB = 3.0
_GB_PER_CHUNK      = 0.25
_SAFETY            = 0.8
_max_chunks: Optional[int] = None  # None = unlimited

app = FastAPI(title="whisper-blaze", version="0.1.14")

_model = None
_queue: Optional[asyncio.Queue] = None
_gpu_executor = ThreadPoolExecutor(max_workers=1)  # one GPU worker


def _apply_vram_limit():
    """Hard-cap this process's CUDA allocations to VRAM_LIMIT_GB."""
    global _max_chunks
    if VRAM_LIMIT_GB <= 0:
        return
    import torch
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    fraction = min(1.0, VRAM_LIMIT_GB / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    _max_chunks = max(1, int((VRAM_LIMIT_GB - _MODEL_OVERHEAD_GB)
                             / _GB_PER_CHUNK * _SAFETY))
    log.info("VRAM limit: %.0f GB of %.0f GB (fraction %.2f), max %d chunks/pass",
             VRAM_LIMIT_GB, total_gb, fraction, _max_chunks)
    if VRAM_LIMIT_GB < _MODEL_OVERHEAD_GB + 2:
        log.warning("VRAM_LIMIT_GB=%.0f is very low; model weights alone need "
                    "~%.0f GB — expect failures below that.",
                    VRAM_LIMIT_GB, _MODEL_OVERHEAD_GB)


def _est_chunks(audio: np.ndarray) -> int:
    """Estimate how many 30s chunks this audio contributes to a GPU pass."""
    return max(1, int(np.ceil(len(audio) / (CHUNK_SEC * SAMPLE_RATE))))


def _load_model():
    from whisper_blaze import WhisperBlaze
    from whisper_blaze import precision as wp

    preset = {
        "full_fp16":      wp.full_fp16,
        "mixed_fp8":      wp.mixed_fp8,
        "aggressive_fp8": wp.aggressive_fp8,
    }.get(PRECISION)
    if preset is None:
        raise ValueError(f"Unknown PRECISION={PRECISION!r}")
    return WhisperBlaze.from_pretrained(MODEL_ID, precision=preset())


def _decode_audio(data: bytes) -> np.ndarray:
    """Decode any soundfile/ffmpeg-readable format to mono float32 @ 16 kHz."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        # fall back to librosa (uses audioread/ffmpeg for mp3 etc.)
        audio, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # [samples, ch] from soundfile or [ch, samples] from librosa
        audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    # The batched path builds one mel per 30s chunk and Whisper requires the
    # full 3000 frames, so clips shorter than 30s must be zero-padded.
    min_samples = CHUNK_SEC * SAMPLE_RATE
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    return audio


async def _batch_worker():
    """Collect requests for BATCH_WAIT_MS, then run one fused GPU pass per
    (language, task) group."""
    loop = asyncio.get_running_loop()
    carry = None  # item deferred from the previous round by the chunk budget
    while True:
        first = carry if carry is not None else await _queue.get()
        carry = None
        batch = [first]
        chunks = first["chunks"]
        deadline = loop.time() + BATCH_WAIT_MS / 1000.0
        while _max_chunks is None or chunks < _max_chunks:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(_queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            if _max_chunks is not None and chunks + item["chunks"] > _max_chunks:
                carry = item  # over budget — defer to the next GPU pass
                break
            batch.append(item)
            chunks += item["chunks"]

        groups = {}
        for item in batch:
            groups.setdefault(
                (item["language"], item["task"], item["mode"]), []).append(item)

        for (language, task, mode), items in groups.items():
            audios = [it["audio"] for it in items]
            t0 = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    _gpu_executor,
                    lambda: _model.transcribe_batch(audios, language=language,
                                                    task=task, mode=mode),
                )
                dt = time.perf_counter() - t0
                log.info("batch=%d language=%s task=%s mode=%s gpu_time=%.2fs",
                         len(items), language, task, mode, dt)
                for it, res in zip(items, results):
                    it["future"].set_result(res)
            except Exception as exc:  # noqa: BLE001 — propagate to every waiter
                for it in items:
                    if not it["future"].done():
                        it["future"].set_exception(exc)


@app.on_event("startup")
async def _startup():
    global _model, _queue
    _queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    _apply_vram_limit()
    _model = await loop.run_in_executor(None, _load_model)
    asyncio.create_task(_batch_worker())
    log.info("ready: model=%s precision=%s batch_wait=%sms",
             MODEL_ID, PRECISION, BATCH_WAIT_MS)


@app.get("/health")
async def health():
    import torch
    return {
        "status": "healthy" if _model is not None else "loading",
        "model": MODEL_ID,
        "precision": PRECISION,
        "batch_wait_ms": BATCH_WAIT_MS,
        "mode": MODE,
        "vram_limit_gb": VRAM_LIMIT_GB or None,
        "max_chunks_per_pass": _max_chunks,
        "queue_depth": _queue.qsize() if _queue else 0,
        "vram_used_gb": round(torch.cuda.memory_allocated() / 2**30, 1),
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    response_format: str = Form("json"),
    mode: Optional[str] = Form(None),
):
    if _model is None:
        raise HTTPException(503, "model is still loading")
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, f"invalid task {task!r}")
    mode = mode or MODE
    from whisper_blaze.model import WhisperBlaze as _WB
    if mode not in _WB.MODES:
        raise HTTPException(
            400, f"invalid mode {mode!r}; expected one of {sorted(_WB.MODES)}")

    data = await file.read()
    try:
        audio = _decode_audio(data)
    except Exception as exc:
        raise HTTPException(400, f"could not decode audio: {exc}") from exc

    future = asyncio.get_running_loop().create_future()
    await _queue.put({"audio": audio, "language": language, "task": task,
                      "mode": mode, "chunks": _est_chunks(audio),
                      "future": future})
    result = await future

    text = result["text"] if isinstance(result, dict) else str(result)
    if response_format == "text":
        return text
    return {"text": text, "language": (result.get("language") if isinstance(result, dict) else language)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
