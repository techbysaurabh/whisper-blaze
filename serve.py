"""
whisper-blaze API server.

OpenAI-compatible transcription endpoint backed by WhisperBlaze with
dynamic cross-request batching: concurrent requests collected within
BATCH_WAIT_MS are fused into a single transcribe_batch() GPU pass.

Environment
-----------
MODEL_ID       HF model id            (default: openai/whisper-large-v3)
PRECISION      full_fp16 | mixed_fp8 | aggressive_fp8   (default: mixed_fp8)
BATCH_WAIT_MS  batching window in ms  (default: 200)
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
PRECISION     = os.environ.get("PRECISION", "mixed_fp8")
BATCH_WAIT_MS = float(os.environ.get("BATCH_WAIT_MS", "200"))
SAMPLE_RATE   = 16000

app = FastAPI(title="whisper-blaze", version="0.1.10")

_model = None
_queue: Optional[asyncio.Queue] = None
_gpu_executor = ThreadPoolExecutor(max_workers=1)  # one GPU worker


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
    return audio


async def _batch_worker():
    """Collect requests for BATCH_WAIT_MS, then run one fused GPU pass per
    (language, task) group."""
    loop = asyncio.get_running_loop()
    while True:
        first = await _queue.get()
        batch = [first]
        deadline = loop.time() + BATCH_WAIT_MS / 1000.0
        while True:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(_queue.get(), timeout))
            except asyncio.TimeoutError:
                break

        groups = {}
        for item in batch:
            groups.setdefault((item["language"], item["task"]), []).append(item)

        for (language, task), items in groups.items():
            audios = [it["audio"] for it in items]
            t0 = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    _gpu_executor,
                    lambda: _model.transcribe_batch(audios, language=language, task=task),
                )
                dt = time.perf_counter() - t0
                log.info("batch=%d language=%s task=%s gpu_time=%.2fs",
                         len(items), language, task, dt)
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
        "queue_depth": _queue.qsize() if _queue else 0,
        "vram_used_gb": round(torch.cuda.memory_allocated() / 2**30, 1),
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    response_format: str = Form("json"),
):
    if _model is None:
        raise HTTPException(503, "model is still loading")
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, f"invalid task {task!r}")

    data = await file.read()
    try:
        audio = _decode_audio(data)
    except Exception as exc:
        raise HTTPException(400, f"could not decode audio: {exc}") from exc

    future = asyncio.get_running_loop().create_future()
    await _queue.put({"audio": audio, "language": language, "task": task,
                      "future": future})
    result = await future

    text = result["text"] if isinstance(result, dict) else str(result)
    if response_format == "text":
        return text
    return {"text": text, "language": (result.get("language") if isinstance(result, dict) else language)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
