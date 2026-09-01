"""
model.py
Drop-in Whisper large-v3 model that replaces HuggingFace attention and norm
layers with whisper_blaze Hopper-native kernels.

Usage:
    from whisper_blaze import WhisperBlaze, PrecisionConfig
    from whisper_blaze.precision import mixed_fp8

    model = WhisperBlaze.from_pretrained(
        "openai/whisper-large-v3",
        precision=mixed_fp8(),
        device="cuda",
    )
    result = model.transcribe(audio_array, language="hi")
"""

from __future__ import annotations

import re
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

from .precision import PrecisionConfig, Precision, LayerPrecision, full_fp16

try:
    import whisper_blaze_kernels as _kernels
    _KERNELS_AVAILABLE = True
except ImportError:
    _KERNELS_AVAILABLE = False

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# FP8-aware linear layer
# Wraps a standard nn.Linear and dispatches to the correct kernel
# based on layer precision config.
# ---------------------------------------------------------------------------

class BlazeLinear(nn.Module):
    """
    Linear layer that uses WGMMA (FP8 or FP16) based on PrecisionConfig.

    Falls back to standard torch.nn.functional.linear if kernels not available.
    """

    def __init__(
        self,
        weight: torch.Tensor,   # [out_features, in_features]
        bias: Optional[torch.Tensor],
        layer_prec: LayerPrecision,
    ) -> None:
        super().__init__()
        self.layer_prec = layer_prec
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias if bias is not None else torch.empty(0))
        self._has_bias = bias is not None

        # Pre-quantise weights to FP8 if configured
        self._w_fp8: Optional[torch.Tensor] = None
        self._w_scale: float = 1.0

        if _KERNELS_AVAILABLE and layer_prec.weight.is_fp8():
            self._quantise_weights()

    def _quantise_weights(self) -> None:
        w_fp16 = self.weight.half().cuda()
        if self.layer_prec.weight == Precision.FP8_E4M3:
            self._w_fp8, self._w_scale = _kernels.quantise_e4m3(w_fp16)
        else:
            self._w_fp8, self._w_scale = _kernels.quantise_e5m2(w_fp16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _KERNELS_AVAILABLE or not self.layer_prec.is_fp8():
            # FP16 path — standard linear
            return nn.functional.linear(
                x, self.weight,
                self.bias if self._has_bias else None
            )

        # FP8 path
        # Quantise activations at runtime
        a_fp8: torch.Tensor
        a_scale: float

        if self.layer_prec.activation == Precision.FP8_E4M3:
            a_fp8, a_scale = _kernels.quantise_e4m3(x.half())
        else:
            a_fp8, a_scale = _kernels.quantise_e5m2(x.half())

        # GEMM via Hopper FP8 kernel
        # (TMA descriptors would be pre-built in production; simplified here)
        out = nn.functional.linear(
            _kernels.dequantise_e4m3(a_fp8, a_scale, list(x.shape))
            if self.layer_prec.activation == Precision.FP8_E4M3
            else _kernels.dequantise_e5m2(a_fp8, a_scale, list(x.shape)),
            _kernels.dequantise_e4m3(self._w_fp8, self._w_scale, list(self.weight.shape))
            if self.layer_prec.weight == Precision.FP8_E4M3
            else _kernels.dequantise_e5m2(self._w_fp8, self._w_scale, list(self.weight.shape)),
            self.bias if self._has_bias else None,
        )
        return out


# ---------------------------------------------------------------------------
# Fused residual + LayerNorm wrapper
# ---------------------------------------------------------------------------

class BlazeLayerNorm(nn.Module):
    """Drop-in nn.LayerNorm backed by the fused CUDA kernel.

    Accepts an optional residual: passing one computes LayerNorm(x + residual)
    in a single pass. Whisper is pre-norm and adds its residual separately, so
    the model uses the residual-free form, which skips the residual read
    entirely.
    """

    #: the kernel reads 8 halves at a time, so the last dim must be a multiple of 8
    VECTOR_WIDTH = 8

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias",   bias)
        self.eps = eps

    @classmethod
    def supports(cls, ln: nn.LayerNorm) -> bool:
        """Whether this LayerNorm can be served by the kernel."""
        return (
            len(ln.normalized_shape) == 1
            and ln.normalized_shape[0] % cls.VECTOR_WIDTH == 0
            and ln.elementwise_affine
            and ln.bias is not None
        )

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm) -> "BlazeLayerNorm":
        return cls(ln.weight.data.detach().half().contiguous(),
                   ln.bias.data.detach().half().contiguous(),
                   ln.eps)

    def forward(self, x: torch.Tensor,
                residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The kernel handles contiguous FP16 CUDA tensors; anything else
        # (FP32 autocast, CPU, a non-contiguous view) goes to torch.
        if (_KERNELS_AVAILABLE and x.is_cuda and x.dtype == torch.float16
                and x.is_contiguous()
                and x.size(-1) % self.VECTOR_WIDTH == 0
                and (residual is None
                     or (residual.is_contiguous()
                         and residual.dtype == torch.float16))):
            return _kernels.layernorm_fused(
                x, residual, self.weight, self.bias, self.eps)

        hidden = x if residual is None else x + residual
        return nn.functional.layer_norm(
            hidden, [hidden.size(-1)],
            self.weight.to(hidden.dtype), self.bias.to(hidden.dtype), self.eps
        )


# ---------------------------------------------------------------------------
# Hopper Multi-Head Attention
# Delegates to flash_attn kernels based on attention type.
# ---------------------------------------------------------------------------

class BlazeAttention(nn.Module):
    def __init__(
        self,
        q_proj: BlazeLinear,
        k_proj: BlazeLinear,
        v_proj: BlazeLinear,
        out_proj: BlazeLinear,
        n_heads: int,
        head_dim: int,
        attn_type: str,   # "encoder_self", "decoder_self", "decoder_cross"
    ) -> None:
        super().__init__()
        self.q_proj   = q_proj
        self.k_proj   = k_proj
        self.v_proj   = v_proj
        self.out_proj = out_proj
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.attn_type = attn_type

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        Q = self.q_proj(x)
        if self.attn_type == "decoder_cross" and encoder_out is not None:
            K = self.k_proj(encoder_out)
            V = self.v_proj(encoder_out)
        else:
            K = self.k_proj(x)
            V = self.v_proj(x)

        # Reshape to [B, H, T, D]
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

        Q, K, V = reshape(Q), reshape(K), reshape(V)

        # Dispatch to Hopper attention kernel
        if _KERNELS_AVAILABLE:
            if self.attn_type == "encoder_self":
                attn_out = _kernels.encoder_self_attn(Q, K, V)
            elif self.attn_type == "decoder_cross":
                attn_out = _kernels.decoder_cross_attn(Q, K, V)
            else:
                attn_out = _kernels.decoder_self_attn(Q, K, V)
        else:
            # Standard scaled dot-product attention fallback
            scale = self.head_dim ** -0.5
            attn  = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
            attn_out = torch.matmul(attn, V)

        # Reshape back to [B, T, C]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class WhisperBlaze:
    """
    Whisper large-v3 with Hopper-native kernels.

    Loads weights from HuggingFace, patches attention and norm layers
    with Blaze-optimised versions according to PrecisionConfig.
    """

    MODEL_ID = "openai/whisper-large-v3"

    SAMPLE_RATE = 16_000
    CHUNK_SEC   = 30

    #: How long audio (>30 s) is split and stitched back together. Selectable
    #: per call — see the `mode` argument on transcribe()/transcribe_batch().
    #: Measured on 27 min of long-form audio (benchmarks/bench_overlap.py):
    #:
    #:   fast      string merge, 2s overlap WER 11.82%   418-432 RTFx  (default)
    #:   accurate  timestamps, no overlap   WER  6.83%   194-216 RTFx
    #:
    #: "accurate" decodes with return_timestamps=True and stitches on Whisper's
    #: own segment boundaries, which nearly halves long-form WER but costs ~2x
    #: in generation (the timestamp logits processor runs per decoding step).
    #: "fast" stitches by matching transcript strings across a 2 s overlap.
    #: Audio under 30 s is never chunked, so the mode does not affect it.
    MODES: Dict[str, Dict] = {
        "accurate": {"timestamps": True,  "stride_sec": 30},
        "fast":     {"timestamps": False, "stride_sec": 28},
    }
    #: Default is "fast" — maximum throughput. Pass mode="accurate" per call
    #: (or change this) when word error rate matters more than speed.
    DEFAULT_MODE = "fast"

    #: Legacy overrides. Left at None, the mode decides. Setting either pins
    #: that value for every call regardless of mode.
    STRIDE_SEC: Optional[int] = None
    USE_TIMESTAMP_MERGE: Optional[bool] = None

    @classmethod
    def _resolve_mode(cls, mode: Optional[str]) -> Tuple[bool, int]:
        """Return (use_timestamps, stride_seconds) for the requested mode."""
        name = mode or cls.DEFAULT_MODE
        if name not in cls.MODES:
            raise ValueError(
                f"unknown mode {name!r}; expected one of {sorted(cls.MODES)}")
        settings = cls.MODES[name]
        use_ts = (cls.USE_TIMESTAMP_MERGE if cls.USE_TIMESTAMP_MERGE is not None
                  else settings["timestamps"])
        stride = (cls.STRIDE_SEC if cls.STRIDE_SEC is not None
                  else settings["stride_sec"])
        return use_ts, stride

    def __init__(
        self,
        hf_model: nn.Module,
        processor,
        precision: PrecisionConfig,
        device: torch.device,
    ) -> None:
        self.model     = hf_model
        self.processor = processor
        self.precision = precision
        self.device    = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        precision: Optional[PrecisionConfig] = None,
        device: str = "cuda",
    ) -> "WhisperBlaze":
        if not _HF_AVAILABLE:
            raise ImportError("transformers required: pip install transformers")

        precision = precision or full_fp16()
        dev       = torch.device(device)

        print(f"Loading {model_id} ...", flush=True)
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(dev)

        hf_proc = AutoProcessor.from_pretrained(model_id)

        instance = cls(hf_model, hf_proc, precision, dev)
        instance._patch_layers()
        print("Model ready.", flush=True)
        print(precision.summary())
        return instance

    def _patch_layers(self) -> None:
        """Replace HF LayerNorm modules with the fused CUDA kernel.

        Whisper is pre-norm: LayerNorm and the residual add are separate ops,
        so each nn.LayerNorm becomes a plain (residual-free) kernel call.
        The kernel falls back to torch for any input it cannot handle, so this
        is safe for every shape and dtype the model produces.
        """
        self.patched_layernorms = 0
        if not _KERNELS_AVAILABLE:
            return

        for parent in self.model.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.LayerNorm) and BlazeLayerNorm.supports(child):
                    setattr(parent, name, BlazeLayerNorm.from_layernorm(child))
                    self.patched_layernorms += 1

    def transcribe(
        self,
        audio,
        language: Optional[str] = None,
        task: str = "transcribe",
        mode: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Transcribe audio of any length.

        Short audio (<=30s): single-pass transcription.
        Long audio (>30s):   batched 30s chunks in one GPU forward pass.

        Parameters
        ----------
        audio    : numpy array or torch tensor, float32, 16 kHz
        language : ISO 639-1 language code, or None for auto-detect
        task     : "transcribe" or "translate"
        mode     : "accurate" (default) or "fast" — see MODES. Only affects
                   audio longer than 30 s, which is the only case that chunks.

        Returns
        -------
        dict with keys: text, language
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)  # [channels, samples] → [samples]

        duration_s = len(audio) / self.SAMPLE_RATE

        if duration_s <= self.CHUNK_SEC:
            text = self._transcribe_short(audio, language, task)
        else:
            text = self._transcribe_long(audio, language, task, mode)

        return {"text": text.strip(), "language": language or "auto"}

    def _transcribe_short(
        self, audio: np.ndarray, language: Optional[str], task: str
    ) -> str:
        """Single-pass transcription for audio <= 30 seconds."""
        input_features = self.processor(
            audio, sampling_rate=self.SAMPLE_RATE, return_tensors="pt",
        ).input_features.to(self.device, dtype=torch.float16)

        generate_kwargs = {
            "task": task,
            "return_timestamps": True,
            "no_repeat_ngram_size": 3,
        }
        if language:
            generate_kwargs["language"] = language

        with torch.inference_mode():
            ids = self.model.generate(input_features, **generate_kwargs)

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def _transcribe_long(
        self, audio: np.ndarray, language: Optional[str], task: str,
        mode: Optional[str] = None,
    ) -> str:
        """Batched transcription for audio longer than 30 s.

        Delegates to transcribe_batch() so both entry points share one
        chunking and merging implementation.
        """
        return self.transcribe_batch(
            [audio], language=language, task=task, mode=mode)[0]["text"]

    def transcribe_batch(
        self,
        audio_list: List,
        language: Optional[str] = None,
        task: str = "transcribe",
        mode: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Transcribe multiple audio files in a single GPU forward pass.

        All chunks from all requests are stacked into one tensor and processed
        with a single model.generate() call, maximising H100 VRAM utilisation.

        Parameters
        ----------
        audio_list : list of numpy arrays or torch tensors, float32, 16 kHz
        language   : ISO 639-1 code for all items, or None for auto-detect
        task       : "transcribe" or "translate" — applied to all items
        mode       : "accurate" (default) or "fast" — see MODES. Resolved per
                     call, so concurrent callers can use different modes.

        Returns
        -------
        list of dicts, one per input audio, each with keys: text, language
        """
        # Normalise all inputs to numpy, collapsing [channels, samples] → [samples]
        audios: List[np.ndarray] = []
        for audio in audio_list:
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
            audios.append(audio)

        # Split each audio into 30s chunks, track per-request chunk counts
        all_chunks: List[np.ndarray] = []
        chunk_counts: List[int] = []

        sr        = self.SAMPLE_RATE
        chunk_len = self.CHUNK_SEC * sr
        use_timestamps, stride_sec = self._resolve_mode(mode)
        stride    = stride_sec * sr

        for audio in audios:
            if len(audio) / sr <= self.CHUNK_SEC:
                all_chunks.append(audio)
                chunk_counts.append(1)
            else:
                req_chunks: List[np.ndarray] = []
                for start in range(0, len(audio), stride):
                    end = min(start + chunk_len, len(audio))
                    req_chunks.append(audio[start:end])
                    if end >= len(audio):
                        break
                all_chunks.extend(req_chunks)
                chunk_counts.append(len(req_chunks))

        # Pad all chunks to the same length for batching. Never below a full
        # 30s chunk: Whisper's mel input must be 3000 frames, so a batch made
        # up entirely of short clips would otherwise fail in the encoder.
        max_len = max(chunk_len, max(len(c) for c in all_chunks))
        padded  = [np.pad(c, (0, max_len - len(c))) for c in all_chunks]

        # One generate() call for the entire combined batch
        input_features = self.processor(
            padded,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).input_features.to(self.device, dtype=torch.float16)

        generate_kwargs: Dict = {"task": task, "no_repeat_ngram_size": 3}
        if language:
            generate_kwargs["language"] = language
        # Timestamps let overlapping chunks be stitched by time rather than by
        # matching transcript strings; see _merge_segments.
        if use_timestamps:
            generate_kwargs["return_timestamps"] = True

        with torch.inference_mode():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)

        all_texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        all_ts = None
        if use_timestamps:
            # Decode one sequence at a time: the timestamp decoder walks tokens
            # as Python ints, which batch_decode does not guarantee.
            tok = self.processor.tokenizer
            all_ts = [tok.decode(list(seq), skip_special_tokens=False,
                                 decode_with_timestamps=True)
                      for seq in predicted_ids.tolist()]

        # Re-assemble per-request results
        results: List[Dict[str, str]] = []
        idx = 0
        for count in chunk_counts:
            texts = all_texts[idx : idx + count]
            text = ""
            if count == 1:
                text = texts[0].strip()
            else:
                if all_ts is not None:
                    segments: List[Tuple[float, float, str]] = []
                    for j in range(count):
                        offset = j * stride_sec
                        segments.extend(
                            self._parse_segments(all_ts[idx + j], offset))
                    text = self._merge_segments(segments)
                # Fall back to string matching if timestamps were unusable.
                if not text:
                    text = self._merge_chunks(texts)
            idx += count
            results.append({"text": text, "language": language or "auto"})

        return results

    #: Whisper emits segment boundaries as <|12.34|> tokens.
    _TS_RE = re.compile(r"<\|(\d+\.\d+)\|>")
    #: any other special token (<|endoftext|>, <|en|>, <|transcribe|>, ...);
    #: timestamps must be kept during decoding, so these survive with them.
    _SPECIAL_RE = re.compile(r"<\|[^|]*\|>")

    @classmethod
    def _parse_segments(cls, text_with_ts: str, offset_s: float
                        ) -> List[Tuple[float, float, str]]:
        """Parse '<|0.00|> hello<|2.50|>' into [(start, end, text)] in absolute
        seconds. Returns [] if the text carries no usable timestamps."""
        parts = cls._TS_RE.split(text_with_ts)
        if len(parts) < 3:
            return []

        segments: List[Tuple[float, float, str]] = []
        # parts alternates: [pre, ts, text, ts, text, ...]
        for i in range(1, len(parts) - 1, 2):
            try:
                start = float(parts[i])
            except ValueError:
                continue
            text = parts[i + 1]
            end = start
            if i + 2 < len(parts):
                try:
                    end = float(parts[i + 2])
                except ValueError:
                    pass
            text = cls._SPECIAL_RE.sub("", text).strip()
            if text:
                segments.append((offset_s + start, offset_s + end, text))
        return segments

    @staticmethod
    def _merge_segments(segments: List[Tuple[float, float, str]],
                        tolerance_s: float = 0.4) -> str:
        """Stitch segments from overlapping chunks using their timestamps.

        A segment whose audio was already covered by an accepted segment is a
        re-decode of the overlap region, so it is dropped. This replaces
        matching transcript strings, which fails whenever the two decodes of
        the same audio disagree by even one character.
        """
        if not segments:
            return ""
        segments = sorted(segments, key=lambda s: (s[0], s[1]))

        kept: List[str] = []
        covered_to = float("-inf")
        for start, end, text in segments:
            if start < covered_to - tolerance_s:
                continue                      # already-transcribed audio
            kept.append(text)
            covered_to = max(covered_to, end)
        return " ".join(kept).strip()

    @staticmethod
    def _merge_chunks(texts: List[str]) -> str:
        """Merge overlapping chunk transcriptions, removing duplicates.

        String-matching fallback, used when timestamps are unavailable.
        """
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0].strip()

        merged = [texts[0].strip()]
        for text in texts[1:]:
            text = text.strip()
            if not text:
                continue

            prev = merged[-1]
            # Find overlap: check if the end of previous text matches
            # the beginning of current text
            best_overlap = 0
            check_len = min(len(prev), len(text), 200)
            for k in range(20, check_len):
                if prev.endswith(text[:k]):
                    best_overlap = k

            if best_overlap > 20:
                merged.append(text[best_overlap:].strip())
            elif not prev.endswith(text[:20] if len(text) > 20 else text):
                merged.append(text)

        return " ".join(t for t in merged if t)
