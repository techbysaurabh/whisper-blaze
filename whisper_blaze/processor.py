"""
processor.py
GPU-native audio preprocessor for Whisper large-v3.

Replaces HuggingFace AutoProcessor's CPU-based mel spectrogram with
the custom CUDA kernel from mel_spectrogram.cu.

Handles:
  - Resampling to 16 kHz  (torchaudio)
  - Stereo → mono
  - Padding / trimming to 30-second chunks
  - GPU mel spectrogram via whisper_blaze_kernels.mel_spectrogram
"""

from __future__ import annotations

import math
from typing import List, Union

import torch
import numpy as np

try:
    import torchaudio
    import torchaudio.functional as F_audio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import whisper_blaze_kernels as _kernels
    _MEL_AVAILABLE = True
except ImportError:
    _MEL_AVAILABLE = False
    import warnings
    warnings.warn(
        "whisper_blaze_kernels not compiled. "
        "Falling back to librosa for mel spectrogram.",
        stacklevel=2,
    )


SAMPLE_RATE   = 16_000
N_SAMPLES_30S = 30 * SAMPLE_RATE   # 480 000
N_MELS        = 128
N_FRAMES      = 3000               # mel frames for 30s audio


class WhisperBlazeProcessor:
    """
    Preprocesses raw audio into a mel spectrogram tensor for Whisper large-v3.

    Parameters
    ----------
    device : str or torch.device
        Target device. Should be 'cuda' for H100 kernel path.
    chunk_length_s : float
        Maximum audio length per forward pass (seconds). Default 30.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        chunk_length_s: float = 30.0,
    ) -> None:
        self.device        = torch.device(device)
        self.chunk_samples = int(chunk_length_s * SAMPLE_RATE)
        self._use_gpu_mel  = _MEL_AVAILABLE and self.device.type == "cuda"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor, List],
        sampling_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio        : np.ndarray or torch.Tensor, shape (samples,) or (channels, samples)
        sampling_rate: original sampling rate of audio

        Returns
        -------
        torch.Tensor : shape (1, N_MELS, N_FRAMES) fp16 on self.device
        """
        audio = self._to_mono_float32(audio)
        audio = self._resample(audio, sampling_rate)
        audio = self._pad_or_trim(audio)

        return self._mel_spectrogram(audio)

    def process_chunks(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sampling_rate: int = SAMPLE_RATE,
        overlap_s: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        Split long audio into overlapping chunks and return a list of
        mel spectrograms, one per chunk.
        """
        audio  = self._to_mono_float32(audio)
        audio  = self._resample(audio, sampling_rate)

        overlap_samples = int(overlap_s * SAMPLE_RATE)
        step            = self.chunk_samples - overlap_samples
        chunks          = []

        for start in range(0, len(audio), step):
            chunk = audio[start : start + self.chunk_samples]
            chunk = self._pad_or_trim(chunk)
            chunks.append(self._mel_spectrogram(chunk))

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_mono_float32(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.float()
        if audio.dim() == 2:          # (channels, samples)
            audio = audio.mean(dim=0)
        return audio.contiguous()

    def _resample(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == SAMPLE_RATE:
            return audio
        if not HAS_TORCHAUDIO:
            raise RuntimeError(
                f"torchaudio required for resampling from {orig_sr} to {SAMPLE_RATE}. "
                "Install with: pip install torchaudio"
            )
        return F_audio.resample(audio, orig_sr, SAMPLE_RATE)

    def _pad_or_trim(self, audio: torch.Tensor) -> torch.Tensor:
        n = len(audio)
        if n > self.chunk_samples:
            return audio[: self.chunk_samples]
        if n < self.chunk_samples:
            return torch.nn.functional.pad(audio, (0, self.chunk_samples - n))
        return audio

    def _mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Run mel spectrogram — GPU kernel or librosa fallback."""
        if self._use_gpu_mel:
            # GPU kernel expects CPU float32 input (uploads internally)
            return _kernels.mel_spectrogram(audio.cpu())
        else:
            return self._librosa_mel(audio)

    @staticmethod
    def _librosa_mel(audio: torch.Tensor) -> torch.Tensor:
        """CPU fallback using librosa (for environments without compiled kernel)."""
        try:
            import librosa
        except ImportError as e:
            raise ImportError("librosa required for CPU mel fallback.") from e

        arr = audio.numpy()
        mel = librosa.feature.melspectrogram(
            y=arr, sr=SAMPLE_RATE,
            n_fft=400, hop_length=160,
            n_mels=128, fmin=0, fmax=8000,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel + 80.0) / 80.0  # normalise to ~[-1, 1]
        t = torch.from_numpy(log_mel).unsqueeze(0).half()  # [1, 128, T]
        return t
