"""
Voice Activity Detection using Silero VAD.

Accumulates audio chunks and emits complete speech segments once silence
is detected after speech.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)

_SILERO_SAMPLE_RATE = 16_000  # Silero VAD requires 16 kHz input


class VAD:
    """Streaming VAD: feed chunks, get speech segment callbacks."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model.eval()

        self._speech_buf: list[np.ndarray] = []
        self._is_speaking = False
        self._speech_duration = 0.0

    def process_chunk(
        self,
        chunk: np.ndarray,
        on_speech: Callable[[np.ndarray], None],
    ) -> None:
        """Feed a 16 kHz float32 chunk.  Calls *on_speech* with the full
        speech segment when silence is detected after speech."""

        tensor = torch.from_numpy(chunk)
        prob = self._model(tensor, _SILERO_SAMPLE_RATE).item()
        chunk_dur = len(chunk) / _SILERO_SAMPLE_RATE

        if prob >= self._cfg.vad_threshold:
            # Speech detected.
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_duration = 0.0
                self._speech_buf.clear()
            self._speech_buf.append(chunk)
            self._speech_duration += chunk_dur

            # Force-emit if segment exceeds max length.
            if self._speech_duration >= self._cfg.vad_max_speech_s:
                self._emit(on_speech)
        else:
            # Silence.
            if self._is_speaking:
                self._speech_buf.append(chunk)  # include trailing pad
                self._emit(on_speech)

    def flush(self, on_speech: Callable[[np.ndarray], None]) -> None:
        """Emit any remaining buffered speech."""
        if self._is_speaking and self._speech_buf:
            self._emit(on_speech)

    def _emit(self, on_speech: Callable[[np.ndarray], None]) -> None:
        if self._speech_duration < self._cfg.vad_min_speech_s:
            logger.debug("Dropping short segment (%.2f s)", self._speech_duration)
        else:
            segment = np.concatenate(self._speech_buf)
            logger.debug("Speech segment: %.2f s", len(segment) / _SILERO_SAMPLE_RATE)
            on_speech(segment)
        self._speech_buf.clear()
        self._is_speaking = False
        self._speech_duration = 0.0
