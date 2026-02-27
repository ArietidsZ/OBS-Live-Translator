"""
Speech-to-text using faster-whisper (CTranslate2 CUDA backend).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class ASR:
    """Wraps faster-whisper for single-shot transcription of speech segments."""

    def __init__(self, cfg: Config) -> None:
        logger.info(
            "Loading faster-whisper %s on %s (%s) …",
            cfg.asr_model,
            cfg.asr_device,
            cfg.asr_compute_type,
        )
        self._model = WhisperModel(
            cfg.asr_model,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
        )
        self._beam_size = cfg.asr_beam_size
        self._language = cfg.asr_language
        logger.info("ASR ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """Return recognised text for a 16 kHz float32 speech segment."""
        segments, info = self._model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=False,  # we already ran VAD upstream
        )
        text = " ".join(seg.text.strip() for seg in segments)
        if text:
            lang = self._language or info.language
            logger.debug("ASR [%s]: %s", lang, text)
        return text
