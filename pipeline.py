"""
Pipeline orchestration — wires audio capture → VAD → ASR → translation → OBS.

Two threads:
  1. Audio-capture thread (daemon) pushes chunks via callback.
  2. Main thread processes VAD → ASR → translate → OBS update.

Threading is chosen over asyncio because PyAudioWPatch and faster-whisper
are synchronous blocking APIs — there is nothing to await.
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from asr import ASR
from audio import AudioCapture
from obs import OBSSubtitle
from translator import Translator
from vad import VAD

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class Pipeline:
    """Real-time OBS subtitle pipeline."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=200)

        # Initialise components — heaviest first (model loading).
        logger.info("Initialising ASR …")
        self._asr = ASR(cfg)
        logger.info("Initialising translator …")
        self._translator = Translator(cfg)
        logger.info("Initialising OBS connection …")
        self._obs = OBSSubtitle(cfg)
        logger.info("Initialising VAD …")
        self._vad = VAD(cfg)
        logger.info("Initialising audio capture …")
        self._audio = AudioCapture(cfg)

    def run(self) -> None:
        """Block forever, processing audio → subtitles.  Ctrl-C to stop."""
        self._install_signal_handlers()
        self._audio.start(self._on_audio_chunk)
        logger.info("Pipeline running — press Ctrl+C to stop")

        try:
            while True:
                try:
                    chunk = self._chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if chunk is None:
                    break  # shutdown sentinel
                self._vad.process_chunk(chunk, self._on_speech_segment)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    # ── callbacks ──────────────────────────────────────────────────

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Called from the audio-capture thread."""
        try:
            self._chunk_queue.put_nowait(chunk)
        except queue.Full:
            pass  # drop oldest if processing can't keep up

    def _on_speech_segment(self, segment: np.ndarray) -> None:
        """Called on the main thread when VAD emits a speech segment."""
        t0 = time.perf_counter()

        text = self._asr.transcribe(segment)
        if not text:
            return

        translation = self._translator.translate(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[%.0f ms] %s → %s",
            latency_ms,
            text,
            translation,
        )
        self._obs.update(translation)

    # ── lifecycle ──────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("Shutting down …")
        self._audio.stop()
        self._vad.flush(self._on_speech_segment)
        self._obs.clear()
        self._obs.close()
        logger.info("Done")

    def _install_signal_handlers(self) -> None:
        def _handler(sig, frame):
            self._chunk_queue.put(None)  # sentinel

        signal.signal(signal.SIGINT, _handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, _handler)
