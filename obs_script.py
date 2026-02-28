"""
OBS Studio 脚本模式 — 直接作为 OBS 插件加载。

使用方法:
  1. OBS → 工具 → 脚本
  2. 添加此脚本 (obs_script.py)
  3. 配置参数（模型、语言、文本源名称）
  4. 点击"启动"

此脚本通过 obspython 直接操作 OBS 文本源，无需 WebSocket 连接。
后台线程处理 音频采集→VAD→ASR→翻译，主线程定时器更新字幕。

注意: OBS 使用内嵌 Python 解释器，需确保 Python 版本与 OBS 匹配。
      需在 OBS 的 Python 设置中配置正确的 Python 路径。
"""

from __future__ import annotations

import sys
import os
import threading
import queue
import logging

# Make our modules importable — the script dir is added to sys.path.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import obspython as obs  # type: ignore[import-not-found]
import numpy as np

from config import Config
from audio import AudioCapture
from vad import VAD
from asr import ASR
from translator import Translator

logger = logging.getLogger("obs-subtitle-plugin")

# ── Global state ───────────────────────────────────────────────────

_pipeline: _PluginPipeline | None = None

# Settings (populated by OBS properties UI)
_settings = {
    "source_name": "subtitle",
    "asr_model": "Qwen/Qwen3-ASR-0.6B",
    "asr_language": "",
    "target_lang": "zh",
    "translation_model": "tencent/HY-MT1.5-1.8B-GPTQ-Int4",
}

_UPDATE_INTERVAL_MS = 100  # poll for new subtitles every 100 ms


# ── OBS script callbacks ──────────────────────────────────────────


def script_description() -> str:
    return (
        "<h2>实时中文字幕</h2>"
        "<p>捕获系统音频 → 语音识别 → 翻译 → 字幕显示</p>"
        "<p>需要 NVIDIA GPU + CUDA</p>"
    )


def script_properties():
    props = obs.obs_properties_create()

    obs.obs_properties_add_text(
        props, "source_name", "字幕文本源名称", obs.OBS_TEXT_DEFAULT
    )
    obs.obs_properties_add_text(
        props,
        "asr_model",
        "ASR 模型 (例如 Qwen/Qwen3-ASR-0.6B)",
        obs.OBS_TEXT_DEFAULT,
    )
    obs.obs_properties_add_text(
        props, "asr_language", "源语言 (留空=自动检测)", obs.OBS_TEXT_DEFAULT
    )
    obs.obs_properties_add_text(props, "target_lang", "目标语言", obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_text(
        props, "translation_model", "翻译模型", obs.OBS_TEXT_DEFAULT
    )

    obs.obs_properties_add_button(props, "btn_start", "▶ 启动", _on_start_clicked)
    obs.obs_properties_add_button(props, "btn_stop", "■ 停止", _on_stop_clicked)

    return props


def script_defaults(settings):
    obs.obs_data_set_default_string(settings, "source_name", "subtitle")
    obs.obs_data_set_default_string(settings, "asr_model", "Qwen/Qwen3-ASR-0.6B")
    obs.obs_data_set_default_string(settings, "asr_language", "")
    obs.obs_data_set_default_string(settings, "target_lang", "zh")
    obs.obs_data_set_default_string(
        settings, "translation_model", "tencent/HY-MT1.5-1.8B-GPTQ-Int4"
    )


def script_update(settings):
    _settings["source_name"] = obs.obs_data_get_string(settings, "source_name")
    _settings["asr_model"] = obs.obs_data_get_string(settings, "asr_model")
    _settings["asr_language"] = obs.obs_data_get_string(settings, "asr_language")
    _settings["target_lang"] = obs.obs_data_get_string(settings, "target_lang")
    _settings["translation_model"] = obs.obs_data_get_string(
        settings, "translation_model"
    )


def script_unload():
    _stop_pipeline()


# ── Button handlers ────────────────────────────────────────────────


def _on_start_clicked(props, prop):
    global _pipeline
    if _pipeline is not None:
        return True  # already running

    cfg = Config(
        asr_model=_settings["asr_model"],
        asr_language=_settings["asr_language"] or None,
        translation_model=_settings["translation_model"],
        translation_target_lang=_settings["target_lang"],
    )

    _pipeline = _PluginPipeline(cfg, _settings["source_name"])
    _pipeline.start()
    obs.timer_add(_timer_tick, _UPDATE_INTERVAL_MS)
    logger.info("Pipeline started")
    return True


def _on_stop_clicked(props, prop):
    _stop_pipeline()
    return True


def _stop_pipeline():
    global _pipeline
    obs.timer_remove(_timer_tick)
    if _pipeline is not None:
        _pipeline.stop()
        _set_text(_pipeline.source_name, "")
        _pipeline = None
        logger.info("Pipeline stopped")


# ── Timer tick (main OBS thread) ───────────────────────────────────


def _timer_tick():
    if _pipeline is None:
        return
    text = _pipeline.poll()
    if text is not None:
        _set_text(_pipeline.source_name, text)


def _set_text(source_name: str, text: str) -> None:
    """Update an OBS Text (GDI+) source directly via obspython."""
    source = obs.obs_get_source_by_name(source_name)
    if source is None:
        return
    settings = obs.obs_data_create()
    obs.obs_data_set_string(settings, "text", text)
    obs.obs_source_update(source, settings)
    obs.obs_data_release(settings)
    obs.obs_source_release(source)


# ── Background pipeline ───────────────────────────────────────────


class _PluginPipeline:
    """Runs ASR + translation in a background thread; exposes results via poll()."""

    def __init__(self, cfg: Config, source_name: str) -> None:
        self.source_name = source_name
        self._result_queue: queue.Queue[str] = queue.Queue()
        self._audio = AudioCapture(cfg)
        self._vad = VAD(cfg)
        self._asr = ASR(cfg)
        self._translator = Translator(cfg)
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=200)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._audio.start(self._on_audio_chunk)
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True, name="plugin-process"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._audio.stop()
        self._chunk_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def poll(self) -> str | None:
        """Non-blocking: returns the latest subtitle or None."""
        result = None
        while not self._result_queue.empty():
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    # ── internals ──────────────────────────────────────────────────

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        try:
            self._chunk_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def _process_loop(self) -> None:
        while self._running:
            try:
                chunk = self._chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self._vad.process_chunk(chunk, self._on_speech)

    def _on_speech(self, segment: np.ndarray) -> None:
        text = self._asr.transcribe(segment)
        if not text:
            return
        translation = self._translator.translate(text)
        logger.info("%s → %s", text, translation)
        self._result_queue.put(translation)
