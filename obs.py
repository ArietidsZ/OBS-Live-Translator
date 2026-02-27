"""
OBS WebSocket integration — pushes subtitle text to a Text (GDI+) source.

Uses the OBS WebSocket v5 protocol (built into OBS 28+).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import obsws_python as obsws

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class OBSSubtitle:
    """Manages the connection to OBS and updates a text source."""

    def __init__(self, cfg: Config) -> None:
        self._source_name = cfg.obs_source_name
        self._client: obsws.ReqClient | None = None
        self._cfg = cfg
        self._connect()

    def _connect(self) -> None:
        try:
            self._client = obsws.ReqClient(
                host=self._cfg.obs_host,
                port=self._cfg.obs_port,
                password=self._cfg.obs_password or None,
                timeout=5,
            )
            logger.info(
                "Connected to OBS WebSocket at %s:%d",
                self._cfg.obs_host,
                self._cfg.obs_port,
            )
        except Exception as exc:
            logger.error("Cannot connect to OBS: %s", exc)
            self._client = None

    def update(self, text: str) -> None:
        """Set the subtitle text.  Reconnects automatically on failure."""
        if self._client is None:
            self._connect()
        if self._client is None:
            return  # still not connected

        try:
            self._client.set_input_settings(
                name=self._source_name,
                settings={"text": text},
                overlay=True,
            )
        except Exception as exc:
            logger.warning("OBS update failed (%s), reconnecting …", exc)
            self._client = None
            self._connect()

    def clear(self) -> None:
        """Clear the subtitle."""
        self.update("")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
