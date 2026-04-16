"""Minimal beta telemetry helpers."""

from __future__ import annotations

import json
import logging
import platform
import queue
import threading
from datetime import datetime, timezone
from urllib import request

from .suggestion_engine import is_shell_app

logger = logging.getLogger(__name__)

INVOCATION_HEADER = "X-Autocompleter-Invocation-Id"
REQUEST_HEADER = "X-Autocompleter-Request-Id"

_BROWSER_APPS = {
    "Arc",
    "Brave Browser",
    "Chromium",
    "Firefox",
    "Google Chrome",
    "Microsoft Edge",
    "Safari",
}

_CHAT_APPS = {
    "ChatGPT",
    "Claude",
    "Claude Desktop",
    "Discord",
    "Google Gemini",
    "Messages",
    "Slack",
    "WhatsApp",
}

_EDITOR_APPS = {
    "BBEdit",
    "Code",
    "Codex",
    "Cursor",
    "Nova",
    "Sublime Text",
    "TextEdit",
    "Visual Studio Code",
    "Xcode",
}


def bucket_latency_ms(latency_ms: float | int | None) -> str:
    if latency_ms is None:
        return "unknown"
    value = float(latency_ms)
    if value < 250:
        return "<250"
    if value < 500:
        return "250-500"
    if value < 1000:
        return "500-1000"
    if value < 2000:
        return "1000-2000"
    return "2000+"


def bucket_length(length: int | None) -> str:
    if length is None or length <= 0:
        return "0"
    if length <= 10:
        return "1-10"
    if length <= 30:
        return "11-30"
    if length <= 80:
        return "31-80"
    return "81+"


def categorize_app(app_name: str) -> str:
    if is_shell_app(app_name):
        return "terminal"
    if app_name in _BROWSER_APPS:
        return "browser"
    if app_name in _CHAT_APPS:
        return "chat"
    if app_name in _EDITOR_APPS:
        return "editor"
    return "other"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TelemetryClient:
    """Small fire-and-forget telemetry client for friend beta builds."""

    _QUEUE_STOP = object()

    def __init__(
        self,
        *,
        enabled: bool,
        url: str,
        install_id: str,
        beta_mode: bool,
        app_version: str,
        api_key: str = "",
    ) -> None:
        self.enabled = enabled and bool(url.strip())
        self.url = url.strip()
        self.install_id = install_id
        self.beta_mode = bool(beta_mode)
        self.app_version = app_version
        self.api_key = api_key.strip()
        self.os_version = platform.mac_ver()[0] or platform.platform()
        self._queue: queue.Queue[dict | object] = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stopped = False
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        elif enabled and not self.url:
            logger.warning("Telemetry enabled but AUTOCOMPLETER_TELEMETRY_URL is empty")

    def emit(self, event: str, **payload: object) -> None:
        if not self.enabled or self._stopped:
            return
        record = {
            "event": event,
            "install_id": self.install_id,
            "app_version": self.app_version,
            "os_version": self.os_version,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "beta_mode": self.beta_mode,
        }
        record.update(payload)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            logger.warning("Dropping telemetry event because the queue is full")

    def stop(self) -> None:
        if not self.enabled or self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(self._QUEUE_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def flush(self, timeout: float = 1.0) -> None:
        if not self.enabled:
            return
        done = threading.Event()

        def _mark_done() -> None:
            done.set()

        self.emit("__flush__", callback=_mark_done)
        done.wait(timeout)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._QUEUE_STOP:
                    return
                payload = dict(item)
                callback = payload.pop("callback", None)
                if payload.get("event") != "__flush__":
                    self._send_payload(payload)
                if callable(callback):
                    callback()
            except Exception:
                logger.warning("Telemetry delivery failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _send_payload(self, payload: dict[str, object]) -> None:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"autocompleter/{self.app_version}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(self.url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=2.0) as resp:
            if getattr(resp, "status", 200) >= 400:
                raise RuntimeError(f"Telemetry HTTP {resp.status}")
