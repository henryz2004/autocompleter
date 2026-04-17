"""Remote debug capture helpers for beta installs."""

from __future__ import annotations

import copy
import json
import logging
import platform
import queue
import re
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any
from urllib import request

DEBUG_CAPTURE_OFF = "off"
DEBUG_CAPTURE_FAILURES = "failures"
DEBUG_CAPTURE_MANUAL = "manual"
DEBUG_CAPTURE_BOTH = "both"
VALID_DEBUG_CAPTURE_MODES = {
    DEBUG_CAPTURE_OFF,
    DEBUG_CAPTURE_FAILURES,
    DEBUG_CAPTURE_MANUAL,
    DEBUG_CAPTURE_BOTH,
}

_DEFAULT_MAX_LOG_RECORDS = 400
_DEFAULT_MAX_LOG_LINE_CHARS = 500
_DEFAULT_LOG_TAIL_LINES = 200
_MAX_DEBUG_PAYLOAD_CHARS = 250_000
_SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer",
    "cookie",
    "password",
    "secret",
    "session_token",
    "token",
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_RE = re.compile(
    r"\b(?:sk|rk|pk|gsk|csk|tgp|fw)_[A-Za-z0-9._-]{10,}\b"
)
_INSTALL_KEY_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_debug_capture_mode(raw: str | None) -> str:
    value = (raw or DEBUG_CAPTURE_OFF).strip().lower()
    if value in VALID_DEBUG_CAPTURE_MODES:
        return value
    return DEBUG_CAPTURE_OFF


def debug_capture_mode_allows_failures(mode: str) -> bool:
    return normalize_debug_capture_mode(mode) in {
        DEBUG_CAPTURE_FAILURES,
        DEBUG_CAPTURE_BOTH,
    }


def debug_capture_mode_allows_manual(mode: str) -> bool:
    return normalize_debug_capture_mode(mode) in {
        DEBUG_CAPTURE_MANUAL,
        DEBUG_CAPTURE_BOTH,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    return f"<nonserializable:{type(value).__name__}>"


def _looks_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(token in normalized for token in _SENSITIVE_KEYWORDS)


def _redact_string(value: str) -> str:
    redacted = _BEARER_RE.sub("Bearer [redacted]", value)
    redacted = _KEY_RE.sub("[redacted]", redacted)
    if len(value) > 24 and _INSTALL_KEY_RE.fullmatch(value):
        return "[redacted]"
    return redacted


def redact_debug_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if _looks_sensitive_key(str(key)):
                cleaned[str(key)] = "[redacted]"
            else:
                cleaned[str(key)] = redact_debug_payload(value)
        return cleaned
    if isinstance(payload, list):
        return [redact_debug_payload(item) for item in payload]
    if isinstance(payload, str):
        return _redact_string(payload)
    return _json_safe(payload)


def _json_chars(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _truncate_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 15] + "... [truncated]"


def _prune_tree(tree: Any, *, max_depth: int) -> Any:
    if not isinstance(tree, dict):
        return tree

    def _walk(node: dict[str, Any], depth: int) -> dict[str, Any]:
        pruned = {
            key: _json_safe(value)
            for key, value in node.items()
            if key != "children"
        }
        children = node.get("children") or []
        if depth >= max_depth:
            if children:
                pruned["children"] = [
                    {"role": "...", "description": f"{len(children)} children omitted"}
                ]
            else:
                pruned["children"] = []
            return pruned
        pruned["children"] = [
            _walk(child, depth + 1)
            for child in children[:20]
            if isinstance(child, dict)
        ]
        if len(children) > 20:
            pruned["children"].append(
                {"role": "...", "description": f"{len(children) - 20} children omitted"}
            )
        return pruned

    return _walk(tree, 0)


def trim_debug_artifact(
    artifact: dict[str, Any],
    *,
    max_chars: int = _MAX_DEBUG_PAYLOAD_CHARS,
) -> dict[str, Any]:
    payload = copy.deepcopy(_json_safe(artifact))
    meta = payload.setdefault("meta", {})
    meta.setdefault("trimmed", False)

    if _json_chars(payload) <= max_chars:
        return payload

    meta["trimmed"] = True

    log_tail = payload.get("log_tail")
    if isinstance(log_tail, list):
        payload["log_tail"] = log_tail[-100:]
    if _json_chars(payload) <= max_chars:
        return payload

    focus_debug = payload.get("focus_debug")
    if isinstance(focus_debug, dict):
        for key in ("window_tree", "app_local_tree"):
            if key in focus_debug:
                focus_debug[key] = _prune_tree(focus_debug[key], max_depth=8)
    if _json_chars(payload) <= max_chars:
        return payload

    trigger_dump = payload.get("trigger_dump")
    if isinstance(trigger_dump, dict):
        for key in ("context",):
            if key in trigger_dump:
                trigger_dump[key] = _truncate_text(trigger_dump[key], 20_000)
        focused = trigger_dump.get("focused")
        if isinstance(focused, dict):
            for key in ("beforeCursor", "afterCursor", "rawValue"):
                if key in focused:
                    focused[key] = _truncate_text(focused[key], 2_000)
        if "conversationTurns" in trigger_dump:
            trigger_dump["conversationTurns"] = (trigger_dump["conversationTurns"] or [])[:6]
        if "tree" in trigger_dump:
            trigger_dump["tree"] = _prune_tree(trigger_dump["tree"], max_depth=8)
    if _json_chars(payload) <= max_chars:
        return payload

    if isinstance(log_tail, list):
        payload["log_tail"] = payload.get("log_tail", [])[-25:]
    if _json_chars(payload) <= max_chars:
        return payload

    if isinstance(trigger_dump, dict):
        trigger_dump["context"] = "[truncated]"
        trigger_dump["conversationTurns"] = []
    focus_debug = payload.get("focus_debug")
    if isinstance(focus_debug, dict):
        focus_debug.pop("window_tree", None)
        focus_debug.pop("app_local_tree", None)
    return payload


class InMemoryLogBuffer(logging.Handler):
    """Rolling in-memory log buffer for debug uploads."""

    def __init__(
        self,
        *,
        max_records: int = _DEFAULT_MAX_LOG_RECORDS,
        max_line_chars: int = _DEFAULT_MAX_LOG_LINE_CHARS,
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self._records: deque[str] = deque(maxlen=max_records)
        self._max_line_chars = max_line_chars
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        if len(message) > self._max_line_chars:
            message = message[: self._max_line_chars - 15] + "... [truncated]"
        with self._lock:
            self._records.append(message)

    def snapshot(self, *, limit: int = _DEFAULT_LOG_TAIL_LINES) -> list[str]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._records)[-limit:]


class DebugArtifactClient:
    """Authenticated background uploader for rich debug artifacts."""

    _QUEUE_STOP = object()

    def __init__(
        self,
        *,
        enabled: bool,
        url: str,
        api_key: str,
        install_id: str,
        app_version: str,
        capture_mode: str,
    ) -> None:
        self.capture_mode = normalize_debug_capture_mode(capture_mode)
        self.enabled = (
            enabled
            and self.capture_mode != DEBUG_CAPTURE_OFF
            and bool(url.strip())
            and bool(api_key.strip())
            and bool(install_id.strip())
        )
        self.url = url.strip()
        self.api_key = api_key.strip()
        self.install_id = install_id.strip()
        self.app_version = app_version
        self.os_version = platform.mac_ver()[0] or platform.platform()
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._stopped = False
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    @property
    def failure_capture_enabled(self) -> bool:
        return self.enabled and debug_capture_mode_allows_failures(self.capture_mode)

    @property
    def manual_capture_enabled(self) -> bool:
        return self.enabled and debug_capture_mode_allows_manual(self.capture_mode)

    def emit_artifact(
        self,
        artifact_type: str,
        payload: dict[str, Any],
        *,
        invocation_id: str | None = None,
        source_app: str | None = None,
        trigger_type: str | None = None,
    ) -> None:
        if not self.enabled or self._stopped:
            return
        record = {
            "artifact_type": artifact_type,
            "invocation_id": invocation_id,
            "source_app": source_app,
            "trigger_type": trigger_type,
            "payload": trim_debug_artifact(redact_debug_payload(payload)),
            "app_version": self.app_version,
            "os_version": self.os_version,
            "timestamp": utcnow_iso(),
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            logging.getLogger(__name__).warning(
                "Dropping debug artifact because the queue is full"
            )

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

        try:
            self._queue.put_nowait({"event": "__flush__", "callback": _mark_done})
        except queue.Full:
            return
        done.wait(timeout)

    def _worker(self) -> None:
        logger = logging.getLogger(__name__)
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
                logger.warning("Debug artifact delivery failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _send_payload(self, payload: dict[str, Any]) -> None:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"autocompleter/{self.app_version}",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(self.url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=5.0) as resp:
            if getattr(resp, "status", 200) >= 400:
                raise RuntimeError(f"Debug artifact HTTP {resp.status}")
