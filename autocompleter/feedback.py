"""User-submitted feedback/bug reports.

Users hit a hotkey to ping a report about the currently-focused app. The
payload is deliberately content-free: we capture app metadata, LLM config,
and latency — never cursor text, visible text, window titles, or suggestion
text. The report is written to ``~/.autocompleter/feedback/<ts>.json`` and,
if configured, POSTed to a webhook URL.
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import __version__

logger = logging.getLogger(__name__)


@dataclass
class FeedbackContext:
    """The snapshot of current state used to build a feedback report.

    All fields are optional because the user may hit the bug-report hotkey
    before any trigger has happened.
    """

    # --- App identity (safe) ---
    app_name: str | None = None
    app_bundle_role: str | None = None  # focused element AX role, e.g. AXTextArea
    placeholder_detected: bool | None = None
    url_domain: str | None = None  # host only, no path/query

    # --- Mode + extractor ---
    mode: str | None = None  # "continuation" | "reply"
    trigger_type: str | None = None  # "manual" | "auto" | "regenerate" | "post_accept"
    extractor_name: str | None = None
    conversation_turns_detected: int | None = None
    conversation_speakers: int | None = None
    shell_detected: bool | None = None
    tui_detected: bool | None = None
    used_subtree_context: bool | None = None
    used_semantic_context: bool | None = None
    used_memory_context: bool | None = None
    visible_source: str | None = None  # "fresh"|"cache"|"worker_refresh"

    # --- LLM config + latency ---
    llm_provider: str | None = None
    llm_model: str | None = None
    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_used: bool | None = None
    latency_ms: float | None = None  # end-to-end
    first_suggestion_ms: float | None = None
    suggestion_count: int | None = None

    # --- Counts (no content) ---
    visible_text_elements_count: int | None = None
    subtree_context_chars: int | None = None

    # --- Free-form notes (always supplied by caller, never by user) ---
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_DOMAIN_SAFE = "domain_only"


def extract_url_domain(url: str | None) -> str | None:
    """Return just the host of *url* (e.g. ``github.com``). Strips path/query.

    Returns ``None`` for empty / unparseable input. ``file://`` and other
    non-http schemes are reduced to the scheme (``file``) so we don't leak
    local paths.
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme in ("http", "https"):
        host = parsed.hostname or ""
        return host.lower() or None
    if parsed.scheme:
        # Don't leak file paths or custom-scheme payloads
        return parsed.scheme
    return None


def build_payload(
    ctx: FeedbackContext,
    *,
    report_id: str | None = None,
    installation_id: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable report payload from *ctx*.

    Only content-free metadata is included. This function is the single
    source of truth for the shape of a feedback ping.
    """
    payload: dict[str, Any] = {
        "schema": "autocompleter.feedback.v1",
        "report_id": report_id or uuid.uuid4().hex,
        "installation_id": installation_id,
        "timestamp": int(time.time()),
        "version": __version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "app": {
            "name": ctx.app_name,
            "role": ctx.app_bundle_role,
            "placeholder_detected": ctx.placeholder_detected,
            "url_domain": ctx.url_domain,
        },
        "context_pipeline": {
            "mode": ctx.mode,
            "trigger_type": ctx.trigger_type,
            "extractor": ctx.extractor_name,
            "conversation_turns_detected": ctx.conversation_turns_detected,
            "conversation_speakers": ctx.conversation_speakers,
            "shell_detected": ctx.shell_detected,
            "tui_detected": ctx.tui_detected,
            "visible_source": ctx.visible_source,
            "visible_text_elements_count": ctx.visible_text_elements_count,
            "subtree_context_chars": ctx.subtree_context_chars,
            "used_subtree_context": ctx.used_subtree_context,
            "used_semantic_context": ctx.used_semantic_context,
            "used_memory_context": ctx.used_memory_context,
        },
        "llm": {
            "provider": ctx.llm_provider,
            "model": ctx.llm_model,
            "fallback_provider": ctx.fallback_provider,
            "fallback_model": ctx.fallback_model,
            "fallback_used": ctx.fallback_used,
            "latency_ms": ctx.latency_ms,
            "first_suggestion_ms": ctx.first_suggestion_ms,
            "suggestion_count": ctx.suggestion_count,
        },
        "note": ctx.note,
        "extra": dict(ctx.extra) if ctx.extra else {},
    }
    return payload


class FeedbackReporter:
    """Persists feedback reports locally and, when configured, POSTs to a webhook.

    The reporter is safe to call from any thread. Network I/O runs on a
    daemon thread so the event-tap thread (which must return within ~1s on
    macOS) never blocks on a slow webhook.
    """

    _INSTALLATION_FILE = "installation_id"

    def __init__(
        self,
        *,
        feedback_dir: Path,
        webhook_url: str | None = None,
        http_timeout_s: float = 5.0,
    ):
        self._dir = Path(feedback_dir)
        self._webhook = webhook_url or None
        self._timeout = http_timeout_s
        self._installation_id: str | None = None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._installation_id = self._load_or_create_installation_id()
        except Exception:
            logger.warning(
                "Could not initialize feedback directory %s", self._dir,
                exc_info=True,
            )

    @property
    def installation_id(self) -> str | None:
        return self._installation_id

    def _load_or_create_installation_id(self) -> str:
        """Create a stable-per-machine random ID so duplicate reports can be merged."""
        id_path = self._dir / self._INSTALLATION_FILE
        if id_path.exists():
            try:
                value = id_path.read_text(encoding="utf-8").strip()
                if value:
                    return value
            except Exception:
                logger.debug("Could not read installation_id", exc_info=True)
        new_id = uuid.uuid4().hex
        try:
            id_path.write_text(new_id, encoding="utf-8")
        except Exception:
            logger.debug("Could not write installation_id", exc_info=True)
        return new_id

    def submit(self, ctx: FeedbackContext) -> dict[str, Any]:
        """Persist a report and (optionally) POST it. Returns the payload sent.

        The webhook POST is dispatched to a daemon thread so callers on the
        event-tap thread never block on the network.
        """
        payload = build_payload(
            ctx,
            installation_id=self._installation_id,
        )
        self._write_local(payload)
        if self._webhook:
            t = threading.Thread(
                target=self._post_webhook,
                args=(payload,),
                daemon=True,
            )
            t.start()
        else:
            logger.debug(
                "Feedback saved locally (no webhook configured): %s",
                payload["report_id"],
            )
        return payload

    def _write_local(self, payload: dict[str, Any]) -> None:
        report_id = payload.get("report_id", "unknown")
        ts = payload.get("timestamp", int(time.time()))
        fname = f"{ts}-{report_id[:8]}.json"
        target = self._dir / fname
        try:
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("Feedback report written: %s", target)
        except Exception:
            logger.warning(
                "Failed to write feedback report to %s", target, exc_info=True,
            )

    def _post_webhook(self, payload: dict[str, Any]) -> None:
        url = self._webhook
        if not url:
            return
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"autocompleter/{__version__}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                logger.info(
                    "Feedback webhook POST to %s -> HTTP %s",
                    _redact_url(url),
                    status,
                )
        except Exception:
            logger.warning(
                "Feedback webhook POST failed (%s)",
                _redact_url(url),
                exc_info=True,
            )


def _redact_url(url: str) -> str:
    """Return ``scheme://host`` for logging so tokens in the path aren't leaked."""
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.hostname:
            return f"{parsed.scheme}://{parsed.hostname}"
    except Exception:
        pass
    return "<redacted>"
