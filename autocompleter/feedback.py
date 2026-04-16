"""Content-free feedback report helpers."""

from __future__ import annotations

import json
import logging
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import __version__

logger = logging.getLogger(__name__)


@dataclass
class FeedbackContext:
    """Safe metadata captured for a user-submitted report."""

    app_name: str | None = None
    app_bundle_role: str | None = None
    placeholder_detected: bool | None = None
    url_domain: str | None = None

    mode: str | None = None
    trigger_type: str | None = None
    extractor_name: str | None = None
    conversation_turns_detected: int | None = None
    conversation_speakers: int | None = None
    shell_detected: bool | None = None
    tui_detected: bool | None = None
    used_subtree_context: bool | None = None
    used_semantic_context: bool | None = None
    used_memory_context: bool | None = None
    visible_source: str | None = None

    llm_provider: str | None = None
    llm_model: str | None = None
    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_used: bool | None = None
    latency_ms: float | None = None
    first_suggestion_ms: float | None = None
    suggestion_count: int | None = None

    visible_text_elements_count: int | None = None
    subtree_context_chars: int | None = None

    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def extract_url_domain(url: str | None) -> str | None:
    """Return a safe URL identifier: host for HTTP(S), scheme otherwise."""
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
        return parsed.scheme
    return None


def build_payload(
    ctx: FeedbackContext,
    *,
    report_id: str | None = None,
    installation_id: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable feedback payload."""
    return {
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


class FeedbackReporter:
    """Write safe feedback payloads to disk for local inspection."""

    def __init__(self, *, feedback_dir: Path):
        self._dir = Path(feedback_dir)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning("Could not initialize feedback directory %s", self._dir, exc_info=True)

    def submit(
        self,
        ctx: FeedbackContext,
        *,
        installation_id: str | None = None,
    ) -> dict[str, Any]:
        payload = build_payload(ctx, installation_id=installation_id)
        self._write_local(payload)
        return payload

    def _write_local(self, payload: dict[str, Any]) -> None:
        report_id = str(payload.get("report_id", "unknown"))
        timestamp = int(payload.get("timestamp", int(time.time())))
        target = self._dir / f"{timestamp}-{report_id[:8]}.json"
        try:
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("Feedback report written: %s", target)
        except Exception:
            logger.warning("Failed to write feedback report to %s", target, exc_info=True)
