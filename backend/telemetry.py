"""Telemetry ingest models and sanitization."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .store import utcnow_iso

_BLOCKED_KEYS = {
    "accepted_text",
    "after_cursor",
    "before_cursor",
    "completion",
    "conversation",
    "full_url",
    "message_text",
    "messages",
    "prompt",
    "prompt_text",
    "raw_completion",
    "raw_prompt",
    "subtree_xml",
    "suggestion_text",
    "terminal_text",
    "url",
    "window_title",
}


class TelemetryEventPayload(BaseModel):
    event: str = Field(min_length=1)
    model_config = ConfigDict(extra="allow")


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if key in _BLOCKED_KEYS:
                continue
            cleaned[key] = sanitize_payload(value)
        return cleaned
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    return payload


def build_telemetry_row(
    *,
    install_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    sanitized = dict(sanitize_payload(payload))
    sanitized["install_id"] = install_id
    return {
        "event_id": str(uuid.uuid4()),
        "install_id": install_id,
        "event_name": sanitized.get("event", "unknown"),
        "payload_json": sanitized,
        "received_at": utcnow_iso(),
    }
