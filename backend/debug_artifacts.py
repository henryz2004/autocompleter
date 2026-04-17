"""Debug artifact ingest models and sanitization."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autocompleter.debug_capture import redact_debug_payload

from .store import utcnow_iso


class DebugArtifactPayload(BaseModel):
    artifact_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


def build_debug_artifact_row(
    *,
    install_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    sanitized = dict(redact_debug_payload(payload))
    sanitized["install_id"] = install_id
    captured_at = (
        _string_value(sanitized.get("timestamp"))
        or _nested_string(sanitized, "payload", "meta", "captured_at")
        or utcnow_iso()
    )
    return {
        "artifact_id": str(uuid.uuid4()),
        "install_id": install_id,
        "invocation_id": (
            _string_value(sanitized.get("invocation_id"))
            or _nested_string(sanitized, "payload", "meta", "invocation_id")
        ),
        "artifact_type": (
            _string_value(sanitized.get("artifact_type"))
            or _nested_string(sanitized, "payload", "meta", "artifact_type")
            or "unknown"
        ),
        "source_app": (
            _string_value(sanitized.get("source_app"))
            or _nested_string(sanitized, "payload", "meta", "source_app")
            or _nested_string(sanitized, "payload", "focus_debug", "frontmost_app", "name")
        ),
        "trigger_type": (
            _string_value(sanitized.get("trigger_type"))
            or _nested_string(sanitized, "payload", "meta", "trigger_type")
        ),
        "created_at": captured_at,
        "payload_json": sanitized,
    }


def _string_value(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _nested_string(payload: dict[str, Any], *path: str) -> str | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _string_value(current)
