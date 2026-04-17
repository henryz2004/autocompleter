"""Debug artifact ingest models and sanitization."""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .store import utcnow_iso

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
_KEY_RE = re.compile(r"\b(?:sk|rk|pk|gsk|csk|tgp|fw)_[A-Za-z0-9._-]{10,}\b")
_INSTALL_KEY_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")


class DebugArtifactPayload(BaseModel):
    artifact_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


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
    return payload


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


def _looks_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(token in normalized for token in _SENSITIVE_KEYWORDS)


def _redact_string(value: str) -> str:
    redacted = _BEARER_RE.sub("Bearer [redacted]", value)
    redacted = _KEY_RE.sub("[redacted]", redacted)
    if len(value) > 24 and _INSTALL_KEY_RE.fullmatch(value):
        return "[redacted]"
    return redacted


def _nested_string(payload: dict[str, Any], *path: str) -> str | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _string_value(current)
