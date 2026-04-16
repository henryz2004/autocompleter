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

_TERMINAL_OUTCOME_BY_EVENT = {
    "suggestion_accepted": "accepted",
    "partial_accept_used": "partial_accepted",
    "suggestion_dismissed": "dismissed",
    "typed_through": "typed_through",
    "invocation_superseded": "superseded",
    "invocation_no_suggestions": "no_suggestions",
    "invocation_errored": "error",
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
    event_time = _string_value(sanitized.get("timestamp")) or utcnow_iso()
    return {
        "event_id": str(uuid.uuid4()),
        "install_id": install_id,
        "event_name": sanitized.get("event", "unknown"),
        "invocation_id": _string_value(sanitized.get("invocation_id")),
        "request_id": _string_value(sanitized.get("request_id")),
        "event_time": event_time,
        "trigger_type": _string_value(sanitized.get("trigger_type")),
        "mode": _string_value(sanitized.get("mode")),
        "source_app": _string_value(sanitized.get("source_app")),
        "app_category": _string_value(sanitized.get("app_category")),
        "requested_route": _string_value(sanitized.get("requested_route")),
        "profile_json": _json_object(sanitized.get("profile")),
        "payload_json": sanitized,
        "received_at": utcnow_iso(),
    }


def build_invocation_row(
    *,
    install_id: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    sanitized = dict(sanitize_payload(payload))
    invocation_id = _string_value(sanitized.get("invocation_id"))
    if not invocation_id:
        return None

    event_name = _string_value(sanitized.get("event")) or "unknown"
    event_time = _string_value(sanitized.get("timestamp")) or utcnow_iso()
    row: dict[str, Any] = {
        "invocation_id": invocation_id,
        "install_id": install_id,
        "updated_at": event_time,
    }

    for field in ("trigger_type", "mode", "source_app", "app_category"):
        value = _string_value(sanitized.get(field))
        if value:
            row[field] = value
    requested_route = _string_value(sanitized.get("requested_route"))
    if requested_route:
        row["requested_route"] = requested_route
    request_id = _string_value(sanitized.get("request_id"))
    if request_id:
        row["proxy_request_id"] = request_id
    profile_json = _json_object(sanitized.get("profile"))
    if profile_json is not None:
        row["profile_json"] = profile_json

    first_display_latency_ms = _int_value(sanitized.get("first_display_latency_ms"))
    if first_display_latency_ms is not None:
        row["first_display_latency_ms"] = first_display_latency_ms
    stream_complete_latency_ms = _int_value(
        sanitized.get("stream_complete_latency_ms")
    )
    if stream_complete_latency_ms is not None:
        row["stream_complete_latency_ms"] = stream_complete_latency_ms
    dwell_ms = _int_value(sanitized.get("dwell_ms"))
    if dwell_ms is not None:
        row["dwell_ms"] = dwell_ms
    accepted_rank = _int_value(sanitized.get("suggestion_rank"))
    if accepted_rank is not None:
        row["accepted_rank"] = accepted_rank
    suggestion_count = _int_value(sanitized.get("count"))
    if suggestion_count is None:
        suggestion_count = _int_value(sanitized.get("count_shown"))
    if suggestion_count is not None:
        row["suggestion_count"] = suggestion_count
    fallback_used = _bool_value(sanitized.get("fallback_used"))
    if fallback_used is not None:
        row["fallback_used"] = fallback_used
    accepted_length_bucket = _string_value(sanitized.get("accepted_length_bucket"))
    if accepted_length_bucket:
        row["accepted_length_bucket"] = accepted_length_bucket
    error_type = _string_value(sanitized.get("error_type"))
    if error_type:
        row["error_type"] = error_type

    if event_name == "trigger_fired":
        row["started_at"] = event_time
    elif event_name == "first_suggestion_displayed":
        row["first_displayed_at"] = event_time
    elif event_name == "suggestions_returned":
        row["stream_completed_at"] = event_time

    terminal_outcome = _TERMINAL_OUTCOME_BY_EVENT.get(event_name)
    if terminal_outcome:
        row["final_outcome"] = terminal_outcome
        row["resolved_at"] = event_time

    if event_name == "suggestion_dismissed":
        row["dismiss_reason"] = "explicit"
    elif event_name == "typed_through":
        row["dismiss_reason"] = "typing"
    elif event_name == "invocation_superseded":
        row["dismiss_reason"] = "superseded"

    return row


def _string_value(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None
