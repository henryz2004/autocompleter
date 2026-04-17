"""Tests for remote debug capture helpers."""

from __future__ import annotations

import json
import logging
import threading

from autocompleter.debug_capture import (
    DEBUG_CAPTURE_BOTH,
    DEBUG_CAPTURE_FAILURES,
    DEBUG_CAPTURE_MANUAL,
    DEBUG_CAPTURE_PROFILE_AGGRESSIVE,
    DEBUG_CAPTURE_PROFILE_NORMAL,
    DebugArtifactClient,
    InMemoryLogBuffer,
    normalize_debug_capture_profile,
    normalize_debug_capture_mode,
    redact_debug_payload,
    trim_debug_artifact,
)


def test_normalize_debug_capture_mode():
    assert normalize_debug_capture_mode(None) == "off"
    assert normalize_debug_capture_mode("BOTH") == DEBUG_CAPTURE_BOTH
    assert normalize_debug_capture_mode("failures") == DEBUG_CAPTURE_FAILURES
    assert normalize_debug_capture_mode("manual") == DEBUG_CAPTURE_MANUAL
    assert normalize_debug_capture_mode("nope") == "off"


def test_normalize_debug_capture_profile():
    assert normalize_debug_capture_profile(None) == DEBUG_CAPTURE_PROFILE_NORMAL
    assert normalize_debug_capture_profile("AGGRESSIVE") == DEBUG_CAPTURE_PROFILE_AGGRESSIVE
    assert normalize_debug_capture_profile("loud") == DEBUG_CAPTURE_PROFILE_NORMAL


def test_in_memory_log_buffer_keeps_recent_lines():
    handler = InMemoryLogBuffer(max_records=3, max_line_chars=80)
    handler.setFormatter(logging.Formatter("%(message)s"))

    for index in range(5):
        record = logging.makeLogRecord({"msg": f"line-{index}", "levelno": logging.INFO, "levelname": "INFO"})
        handler.emit(record)

    assert handler.snapshot(limit=10) == ["line-2", "line-3", "line-4"]


def test_redact_debug_payload_scrubs_secret_like_values():
    payload = {
        "headers": {"Authorization": "Bearer super-secret-token"},
        "api_key": "sk_live_abcdef1234567890",
        "nested": {"cookie": "session=abcd", "note": "keep me"},
    }

    redacted = redact_debug_payload(payload)

    assert redacted["headers"]["Authorization"] == "[redacted]"
    assert redacted["api_key"] == "[redacted]"
    assert redacted["nested"]["cookie"] == "[redacted]"
    assert redacted["nested"]["note"] == "keep me"


def test_trim_debug_artifact_prunes_large_payload_deterministically():
    artifact = {
        "meta": {"artifact_type": "focus_failure"},
        "log_tail": [f"log-{index}" for index in range(250)],
        "focus_debug": {
            "window_tree": {
                "role": "AXWindow",
                "children": [
                    {"role": "AXGroup", "children": [{"role": "AXTextArea", "value": "x" * 2000}]}
                    for _ in range(40)
                ],
            }
        },
        "trigger_dump": {
            "context": "y" * 100_000,
            "focused": {"beforeCursor": "z" * 20_000, "afterCursor": "", "rawValue": "r" * 20_000},
            "conversationTurns": [{"speaker": "User", "text": "hi"} for _ in range(20)],
            "tree": {"role": "AXWindow", "children": [{"role": "AXGroup", "children": []} for _ in range(40)]},
        },
    }

    trimmed = trim_debug_artifact(artifact, max_chars=25_000)

    assert trimmed["meta"]["trimmed"] is True
    assert len(trimmed["log_tail"]) <= 100
    assert len(json.dumps(trimmed, sort_keys=True, ensure_ascii=False)) <= 25_000


def test_trim_debug_artifact_preserves_focus_summaries_before_dropping_full_trees():
    artifact = {
        "meta": {"artifact_type": "focus_failure"},
        "focus_debug": {
            "window_inventory": [{"title": "ChatGPT", "role": "AXWindow"} for _ in range(5)],
            "window_trees": [
                {
                    "index": index,
                    "role_counts": {"AXGroup": 40, "AXWebArea": 1},
                    "editable_candidates": [{"role": "AXTextArea", "value_preview": "x" * 100}],
                    "tree": {
                        "role": "AXWindow",
                        "children": [
                            {"role": "AXGroup", "children": [{"role": "AXTextArea", "value": "x" * 10_000}]}
                            for _ in range(30)
                        ],
                    },
                }
                for index in range(3)
            ],
            "cdp_probe": {
                "status": "success",
                "target_title": "ChatGPT",
                "editable_candidates": [{"tag": "textarea", "value_preview": "y" * 200}],
            },
        },
        "log_tail": [f"log-{index}" for index in range(300)],
        "trigger_dump": {"context": "z" * 80_000},
    }

    trimmed = trim_debug_artifact(artifact, max_chars=12_000)

    assert trimmed["meta"]["trimmed"] is True
    assert trimmed["focus_debug"]["window_inventory"]
    assert trimmed["focus_debug"]["cdp_probe"]["status"] == "success"
    assert "window_trees" in trimmed["focus_debug"]
    assert all("role_counts" in item for item in trimmed["focus_debug"]["window_trees"])
    assert all("tree" not in item for item in trimmed["focus_debug"]["window_trees"])


def test_debug_artifact_client_posts_with_install_auth(monkeypatch):
    captured = {}
    delivered = threading.Event()

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        delivered.set()
        return FakeResponse()

    monkeypatch.setattr("autocompleter.debug_capture.request.urlopen", fake_urlopen)

    client = DebugArtifactClient(
        enabled=True,
        url="https://proxy.example/v1/debug-artifacts",
        api_key="install-key",
        install_id="install-123",
        app_version="0.1.0",
        capture_mode="failures",
    )
    try:
        client.emit_artifact(
            "focus_failure",
            {"meta": {"artifact_type": "focus_failure"}, "payload": {"Authorization": "Bearer abc"}},
            invocation_id="inv-1",
            source_app="ChatGPT",
            trigger_type="manual",
        )
        assert delivered.wait(1.0)
    finally:
        client.stop()

    assert captured["url"] == "https://proxy.example/v1/debug-artifacts"
    assert captured["auth"] == "Bearer install-key"
    assert captured["body"]["artifact_type"] == "focus_failure"
    assert captured["body"]["invocation_id"] == "inv-1"
    assert captured["body"]["payload"]["payload"]["Authorization"] == "[redacted]"
