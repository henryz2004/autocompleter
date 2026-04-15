"""Tests for the beta telemetry client and helpers."""

from __future__ import annotations

from autocompleter.telemetry import (
    TelemetryClient,
    bucket_latency_ms,
    bucket_length,
    categorize_app,
)


class TestTelemetryHelpers:
    def test_latency_buckets(self):
        assert bucket_latency_ms(100) == "<250"
        assert bucket_latency_ms(250) == "250-500"
        assert bucket_latency_ms(700) == "500-1000"
        assert bucket_latency_ms(1500) == "1000-2000"
        assert bucket_latency_ms(2500) == "2000+"

    def test_length_buckets(self):
        assert bucket_length(5) == "1-10"
        assert bucket_length(20) == "11-30"
        assert bucket_length(40) == "31-80"
        assert bucket_length(120) == "81+"

    def test_app_categories(self):
        assert categorize_app("Terminal") == "terminal"
        assert categorize_app("Safari") == "browser"
        assert categorize_app("Slack") == "chat"
        assert categorize_app("TextEdit") == "editor"
        assert categorize_app("Preview") == "other"


class TestTelemetryClient:
    def test_emit_posts_expected_payload(self, monkeypatch):
        payloads = []
        client = TelemetryClient(
            enabled=True,
            url="https://telemetry.example/events",
            install_id="install-123",
            beta_mode=True,
            app_version="0.1.0",
        )
        monkeypatch.setattr(client, "_send_payload", lambda payload: payloads.append(payload))

        client.emit("trigger_fired", mode="reply", trigger_type="manual", app_category="chat")
        client.flush()
        client.stop()

        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["event"] == "trigger_fired"
        assert payload["install_id"] == "install-123"
        assert payload["app_version"] == "0.1.0"
        assert payload["beta_mode"] is True
        assert payload["mode"] == "reply"
        assert payload["trigger_type"] == "manual"
        assert payload["app_category"] == "chat"
        assert "timestamp" in payload
        assert "prompt" not in payload
        assert "suggestion_text" not in payload

    def test_disabled_client_noops(self, monkeypatch):
        client = TelemetryClient(
            enabled=False,
            url="https://telemetry.example/events",
            install_id="install-123",
            beta_mode=False,
            app_version="0.1.0",
        )
        monkeypatch.setattr(client, "_send_payload", lambda payload: (_ for _ in ()).throw(AssertionError("should not send")))
        client.emit("app_started")
        client.stop()

    def test_delivery_failures_do_not_raise(self, monkeypatch):
        client = TelemetryClient(
            enabled=True,
            url="https://telemetry.example/events",
            install_id="install-123",
            beta_mode=True,
            app_version="0.1.0",
        )
        monkeypatch.setattr(
            client,
            "_send_payload",
            lambda payload: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        client.emit("app_started")
        client.flush()
        client.stop()
