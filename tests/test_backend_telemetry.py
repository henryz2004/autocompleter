"""Tests for backend telemetry ingest and sanitization."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import BackendConfig, UpstreamConfig
from backend.proxy import ProxyService
from backend.store import InMemoryStore


def make_config() -> BackendConfig:
    return BackendConfig(
        admin_secret="admin-secret",
        supabase_url="https://supabase.example.co",
        supabase_secret_key="sb_secret_test",
        request_timeout_s=5.0,
        stream_timeout_s=20.0,
        allow_upstream_override_headers=False,
        primary_upstream=UpstreamConfig(
            name="primary",
            base_url="https://primary.example/v1",
            api_key="primary-key",
            default_model="beta-model",
        ),
        fallback_upstream=UpstreamConfig(
            name="fallback",
            base_url="https://fallback.example/v1",
            api_key="fallback-key",
            default_model="beta-model",
        ),
    )


class TestBackendTelemetry:
    def test_telemetry_payload_is_accepted_and_sanitized(self):
        store = InMemoryStore()
        config = make_config()
        app = create_app(
            config=config,
            store=store,
            proxy_service=ProxyService(config, store),
        )
        record, install_key = asyncio.run(store.create_install(label="friend"))

        with TestClient(app) as client:
            response = client.post(
                "/v1/telemetry/events",
                headers={"Authorization": f"Bearer {install_key}"},
                json={
                    "event": "trigger_fired",
                    "install_id": "client-generated-id",
                    "mode": "reply",
                    "prompt": "do not persist me",
                    "url": "https://sensitive.example/path",
                },
            )

        assert response.status_code == 202
        assert len(store.telemetry_events) == 1
        row = store.telemetry_events[0]
        assert row["install_id"] == record.install_id
        assert row["event_name"] == "trigger_fired"
        assert row["payload_json"]["install_id"] == record.install_id
        assert row["payload_json"]["mode"] == "reply"
        assert "prompt" not in row["payload_json"]
        assert "url" not in row["payload_json"]

    def test_malformed_telemetry_payload_is_rejected(self):
        store = InMemoryStore()
        config = make_config()
        app = create_app(
            config=config,
            store=store,
            proxy_service=ProxyService(config, store),
        )
        _, install_key = asyncio.run(store.create_install(label="friend"))

        with TestClient(app) as client:
            response = client.post(
                "/v1/telemetry/events",
                headers={"Authorization": f"Bearer {install_key}"},
                json={"mode": "reply"},
            )

        assert response.status_code == 422
        assert store.telemetry_events == []
