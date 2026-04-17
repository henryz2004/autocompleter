"""Tests for backend debug artifact ingest."""

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


class TestBackendDebugArtifacts:
    def test_debug_artifact_is_accepted_and_redacted(self):
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
                "/v1/debug-artifacts",
                headers={"Authorization": f"Bearer {install_key}"},
                json={
                    "artifact_type": "focus_failure",
                    "invocation_id": "inv-123",
                    "source_app": "ChatGPT",
                    "trigger_type": "manual",
                    "payload": {
                        "meta": {
                            "artifact_type": "focus_failure",
                            "trigger_type": "manual",
                            "source_app": "ChatGPT",
                        },
                        "headers": {"Authorization": "Bearer secret-token"},
                    },
                },
            )

        assert response.status_code == 202
        assert len(store.debug_artifacts) == 1
        row = store.debug_artifacts[0]
        assert row["install_id"] == record.install_id
        assert row["invocation_id"] == "inv-123"
        assert row["artifact_type"] == "focus_failure"
        assert row["source_app"] == "ChatGPT"
        assert row["trigger_type"] == "manual"
        assert row["payload_json"]["install_id"] == record.install_id
        assert row["payload_json"]["payload"]["headers"]["Authorization"] == "[redacted]"

    def test_debug_artifact_requires_valid_install_key(self):
        store = InMemoryStore()
        config = make_config()
        app = create_app(
            config=config,
            store=store,
            proxy_service=ProxyService(config, store),
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/debug-artifacts",
                headers={"Authorization": "Bearer wrong-key"},
                json={
                    "artifact_type": "focus_failure",
                    "payload": {"meta": {"artifact_type": "focus_failure"}},
                },
            )

        assert response.status_code == 401
        assert store.debug_artifacts == []
