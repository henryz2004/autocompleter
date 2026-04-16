"""Tests for backend auth and admin flows."""

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


class TestBackendAuth:
    def test_admin_endpoints_require_separate_secret(self):
        store = InMemoryStore()
        app = create_app(
            config=make_config(),
            store=store,
            proxy_service=ProxyService(make_config(), store),
        )

        with TestClient(app) as client:
            response = client.post("/admin/install-keys", json={"label": "friend"})
            assert response.status_code == 401

            authed = client.post(
                "/admin/install-keys",
                headers={"X-Admin-Secret": "admin-secret"},
                json={"label": "friend"},
            )
            assert authed.status_code == 200
            body = authed.json()
            assert body["install_id"]
            assert body["install_key"]

    def test_revoked_install_key_is_rejected(self):
        store = InMemoryStore()
        config = make_config()
        app = create_app(
            config=config,
            store=store,
            proxy_service=ProxyService(config, store),
        )
        record, install_key = asyncio.run(store.create_install(label="friend"))

        with TestClient(app) as client:
            accepted = client.post(
                "/v1/telemetry/events",
                headers={"Authorization": f"Bearer {install_key}"},
                json={"event": "app_started"},
            )
            assert accepted.status_code == 202

            revoked = client.post(
                f"/admin/install-keys/{record.install_id}/revoke",
                headers={"X-Admin-Secret": "admin-secret"},
            )
            assert revoked.status_code == 200
            assert revoked.json()["revoked"] is True

            rejected = client.post(
                "/v1/telemetry/events",
                headers={"Authorization": f"Bearer {install_key}"},
                json={"event": "app_started"},
            )
            assert rejected.status_code == 401
