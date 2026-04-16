"""Tests for the Supabase store contract."""

from __future__ import annotations

import asyncio
import json

import httpx

from backend.config import BackendConfig, UpstreamConfig
from backend.store import SupabaseStore


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


class TestSupabaseStore:
    def test_create_install_stores_hashed_key_only(self):
        captured_rows: list[dict[str, object]] = []

        def dispatch(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path.endswith("/beta_installs"):
                row = json.loads(request.content.decode("utf-8"))
                captured_rows.append(row)
                return httpx.Response(201, json=[row])
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        async def run_test() -> tuple[str, str]:
            client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
            store = SupabaseStore(make_config(), client=client)
            try:
                record, plaintext_key = await store.create_install(label="friend")
                return record.key_hash, plaintext_key
            finally:
                await store.close()

        key_hash, plaintext_key = asyncio.run(run_test())

        assert len(captured_rows) == 1
        row = captured_rows[0]
        assert row["key_hash"] == key_hash
        assert "install_key" not in row
        assert plaintext_key not in json.dumps(row)
