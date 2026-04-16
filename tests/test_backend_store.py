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

    def test_insert_error_includes_supabase_response_body(self):
        def dispatch(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "code": "PGRST204",
                    "details": None,
                    "hint": None,
                    "message": "Could not find the 'invocation_id' column of 'beta_telemetry_events' in the schema cache",
                },
                request=request,
            )

        async def run_test() -> str:
            client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
            store = SupabaseStore(make_config(), client=client)
            try:
                try:
                    await store.record_telemetry_event(
                        {
                            "event_id": "evt-1",
                            "install_id": "inst-1",
                            "event_name": "trigger_fired",
                            "payload_json": {"event": "trigger_fired"},
                        }
                    )
                except httpx.HTTPStatusError as exc:
                    return str(exc)
                raise AssertionError("expected HTTPStatusError")
            finally:
                await store.close()

        message = asyncio.run(run_test())

        assert "Supabase response body:" in message
        assert "invocation_id" in message
