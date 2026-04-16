"""Tests for proxy routing, fallback, and metadata logging."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from collections.abc import AsyncIterator

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import BackendConfig, UpstreamConfig
from backend.proxy import ProxyService
from backend.store import InMemoryStore


class ChunkedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


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


class TestBackendProxy:
    def test_proxy_adds_groq_reasoning_effort_none_by_default(self):
        captured_bodies: list[dict] = []

        def dispatch(request: httpx.Request) -> httpx.Response:
            captured_bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello there"}}
                    ],
                },
            )

        store = InMemoryStore()
        config = make_config()
        config = replace(
            config,
            primary_upstream=replace(
                config.primary_upstream,
                base_url="https://api.groq.com/openai/v1",
            ),
        )
        proxy = ProxyService(
            config,
            store,
            client_factory=lambda timeout: httpx.AsyncClient(
                transport=httpx.MockTransport(dispatch),
                timeout=timeout,
            ),
        )
        app = create_app(config=config, store=store, proxy_service=proxy)
        _, install_key = asyncio.run(store.create_install(label="friend"))

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {install_key}"},
                json={
                    "model": "",
                    "messages": [{"role": "user", "content": "hello backend"}],
                    "stream": False,
                },
            )

        assert response.status_code == 200
        assert captured_bodies[0]["reasoning_effort"] == "none"

    def test_non_streaming_proxy_passthrough_logs_metadata_only(self):
        calls: list[str] = []

        def dispatch(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello there"}}
                    ],
                },
            )

        store = InMemoryStore()
        config = make_config()
        proxy = ProxyService(
            config,
            store,
            client_factory=lambda timeout: httpx.AsyncClient(
                transport=httpx.MockTransport(dispatch),
                timeout=timeout,
            ),
        )
        app = create_app(config=config, store=store, proxy_service=proxy)
        _, install_key = asyncio.run(store.create_install(label="friend"))

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {install_key}",
                    "X-Autocompleter-Invocation-Id": "inv-123",
                },
                json={
                    "model": "",
                    "messages": [{"role": "user", "content": "hello backend"}],
                    "stream": False,
                    "temperature": 0.2,
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello there"
        assert response.headers["X-Autocompleter-Invocation-Id"] == "inv-123"
        assert response.headers["X-Autocompleter-Request-Id"]
        assert len(calls) == 1
        assert len(store.proxy_requests) == 1
        assert len(store.proxy_attempts) == 1
        row = store.proxy_requests[0]
        assert row["invocation_id"] == "inv-123"
        assert row["requested_model"] == ""
        assert row["resolved_model"] == "beta-model"
        assert row["fallback_used"] is False
        assert row["status"] == "success"
        assert row["attempt_count"] == 1
        assert row["input_chars_estimate"] >= len("hello backend")
        assert row["output_chars_estimate"] == len("hello there")
        assert row["profile_json"]["request_route"] == "proxy"
        assert row["profile_json"]["timings_ms"]["proxy_total"] >= 0
        assert "messages" not in row
        assert "prompt" not in row
        attempt = store.proxy_attempts[0]
        assert attempt["request_id"] == row["request_id"]
        assert attempt["attempt_number"] == 1
        assert attempt["is_fallback_attempt"] is False
        assert attempt["profile_json"]["timings_ms"]["upstream_response_complete"] >= 0

    def test_streaming_proxy_passthrough_and_server_side_fallback(self):
        calls: list[str] = []

        def dispatch(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if "primary.example" in str(request.url):
                return httpx.Response(503, json={"error": {"message": "try fallback"}})
            chunks = [
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ChunkedAsyncStream(chunks),
            )

        store = InMemoryStore()
        config = make_config()
        proxy = ProxyService(
            config,
            store,
            client_factory=lambda timeout: httpx.AsyncClient(
                transport=httpx.MockTransport(dispatch),
                timeout=timeout,
            ),
        )
        app = create_app(config=config, store=store, proxy_service=proxy)
        _, install_key = asyncio.run(store.create_install(label="friend"))

        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {install_key}",
                    "X-Autocompleter-Invocation-Id": "inv-456",
                },
                json={
                    "model": "",
                    "messages": [{"role": "user", "content": "hello backend"}],
                    "stream": True,
                },
            ) as response:
                body = b"".join(response.iter_bytes())

        assert response.status_code == 200
        decoded = body.decode("utf-8")
        assert "hi" in decoded
        assert "there" in decoded
        assert len(calls) == 2
        assert "primary.example" in calls[0]
        assert "fallback.example" in calls[1]
        assert len(store.proxy_requests) == 1
        assert len(store.proxy_attempts) == 2
        row = store.proxy_requests[0]
        assert row["invocation_id"] == "inv-456"
        assert row["requested_model"] == ""
        assert row["fallback_used"] is True
        assert row["status"] == "success"
        assert row["attempt_count"] == 2
        assert row["output_chars_estimate"] == len("hi there")
        assert row["profile_json"]["timings_ms"]["upstream_first_chunk"] >= 0
        assert store.proxy_attempts[0]["is_fallback_attempt"] is False
        assert store.proxy_attempts[1]["is_fallback_attempt"] is True
        assert store.proxy_attempts[1]["profile_json"]["timings_ms"]["upstream_first_chunk"] >= 0
