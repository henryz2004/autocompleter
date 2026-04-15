"""Tests for proxy routing, fallback, and metadata logging."""

from __future__ import annotations

import asyncio
import json
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
        supabase_service_role_key="service-role-key",
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
                headers={"Authorization": f"Bearer {install_key}"},
                json={
                    "model": "beta-model",
                    "messages": [{"role": "user", "content": "hello backend"}],
                    "stream": False,
                    "temperature": 0.2,
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello there"
        assert len(calls) == 1
        assert len(store.proxy_requests) == 1
        row = store.proxy_requests[0]
        assert row["fallback_used"] is False
        assert row["status"] == "success"
        assert row["input_chars_estimate"] >= len("hello backend")
        assert row["output_chars_estimate"] == len("hello there")
        assert "messages" not in row
        assert "prompt" not in row

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
                headers={"Authorization": f"Bearer {install_key}"},
                json={
                    "model": "beta-model",
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
        row = store.proxy_requests[0]
        assert row["fallback_used"] is True
        assert row["status"] == "success"
        assert row["output_chars_estimate"] == len("hi there")

