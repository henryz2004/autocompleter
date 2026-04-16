"""OpenAI-compatible proxy implementation with server-side fallback."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import BackendConfig, UpstreamConfig
from .store import utcnow_iso

logger = logging.getLogger(__name__)

INVOCATION_HEADER = "X-Autocompleter-Invocation-Id"
REQUEST_HEADER = "X-Autocompleter-Request-Id"


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list, min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class ProxyAttempt:
    upstream: UpstreamConfig
    resolved_model: str
    fallback_used: bool


def _extract_text_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_extract_text_length(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return _extract_text_length(value["text"])
        if "content" in value:
            return _extract_text_length(value["content"])
        return sum(_extract_text_length(item) for item in value.values())
    return 0


def estimate_input_chars(messages: list[dict[str, Any]]) -> int:
    return sum(_extract_text_length(message.get("content")) for message in messages)


def estimate_output_chars(response_json: dict[str, Any]) -> int:
    total = 0
    for choice in response_json.get("choices", []):
        if isinstance(choice, dict):
            total += _extract_text_length(choice.get("message", {}))
            total += _extract_text_length(choice.get("delta", {}))
            total += _extract_text_length(choice.get("text"))
    return total


class StreamingUsageCounter:
    """Tracks output character estimates from SSE chunks without storing text."""

    def __init__(self) -> None:
        self.output_chars_estimate = 0
        self._buffer = ""

    def observe(self, chunk: bytes) -> None:
        self._buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return
        self.output_chars_estimate += estimate_output_chars(parsed)


class ProxyService:
    def __init__(
        self,
        config: BackendConfig,
        store: Any,
        *,
        client_factory: Callable[[httpx.Timeout], httpx.AsyncClient] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._client_factory = client_factory or (
            lambda timeout: httpx.AsyncClient(timeout=timeout)
        )

    async def handle_chat_completion(
        self,
        payload: ChatCompletionRequest,
        *,
        install_id: str,
        request: Request,
    ) -> Response:
        request_id = str(uuid.uuid4())
        invocation_id = request.headers.get(INVOCATION_HEADER, "").strip() or None
        if payload.stream:
            response = await self._handle_streaming(
                payload,
                install_id=install_id,
                invocation_id=invocation_id,
                request_id=request_id,
                request=request,
            )
        else:
            response = await self._handle_non_streaming(
                payload,
                install_id=install_id,
                invocation_id=invocation_id,
                request_id=request_id,
                request=request,
            )
        response.headers[REQUEST_HEADER] = request_id
        if invocation_id:
            response.headers[INVOCATION_HEADER] = invocation_id
        return response

    async def _handle_non_streaming(
        self,
        payload: ChatCompletionRequest,
        *,
        install_id: str,
        invocation_id: str | None,
        request_id: str,
        request: Request,
    ) -> Response:
        request_received_at = utcnow_iso()
        started_perf = time.perf_counter()
        first_attempt_started_at: str | None = None
        attempts = self._build_attempts(payload.model, request)
        request_body = self._build_upstream_payload(payload, attempts[0].resolved_model, attempts[0].upstream)
        input_chars_estimate = estimate_input_chars(payload.messages)
        response_json: dict[str, Any] | None = None
        status_label = "error"
        error_type: str | None = None
        attempt_used: ProxyAttempt | None = None
        attempt_count = 0

        for attempt_number, attempt in enumerate(attempts, start=1):
            attempt_count = attempt_number
            attempt_started_at = utcnow_iso()
            first_attempt_started_at = first_attempt_started_at or attempt_started_at
            attempt_started_perf = time.perf_counter()
            attempt_start_offset_ms = self._offset_ms(started_perf, attempt_started_perf)
            request_body["model"] = attempt.resolved_model
            try:
                response = await self._post_json(attempt.upstream, request_body)
            except Exception as exc:
                attempt_latency_ms = self._elapsed_ms(attempt_started_perf)
                completed_at = utcnow_iso()
                await self._record_proxy_attempt(
                    attempt_id=str(uuid.uuid4()),
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    attempt_number=attempt_number,
                    attempt=attempt,
                    stream=False,
                    status="error",
                    error_type=type(exc).__name__,
                    latency_ms=attempt_latency_ms,
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=0,
                    started_at=attempt_started_at,
                    completed_at=completed_at,
                    profile_json={
                        "request_route": "proxy",
                        "timings_ms": {
                            "attempt_start_offset": attempt_start_offset_ms,
                            "upstream_response_complete": attempt_latency_ms,
                        },
                    },
                )
                if self._should_fallback_on_exception(exc) and attempt is attempts[0] and len(attempts) > 1:
                    logger.warning("Primary upstream failed before response; retrying fallback", exc_info=True)
                    continue
                error_type = type(exc).__name__
                await self._record_proxy_request(
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    requested_model=payload.model,
                    attempt=attempt,
                    stream=False,
                    status=status_label,
                    error_type=error_type,
                    latency_ms=self._elapsed_ms(started_perf),
                    message_count=len(payload.messages),
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=0,
                    attempt_count=attempt_count,
                    first_attempt_started_at=first_attempt_started_at or attempt_started_at,
                    completed_at=completed_at,
                    profile_json={
                        "request_route": "proxy",
                        "request_received_at": request_received_at,
                        "timings_ms": {
                            "proxy_total": self._elapsed_ms(started_perf),
                            "attempt_start_offset": attempt_start_offset_ms,
                            "upstream_response_complete": attempt_latency_ms,
                        },
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"upstream request failed: {error_type}",
                ) from exc

            if self._should_fallback_on_status(response.status_code) and attempt is attempts[0] and len(attempts) > 1:
                completed_at = utcnow_iso()
                await self._record_proxy_attempt(
                    attempt_id=str(uuid.uuid4()),
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    attempt_number=attempt_number,
                    attempt=attempt,
                    stream=False,
                    status="error",
                    error_type=f"http_{response.status_code}",
                    latency_ms=self._elapsed_ms(attempt_started_perf),
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=0,
                    started_at=attempt_started_at,
                    completed_at=completed_at,
                    profile_json={
                        "request_route": "proxy",
                        "http_status": response.status_code,
                        "timings_ms": {
                            "attempt_start_offset": attempt_start_offset_ms,
                            "upstream_response_complete": self._elapsed_ms(attempt_started_perf),
                        },
                    },
                )
                logger.warning("Primary upstream returned %s; retrying fallback", response.status_code)
                continue

            attempt_used = attempt
            if response.status_code >= 400:
                error_type = f"http_{response.status_code}"
                response_json = response.json()
            else:
                status_label = "success"
                response_json = response.json()
            completed_at = utcnow_iso()
            await self._record_proxy_attempt(
                attempt_id=str(uuid.uuid4()),
                request_id=request_id,
                install_id=install_id,
                invocation_id=invocation_id,
                attempt_number=attempt_number,
                attempt=attempt,
                stream=False,
                status=status_label,
                error_type=error_type,
                latency_ms=self._elapsed_ms(attempt_started_perf),
                input_chars_estimate=input_chars_estimate,
                output_chars_estimate=(
                    estimate_output_chars(response_json)
                    if status_label == "success"
                    else 0
                ),
                started_at=attempt_started_at,
                completed_at=completed_at,
                profile_json={
                    "request_route": "proxy",
                    "http_status": response.status_code,
                    "timings_ms": {
                        "attempt_start_offset": attempt_start_offset_ms,
                        "upstream_response_complete": self._elapsed_ms(attempt_started_perf),
                    },
                },
            )
            break

        if response_json is None or attempt_used is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="all upstreams failed",
            )

        await self._record_proxy_request(
            request_id=request_id,
            install_id=install_id,
            invocation_id=invocation_id,
            requested_model=payload.model,
            attempt=attempt_used,
            stream=False,
            status=status_label,
            error_type=error_type,
            latency_ms=self._elapsed_ms(started_perf),
            message_count=len(payload.messages),
            input_chars_estimate=input_chars_estimate,
            output_chars_estimate=estimate_output_chars(response_json),
            attempt_count=attempt_count,
            first_attempt_started_at=first_attempt_started_at or utcnow_iso(),
            completed_at=utcnow_iso(),
            profile_json={
                "request_route": "proxy",
                "request_received_at": request_received_at,
                "stream": False,
                "timings_ms": {
                    "proxy_total": self._elapsed_ms(started_perf),
                },
            },
        )
        status_code = 200
        if status_label != "success":
            status_code = (
                int(error_type.removeprefix("http_"))
                if error_type and error_type.startswith("http_")
                else 502
            )
        return JSONResponse(status_code=status_code, content=response_json)

    async def _handle_streaming(
        self,
        payload: ChatCompletionRequest,
        *,
        install_id: str,
        invocation_id: str | None,
        request_id: str,
        request: Request,
    ) -> Response:
        request_received_at = utcnow_iso()
        started_perf = time.perf_counter()
        first_attempt_started_at: str | None = None
        attempts = self._build_attempts(payload.model, request)
        request_body = self._build_upstream_payload(payload, attempts[0].resolved_model, attempts[0].upstream)
        message_count = len(payload.messages)
        input_chars_estimate = estimate_input_chars(payload.messages)
        attempt_used: ProxyAttempt | None = None
        client: httpx.AsyncClient | None = None
        response_cm = None
        upstream_response: httpx.Response | None = None
        first_chunk = b""
        stream_iterator = None
        attempt_used_number = 0
        attempt_used_started_at: str | None = None
        attempt_used_perf = 0.0
        attempt_used_headers_ms: int | None = None
        attempt_used_first_chunk_ms: int | None = None

        for attempt_number, attempt in enumerate(attempts, start=1):
            attempt_started_at = utcnow_iso()
            first_attempt_started_at = first_attempt_started_at or attempt_started_at
            attempt_started_perf = time.perf_counter()
            attempt_start_offset_ms = self._offset_ms(started_perf, attempt_started_perf)
            request_body["model"] = attempt.resolved_model
            timeout = httpx.Timeout(
                self.config.request_timeout_s,
                read=self.config.stream_timeout_s,
            )
            client = self._client_factory(timeout)
            response_cm = client.stream(
                "POST",
                self._upstream_url(attempt.upstream),
                headers=self._upstream_headers(attempt.upstream),
                json=request_body,
            )
            try:
                upstream_response = await response_cm.__aenter__()
                response_headers_ms = self._elapsed_ms(attempt_started_perf)
                if self._should_fallback_on_status(upstream_response.status_code) and attempt is attempts[0] and len(attempts) > 1:
                    await upstream_response.aread()
                    completed_at = utcnow_iso()
                    await self._record_proxy_attempt(
                        attempt_id=str(uuid.uuid4()),
                        request_id=request_id,
                        install_id=install_id,
                        invocation_id=invocation_id,
                        attempt_number=attempt_number,
                        attempt=attempt,
                        stream=True,
                        status="error",
                        error_type=f"http_{upstream_response.status_code}",
                        latency_ms=response_headers_ms,
                        input_chars_estimate=input_chars_estimate,
                        output_chars_estimate=0,
                        started_at=attempt_started_at,
                        completed_at=completed_at,
                        profile_json={
                            "request_route": "proxy",
                            "http_status": upstream_response.status_code,
                            "timings_ms": {
                                "attempt_start_offset": attempt_start_offset_ms,
                                "upstream_response_headers": response_headers_ms,
                            },
                        },
                    )
                    await response_cm.__aexit__(None, None, None)
                    await client.aclose()
                    client = None
                    response_cm = None
                    upstream_response = None
                    continue
                if upstream_response.status_code >= 400:
                    body = await upstream_response.aread()
                    await response_cm.__aexit__(None, None, None)
                    await client.aclose()
                    completed_at = utcnow_iso()
                    await self._record_proxy_attempt(
                        attempt_id=str(uuid.uuid4()),
                        request_id=request_id,
                        install_id=install_id,
                        invocation_id=invocation_id,
                        attempt_number=attempt_number,
                        attempt=attempt,
                        stream=True,
                        status="error",
                        error_type=f"http_{upstream_response.status_code}",
                        latency_ms=response_headers_ms,
                        input_chars_estimate=input_chars_estimate,
                        output_chars_estimate=0,
                        started_at=attempt_started_at,
                        completed_at=completed_at,
                        profile_json={
                            "request_route": "proxy",
                            "http_status": upstream_response.status_code,
                            "timings_ms": {
                                "attempt_start_offset": attempt_start_offset_ms,
                                "upstream_response_headers": response_headers_ms,
                            },
                        },
                    )
                    await self._record_proxy_request(
                        request_id=request_id,
                        install_id=install_id,
                        invocation_id=invocation_id,
                        requested_model=payload.model,
                        attempt=attempt,
                        stream=True,
                        status="error",
                        error_type=f"http_{upstream_response.status_code}",
                        latency_ms=self._elapsed_ms(started_perf),
                        message_count=message_count,
                        input_chars_estimate=input_chars_estimate,
                        output_chars_estimate=0,
                        attempt_count=attempt_number,
                        first_attempt_started_at=first_attempt_started_at or attempt_started_at,
                        completed_at=completed_at,
                        profile_json={
                            "request_route": "proxy",
                            "request_received_at": request_received_at,
                            "stream": True,
                            "timings_ms": {
                                "proxy_total": self._elapsed_ms(started_perf),
                                "attempt_start_offset": attempt_start_offset_ms,
                                "upstream_response_headers": response_headers_ms,
                            },
                        },
                    )
                    return Response(
                        content=body,
                        status_code=upstream_response.status_code,
                        media_type=upstream_response.headers.get("content-type", "application/json"),
                    )
                iterator = upstream_response.aiter_bytes()
                try:
                    first_chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    first_chunk = b""
                first_chunk_ms = self._elapsed_ms(attempt_started_perf)
                stream_iterator = iterator
                attempt_used = attempt
                attempt_used_number = attempt_number
                attempt_used_started_at = attempt_started_at
                attempt_used_perf = attempt_started_perf
                attempt_used_headers_ms = response_headers_ms
                attempt_used_first_chunk_ms = first_chunk_ms
                break
            except Exception as exc:
                if response_cm is not None:
                    await response_cm.__aexit__(type(exc), exc, exc.__traceback__)
                if client is not None:
                    await client.aclose()
                client = None
                response_cm = None
                upstream_response = None
                completed_at = utcnow_iso()
                await self._record_proxy_attempt(
                    attempt_id=str(uuid.uuid4()),
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    attempt_number=attempt_number,
                    attempt=attempt,
                    stream=True,
                    status="error",
                    error_type=type(exc).__name__,
                    latency_ms=self._elapsed_ms(attempt_started_perf),
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=0,
                    started_at=attempt_started_at,
                    completed_at=completed_at,
                    profile_json={
                        "request_route": "proxy",
                        "timings_ms": {
                            "attempt_start_offset": attempt_start_offset_ms,
                            "upstream_response_headers": self._elapsed_ms(attempt_started_perf),
                        },
                    },
                )
                if self._should_fallback_on_exception(exc) and attempt is attempts[0] and len(attempts) > 1:
                    logger.warning("Primary upstream failed before first stream chunk; retrying fallback", exc_info=True)
                    continue
                await self._record_proxy_request(
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    requested_model=payload.model,
                    attempt=attempt,
                    stream=True,
                    status="error",
                    error_type=type(exc).__name__,
                    latency_ms=self._elapsed_ms(started_perf),
                    message_count=message_count,
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=0,
                    attempt_count=attempt_number,
                    first_attempt_started_at=first_attempt_started_at or attempt_started_at,
                    completed_at=completed_at,
                    profile_json={
                        "request_route": "proxy",
                        "request_received_at": request_received_at,
                        "stream": True,
                        "timings_ms": {
                            "proxy_total": self._elapsed_ms(started_perf),
                            "attempt_start_offset": attempt_start_offset_ms,
                            "upstream_response_headers": self._elapsed_ms(attempt_started_perf),
                        },
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"upstream stream failed: {type(exc).__name__}",
                ) from exc

        if (
            attempt_used is None
            or client is None
            or response_cm is None
            or upstream_response is None
            or stream_iterator is None
        ):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="all upstreams failed",
            )

        counter = StreamingUsageCounter()

        async def event_stream() -> AsyncGenerator[bytes, None]:
            status_label = "success"
            error_type: str | None = None
            try:
                if first_chunk:
                    counter.observe(first_chunk)
                    yield first_chunk
                async for chunk in stream_iterator:
                    counter.observe(chunk)
                    yield chunk
            except Exception as exc:
                status_label = "error"
                error_type = type(exc).__name__
                raise
            finally:
                await response_cm.__aexit__(None, None, None)
                await client.aclose()
                await self._record_proxy_attempt(
                    attempt_id=str(uuid.uuid4()),
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    attempt_number=attempt_used_number,
                    attempt=attempt_used,
                    stream=True,
                    status=status_label,
                    error_type=error_type,
                    latency_ms=self._elapsed_ms(attempt_used_perf),
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=counter.output_chars_estimate,
                    started_at=attempt_used_started_at or utcnow_iso(),
                    completed_at=utcnow_iso(),
                    profile_json={
                        "request_route": "proxy",
                        "timings_ms": {
                            "upstream_response_headers": attempt_used_headers_ms,
                            "upstream_first_chunk": attempt_used_first_chunk_ms,
                            "upstream_stream_complete": self._elapsed_ms(attempt_used_perf),
                        },
                    },
                )
                await self._record_proxy_request(
                    request_id=request_id,
                    install_id=install_id,
                    invocation_id=invocation_id,
                    requested_model=payload.model,
                    attempt=attempt_used,
                    stream=True,
                    status=status_label,
                    error_type=error_type,
                    latency_ms=self._elapsed_ms(started_perf),
                    message_count=message_count,
                    input_chars_estimate=input_chars_estimate,
                    output_chars_estimate=counter.output_chars_estimate,
                    attempt_count=attempt_used_number,
                    first_attempt_started_at=first_attempt_started_at or utcnow_iso(),
                    completed_at=utcnow_iso(),
                    profile_json={
                        "request_route": "proxy",
                        "request_received_at": request_received_at,
                        "stream": True,
                        "timings_ms": {
                            "proxy_total": self._elapsed_ms(started_perf),
                            "upstream_response_headers": attempt_used_headers_ms,
                            "upstream_first_chunk": attempt_used_first_chunk_ms,
                            "upstream_stream_complete": self._elapsed_ms(attempt_used_perf),
                        },
                    },
                )

        return StreamingResponse(
            event_stream(),
            media_type=upstream_response.headers.get("content-type", "text/event-stream"),
        )

    def _build_attempts(self, requested_model: str, request: Request) -> list[ProxyAttempt]:
        if not self.config.primary_upstream.enabled:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="primary upstream is not configured",
            )
        try:
            primary_model = self.config.primary_upstream.resolve_model(requested_model)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        primary = ProxyAttempt(
            upstream=self.config.primary_upstream,
            resolved_model=primary_model,
            fallback_used=False,
        )
        attempts = [primary]
        if self.config.fallback_upstream.enabled:
            try:
                fallback_model = self.config.fallback_upstream.resolve_model(requested_model)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            attempts.append(
                ProxyAttempt(
                    upstream=self.config.fallback_upstream,
                    resolved_model=fallback_model,
                    fallback_used=True,
                )
            )
        if self.config.allow_upstream_override_headers:
            override = request.headers.get("X-Autocompleter-Upstream", "").strip().lower()
            if override == "fallback" and len(attempts) > 1:
                return [attempts[1], attempts[0]]
        return attempts

    def _build_upstream_payload(
        self,
        payload: ChatCompletionRequest,
        resolved_model: str,
        upstream: UpstreamConfig,
    ) -> dict[str, Any]:
        body = {
            "model": resolved_model,
            "messages": payload.messages,
            "stream": payload.stream,
        }
        if payload.temperature is not None:
            body["temperature"] = payload.temperature
        if payload.max_tokens is not None:
            body["max_tokens"] = payload.max_tokens
        for key, value in payload.model_extra.items():
            if key != "extra_body":
                body[key] = value
        hostname = urlparse(upstream.base_url).hostname or ""
        if hostname.endswith("groq.com"):
            body["reasoning_effort"] = "none"
        if payload.extra_body:
            body.update(payload.extra_body)
        return body

    def _upstream_url(self, upstream: UpstreamConfig) -> str:
        return upstream.base_url.rstrip("/") + "/chat/completions"

    def _upstream_headers(self, upstream: UpstreamConfig) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {upstream.api_key}",
            "Content-Type": "application/json",
        }

    async def _post_json(
        self,
        upstream: UpstreamConfig,
        body: dict[str, Any],
    ) -> httpx.Response:
        timeout = httpx.Timeout(self.config.request_timeout_s)
        async with self._client_factory(timeout) as client:
            return await client.post(
                self._upstream_url(upstream),
                headers=self._upstream_headers(upstream),
                json=body,
            )

    def _should_fallback_on_status(self, status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    def _should_fallback_on_exception(self, exc: Exception) -> bool:
        return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError))

    def _elapsed_ms(self, started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    @staticmethod
    def _offset_ms(origin: float, point: float) -> int:
        return int((point - origin) * 1000)

    async def _record_proxy_request(
        self,
        *,
        request_id: str,
        install_id: str,
        invocation_id: str | None,
        requested_model: str,
        attempt: ProxyAttempt,
        stream: bool,
        status: str,
        error_type: str | None,
        latency_ms: int,
        message_count: int,
        input_chars_estimate: int,
        output_chars_estimate: int,
        attempt_count: int,
        first_attempt_started_at: str,
        completed_at: str,
        profile_json: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "request_id": request_id,
            "install_id": install_id,
            "invocation_id": invocation_id,
            "requested_model": requested_model,
            "resolved_model": attempt.resolved_model,
            "primary_upstream": self.config.primary_upstream.name,
            "fallback_used": attempt.fallback_used,
            "stream": stream,
            "status": status,
            "error_type": error_type,
            "latency_ms": latency_ms,
            "message_count": message_count,
            "input_chars_estimate": input_chars_estimate,
            "output_chars_estimate": output_chars_estimate,
            "attempt_count": attempt_count,
            "first_attempt_started_at": first_attempt_started_at,
            "completed_at": completed_at,
            "created_at": first_attempt_started_at,
        }
        if profile_json is not None:
            row["profile_json"] = profile_json
        await self.store.record_proxy_request(row)

    async def _record_proxy_attempt(
        self,
        *,
        attempt_id: str,
        request_id: str,
        install_id: str,
        invocation_id: str | None,
        attempt_number: int,
        attempt: ProxyAttempt,
        stream: bool,
        status: str,
        error_type: str | None,
        latency_ms: int,
        input_chars_estimate: int,
        output_chars_estimate: int,
        started_at: str,
        completed_at: str,
        profile_json: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "attempt_id": attempt_id,
            "request_id": request_id,
            "install_id": install_id,
            "invocation_id": invocation_id,
            "attempt_number": attempt_number,
            "upstream_name": attempt.upstream.name,
            "resolved_model": attempt.resolved_model,
            "is_fallback_attempt": attempt.fallback_used,
            "stream": stream,
            "status": status,
            "error_type": error_type,
            "latency_ms": latency_ms,
            "input_chars_estimate": input_chars_estimate,
            "output_chars_estimate": output_chars_estimate,
            "started_at": started_at,
            "completed_at": completed_at,
            "created_at": started_at,
        }
        if profile_json is not None:
            row["profile_json"] = profile_json
        await self.store.record_proxy_attempt(row)
