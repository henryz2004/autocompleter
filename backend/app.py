"""FastAPI app for the friend-beta proxy backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import require_admin, require_install
from .config import BackendConfig, load_backend_config
from .proxy import ChatCompletionRequest, ProxyService
from .store import InstallRecord, SupabaseStore
from .telemetry import TelemetryEventPayload, build_telemetry_row


class CreateInstallKeyRequest(BaseModel):
    label: str | None = None
    notes: str | None = None


def create_app(
    *,
    config: BackendConfig | None = None,
    store: Any | None = None,
    proxy_service: ProxyService | None = None,
) -> FastAPI:
    backend_config = config or load_backend_config()
    backend_store = store or SupabaseStore(backend_config)
    backend_proxy_service = proxy_service or ProxyService(backend_config, backend_store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await backend_store.close()

    app = FastAPI(
        title="Autocompleter Beta Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.backend_config = backend_config
    app.state.store = backend_store
    app.state.proxy_service = backend_proxy_service

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "supabase_configured": bool(
                backend_config.supabase_url and backend_config.supabase_secret_key
            ),
            "primary_upstream_configured": backend_config.primary_upstream.enabled,
            "fallback_upstream_configured": backend_config.fallback_upstream.enabled,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        install: InstallRecord = Depends(require_install),
    ):
        return await backend_proxy_service.handle_chat_completion(
            payload,
            install_id=install.install_id,
            request=request,
        )

    @app.post("/v1/telemetry/events", status_code=status.HTTP_202_ACCEPTED)
    async def ingest_telemetry(
        payload: TelemetryEventPayload,
        install: InstallRecord = Depends(require_install),
    ) -> dict[str, object]:
        row = build_telemetry_row(
            install_id=install.install_id,
            payload=payload.model_dump(mode="json"),
        )
        await backend_store.record_telemetry_event(row)
        return {"accepted": True, "event_id": row["event_id"]}

    @app.post("/admin/install-keys")
    async def create_install_key(
        payload: CreateInstallKeyRequest,
        request: Request,
    ) -> dict[str, object]:
        require_admin(request)
        record, plaintext_key = await backend_store.create_install(
            label=payload.label,
            notes=payload.notes,
        )
        return {
            "install_id": record.install_id,
            "install_key": plaintext_key,
            "status": record.status,
            "label": record.label,
            "created_at": record.created_at,
        }

    @app.post("/admin/install-keys/{install_id}/revoke")
    async def revoke_install_key(install_id: str, request: Request) -> dict[str, object]:
        require_admin(request)
        revoked = await backend_store.revoke_install(install_id)
        if not revoked:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"revoked": False, "detail": "install not found or already revoked"},
            )
        return {"revoked": True, "install_id": install_id}

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
