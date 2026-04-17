"""FastAPI app for the friend-beta proxy backend."""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .auth import require_admin, require_install
from .config import BackendConfig, load_backend_config
from .debug_artifacts import DebugArtifactPayload, build_debug_artifact_row
from .proxy import ChatCompletionRequest, ProxyService
from .store import DuplicateApplicationError, InstallRecord, SupabaseStore
from .telemetry import (
    TelemetryEventPayload,
    build_invocation_row,
    build_telemetry_row,
)

logger = logging.getLogger(__name__)


class CreateInstallKeyRequest(BaseModel):
    label: str | None = None
    notes: str | None = None


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class BetaApplicationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field(..., min_length=1, max_length=200)
    primary_use_case: str = Field(..., min_length=1, max_length=2000)
    # Honeypot — hidden field that real users never fill in. Bots often do.
    company: str | None = Field(default=None, max_length=200)
    source: str | None = Field(default=None, max_length=200)

    @field_validator("name", "email", "role", "primary_use_case")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed

    @field_validator("company", "source")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


def _duplicate_email_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "an application for this email already exists"},
    )


def _build_env_setup_block(
    *,
    install_id: str,
    install_key: str,
    proxy_base_url: str,
    telemetry_url: str,
) -> str:
    """Render the exact `.env` block a new tester pastes in to finish setup."""
    lines = [
        "AUTOCOMPLETER_BETA_MODE=1",
        "AUTOCOMPLETER_PROXY_ENABLED=1",
        f"AUTOCOMPLETER_PROXY_BASE_URL={proxy_base_url}",
        f"AUTOCOMPLETER_PROXY_API_KEY={install_key}",
        "AUTOCOMPLETER_TELEMETRY_ENABLED=1",
        f"AUTOCOMPLETER_TELEMETRY_URL={telemetry_url}",
        f"AUTOCOMPLETER_INSTALL_ID={install_id}",
    ]
    return "\n".join(lines) + "\n"


def _public_backend_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


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

    if backend_config.public_cors_origins or backend_config.public_cors_origin_regex:
        # CORS is primarily for the landing page hitting /v1/beta/applications.
        # We also allow Authorization so future browser-side clients (e.g. a web
        # console) can call the bearer-authed routes without a preflight tell.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=backend_config.public_cors_origins,
            allow_origin_regex=backend_config.public_cors_origin_regex or None,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )

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
        invocation_row = build_invocation_row(
            install_id=install.install_id,
            payload=payload.model_dump(mode="json"),
        )
        if invocation_row is not None:
            await backend_store.upsert_invocation(invocation_row)
        return {"accepted": True, "event_id": row["event_id"]}

    @app.post("/v1/debug-artifacts", status_code=status.HTTP_202_ACCEPTED)
    async def ingest_debug_artifact(
        payload: DebugArtifactPayload,
        install: InstallRecord = Depends(require_install),
    ) -> dict[str, object]:
        row = build_debug_artifact_row(
            install_id=install.install_id,
            payload=payload.model_dump(mode="json"),
        )
        await backend_store.record_debug_artifact(row)
        return {"accepted": True, "artifact_id": row["artifact_id"]}

    @app.post("/v1/beta/applications", status_code=status.HTTP_201_CREATED)
    async def create_beta_application(
        payload: BetaApplicationRequest,
        request: Request,
    ) -> JSONResponse:
        # NOTE: The 201 response contains a one-time plaintext install key.
        # Do NOT add access-log body dumps or broad request-logging middleware
        # that would capture it on this route.

        # Honeypot: real users never see or fill this field.
        if payload.company:
            logger.info("beta application rejected: honeypot triggered")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": "submission rejected"},
            )

        email = payload.email
        if not _EMAIL_RE.match(email):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": "invalid email address"},
            )

        existing = await backend_store.get_application_by_email(email)
        if existing is not None:
            return _duplicate_email_response()

        # Mint an install first so we can link it into the application row.
        # If the application insert fails we revoke the install immediately
        # to avoid leaving an orphaned grant. The `notes` field intentionally
        # does NOT include the email — the email lives only in
        # beta_applications, which has tighter access boundaries.
        install_record, plaintext_key = await backend_store.create_install(
            label="beta-application",
            notes=None,
        )
        try:
            application = await backend_store.create_application(
                email=email,
                name=payload.name,
                role=payload.role,
                primary_use_case=payload.primary_use_case,
                install_id=install_record.install_id,
                source=payload.source,
            )
        except DuplicateApplicationError:
            try:
                await backend_store.revoke_install(install_record.install_id)
            except Exception:
                logger.exception(
                    "failed to revoke install %s after duplicate application",
                    install_record.install_id,
                )
            logger.info(
                "duplicate beta application rejected; revoked install %s",
                install_record.install_id,
            )
            return _duplicate_email_response()
        except Exception as exc:
            try:
                await backend_store.revoke_install(install_record.install_id)
            except Exception:
                logger.exception(
                    "failed to revoke install %s after application failure",
                    install_record.install_id,
                )
            logger.error(
                "beta application insert failed (%s); revoked install %s",
                type(exc).__name__,
                install_record.install_id,
            )
            raise

        public_backend_base_url = _public_backend_base_url(request)
        proxy_base_url = f"{public_backend_base_url}/v1"
        telemetry_url = f"{proxy_base_url}/telemetry/events"

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "application_id": application.application_id,
                "install_id": install_record.install_id,
                "install_key": plaintext_key,
                "status": application.status,
                "proxy_base_url": proxy_base_url,
                "telemetry_url": telemetry_url,
                "install_docs_url": backend_config.public_install_docs_url,
                "env_setup": _build_env_setup_block(
                    install_id=install_record.install_id,
                    install_key=plaintext_key,
                    proxy_base_url=proxy_base_url,
                    telemetry_url=telemetry_url,
                ),
            },
        )

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
        host=os.environ.get("BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT") or os.environ.get("BACKEND_PORT") or "8000"),
        reload=False,
    )
