"""Configuration helpers for the beta backend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true")


def _parse_model_map(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("model map must be a JSON object")
    return {
        str(key).strip(): str(value).strip()
        for key, value in parsed.items()
        if str(key).strip() and str(value).strip()
    }


def _parse_allowed_models(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class UpstreamConfig:
    name: str
    base_url: str
    api_key: str
    default_model: str = ""
    model_map: dict[str, str] = field(default_factory=dict)
    allowed_models: set[str] = field(default_factory=set)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip() and self.api_key.strip())

    def resolve_model(self, requested_model: str) -> str:
        requested = requested_model.strip()
        if requested and self.allowed_models and requested not in self.allowed_models:
            raise ValueError(f"model '{requested}' is not allowed for {self.name}")
        if requested and requested in self.model_map:
            return self.model_map[requested]
        if requested:
            return requested
        if self.default_model:
            return self.default_model
        raise ValueError(f"no model configured for {self.name}")


@dataclass(frozen=True)
class BackendConfig:
    admin_secret: str
    supabase_url: str
    supabase_service_role_key: str
    request_timeout_s: float
    stream_timeout_s: float
    allow_upstream_override_headers: bool
    primary_upstream: UpstreamConfig
    fallback_upstream: UpstreamConfig

    @property
    def supabase_rest_url(self) -> str:
        return self.supabase_url.rstrip("/") + "/rest/v1"


def load_backend_config() -> BackendConfig:
    primary_name = os.environ.get(
        "AUTOCOMPLETER_PROXY_PRIMARY_NAME",
        "primary",
    ).strip() or "primary"
    fallback_name = os.environ.get(
        "AUTOCOMPLETER_PROXY_FALLBACK_NAME",
        "fallback",
    ).strip() or "fallback"

    primary = UpstreamConfig(
        name=primary_name,
        base_url=os.environ.get("AUTOCOMPLETER_PROXY_PRIMARY_BASE_URL", "").strip(),
        api_key=os.environ.get("AUTOCOMPLETER_PROXY_PRIMARY_API_KEY", "").strip(),
        default_model=os.environ.get(
            "AUTOCOMPLETER_PROXY_PRIMARY_DEFAULT_MODEL",
            "",
        ).strip(),
        model_map=_parse_model_map(
            os.environ.get("AUTOCOMPLETER_PROXY_PRIMARY_MODEL_MAP", "")
        ),
        allowed_models=_parse_allowed_models(
            os.environ.get("AUTOCOMPLETER_PROXY_PRIMARY_ALLOWED_MODELS", "")
        ),
    )
    fallback = UpstreamConfig(
        name=fallback_name,
        base_url=os.environ.get("AUTOCOMPLETER_PROXY_FALLBACK_BASE_URL", "").strip(),
        api_key=os.environ.get("AUTOCOMPLETER_PROXY_FALLBACK_API_KEY", "").strip(),
        default_model=os.environ.get(
            "AUTOCOMPLETER_PROXY_FALLBACK_DEFAULT_MODEL",
            "",
        ).strip(),
        model_map=_parse_model_map(
            os.environ.get("AUTOCOMPLETER_PROXY_FALLBACK_MODEL_MAP", "")
        ),
        allowed_models=_parse_allowed_models(
            os.environ.get("AUTOCOMPLETER_PROXY_FALLBACK_ALLOWED_MODELS", "")
        ),
    )
    return BackendConfig(
        admin_secret=os.environ.get("AUTOCOMPLETER_BACKEND_ADMIN_SECRET", "").strip(),
        supabase_url=os.environ.get("AUTOCOMPLETER_SUPABASE_URL", "").strip(),
        supabase_service_role_key=os.environ.get(
            "AUTOCOMPLETER_SUPABASE_SERVICE_ROLE_KEY",
            "",
        ).strip(),
        request_timeout_s=float(
            os.environ.get("AUTOCOMPLETER_BACKEND_REQUEST_TIMEOUT_S", "15")
        ),
        stream_timeout_s=float(
            os.environ.get("AUTOCOMPLETER_BACKEND_STREAM_TIMEOUT_S", "60")
        ),
        allow_upstream_override_headers=_env_bool(
            "AUTOCOMPLETER_BACKEND_ALLOW_UPSTREAM_OVERRIDE_HEADERS"
        ),
        primary_upstream=primary,
        fallback_upstream=fallback,
    )
