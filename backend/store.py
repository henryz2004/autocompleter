"""Supabase-backed persistence for the beta backend."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import BackendConfig


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_install_key(install_key: str) -> str:
    return hashlib.sha256(install_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InstallRecord:
    install_id: str
    key_hash: str
    status: str
    label: str | None
    created_at: str | None
    revoked_at: str | None
    notes: str | None

    @property
    def is_active(self) -> bool:
        return self.status == "active" and not self.revoked_at

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "InstallRecord":
        return cls(
            install_id=str(row.get("install_id", "")),
            key_hash=str(row.get("key_hash", "")),
            status=str(row.get("status", "")),
            label=row.get("label"),
            created_at=row.get("created_at"),
            revoked_at=row.get("revoked_at"),
            notes=row.get("notes"),
        )


class SupabaseStore:
    """Small PostgREST client used by the beta backend."""

    def __init__(
        self,
        config: BackendConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.request_timeout_s)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_install_by_key(self, install_key: str) -> InstallRecord | None:
        key_hash = hash_install_key(install_key)
        rows = await self._select_rows(
            "beta_installs",
            params={
                "select": "install_id,key_hash,status,label,created_at,revoked_at,notes",
                "key_hash": f"eq.{key_hash}",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return InstallRecord.from_row(rows[0])

    async def create_install(
        self,
        *,
        label: str | None = None,
        notes: str | None = None,
    ) -> tuple[InstallRecord, str]:
        install_id = str(uuid.uuid4())
        plaintext_key = secrets.token_urlsafe(32)
        row = {
            "install_id": install_id,
            "key_hash": hash_install_key(plaintext_key),
            "status": "active",
            "label": label,
            "created_at": utcnow_iso(),
            "revoked_at": None,
            "notes": notes,
        }
        inserted = await self._insert_row("beta_installs", row)
        return InstallRecord.from_row(inserted), plaintext_key

    async def revoke_install(self, install_id: str) -> bool:
        rows = await self._patch_rows(
            "beta_installs",
            {
                "status": "revoked",
                "revoked_at": utcnow_iso(),
            },
            params={
                "install_id": f"eq.{install_id}",
                "status": "eq.active",
            },
        )
        return bool(rows)

    async def record_proxy_request(self, row: dict[str, Any]) -> None:
        await self._insert_row("beta_proxy_requests", row, return_representation=False)

    async def record_proxy_attempt(self, row: dict[str, Any]) -> None:
        await self._insert_row("beta_proxy_attempts", row, return_representation=False)

    async def record_telemetry_event(self, row: dict[str, Any]) -> None:
        await self._insert_row("beta_telemetry_events", row, return_representation=False)

    async def upsert_invocation(self, row: dict[str, Any]) -> None:
        await self._upsert_row(
            "beta_invocations",
            row,
            on_conflict="invocation_id",
            return_representation=False,
        )

    async def _select_rows(
        self,
        table: str,
        *,
        params: dict[str, str],
    ) -> list[dict[str, Any]]:
        response = await self._client.get(
            f"{self.config.supabase_rest_url}/{table}",
            params=params,
            headers=self._headers(),
        )
        self._raise_for_status(response)
        return list(response.json())

    async def _insert_row(
        self,
        table: str,
        row: dict[str, Any],
        *,
        return_representation: bool = True,
    ) -> dict[str, Any]:
        headers = self._headers(prefer="return=representation" if return_representation else "return=minimal")
        response = await self._client.post(
            f"{self.config.supabase_rest_url}/{table}",
            json=row,
            headers=headers,
        )
        self._raise_for_status(response)
        if not return_representation:
            return row
        payload = response.json()
        return payload[0] if isinstance(payload, list) else payload

    async def _upsert_row(
        self,
        table: str,
        row: dict[str, Any],
        *,
        on_conflict: str,
        return_representation: bool = True,
    ) -> dict[str, Any]:
        prefer = "resolution=merge-duplicates"
        prefer += ",return=representation" if return_representation else ",return=minimal"
        response = await self._client.post(
            f"{self.config.supabase_rest_url}/{table}",
            params={"on_conflict": on_conflict},
            json=row,
            headers=self._headers(prefer=prefer),
        )
        self._raise_for_status(response)
        if not return_representation:
            return row
        payload = response.json()
        return payload[0] if isinstance(payload, list) else payload

    async def _patch_rows(
        self,
        table: str,
        row: dict[str, Any],
        *,
        params: dict[str, str],
    ) -> list[dict[str, Any]]:
        response = await self._client.patch(
            f"{self.config.supabase_rest_url}/{table}",
            params=params,
            json=row,
            headers=self._headers(prefer="return=representation"),
        )
        self._raise_for_status(response)
        return list(response.json())

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._error_detail(response)
            if detail:
                message = f"{exc}\nSupabase response body: {detail}"
                raise httpx.HTTPStatusError(
                    message,
                    request=exc.request,
                    response=exc.response,
                ) from exc
            raise

    @staticmethod
    def _error_detail(response: httpx.Response, *, max_chars: int = 500) -> str:
        text = response.text.strip()
        if not text:
            return ""
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.config.supabase_secret_key,
            "Authorization": f"Bearer {self.config.supabase_secret_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers


class InMemoryStore:
    """Test-friendly store implementing the same contract as SupabaseStore."""

    def __init__(self) -> None:
        self.installs: dict[str, InstallRecord] = {}
        self.proxy_requests: list[dict[str, Any]] = []
        self.proxy_attempts: list[dict[str, Any]] = []
        self.telemetry_events: list[dict[str, Any]] = []
        self.invocations: dict[str, dict[str, Any]] = {}

    async def close(self) -> None:
        return None

    async def get_install_by_key(self, install_key: str) -> InstallRecord | None:
        key_hash = hash_install_key(install_key)
        for install in self.installs.values():
            if install.key_hash == key_hash:
                return install
        return None

    async def create_install(
        self,
        *,
        label: str | None = None,
        notes: str | None = None,
    ) -> tuple[InstallRecord, str]:
        install_id = str(uuid.uuid4())
        plaintext_key = secrets.token_urlsafe(32)
        record = InstallRecord(
            install_id=install_id,
            key_hash=hash_install_key(plaintext_key),
            status="active",
            label=label,
            created_at=utcnow_iso(),
            revoked_at=None,
            notes=notes,
        )
        self.installs[install_id] = record
        return record, plaintext_key

    async def revoke_install(self, install_id: str) -> bool:
        record = self.installs.get(install_id)
        if record is None or not record.is_active:
            return False
        self.installs[install_id] = InstallRecord(
            install_id=record.install_id,
            key_hash=record.key_hash,
            status="revoked",
            label=record.label,
            created_at=record.created_at,
            revoked_at=utcnow_iso(),
            notes=record.notes,
        )
        return True

    async def record_proxy_request(self, row: dict[str, Any]) -> None:
        self.proxy_requests.append(dict(row))

    async def record_proxy_attempt(self, row: dict[str, Any]) -> None:
        self.proxy_attempts.append(dict(row))

    async def record_telemetry_event(self, row: dict[str, Any]) -> None:
        self.telemetry_events.append(dict(row))

    async def upsert_invocation(self, row: dict[str, Any]) -> None:
        invocation_id = str(row["invocation_id"])
        existing = dict(self.invocations.get(invocation_id, {}))
        existing.update(row)
        self.invocations[invocation_id] = existing
