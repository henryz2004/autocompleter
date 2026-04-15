"""Auth helpers for install-key and admin-secret validation."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from .store import InstallRecord


def parse_bearer_token(header_value: str | None) -> str:
    if not header_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization header",
        )
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="expected Bearer token",
        )
    return token.strip()


async def require_install(request: Request) -> InstallRecord:
    install_key = parse_bearer_token(request.headers.get("Authorization"))
    store = request.app.state.store
    install = await store.get_install_by_key(install_key)
    if install is None or not install.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or revoked install key",
        )
    return install


def require_admin(request: Request) -> None:
    expected = request.app.state.backend_config.admin_secret
    provided = request.headers.get("X-Admin-Secret", "").strip()
    if not expected or provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin secret",
        )
