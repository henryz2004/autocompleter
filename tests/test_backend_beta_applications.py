"""Tests for the public /v1/beta/applications intake endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import BackendConfig, UpstreamConfig
from backend.proxy import ProxyService
from backend.store import InMemoryStore


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
        public_cors_origins=["https://autocompleter.dev"],
        public_cors_origin_regex=r"^https://[a-z0-9-]+\.autocompleter-259\.pages\.dev$",
        public_install_docs_url="https://example.com/docs/friend-beta",
    )


def _make_app(store: InMemoryStore | None = None):
    store = store or InMemoryStore()
    config = make_config()
    app = create_app(
        config=config,
        store=store,
        proxy_service=ProxyService(config, store),
    )
    return app, store


def _valid_payload(**overrides):
    body = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "role": "Founding engineer",
        "primary_use_case": "Reply drafting in Slack and terminal commands.",
    }
    body.update(overrides)
    return body


class TestBetaApplicationsHappyPath:
    def test_submission_returns_credentials_and_env_setup(self):
        app, store = _make_app()
        with TestClient(app, base_url="https://proxy.example") as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(),
            )
        assert response.status_code == 201, response.text
        body = response.json()

        # Credentials are returned exactly once, immediately.
        assert body["install_id"]
        assert body["install_key"]
        assert body["application_id"]
        assert body["status"] == "granted"

        # Setup URLs and .env block are rendered for the success screen.
        assert body["proxy_base_url"] == "https://proxy.example/v1"
        assert body["telemetry_url"] == "https://proxy.example/v1/telemetry/events"
        assert body["install_docs_url"] == "https://example.com/docs/friend-beta"
        env_block = body["env_setup"]
        assert f"AUTOCOMPLETER_INSTALL_ID={body['install_id']}" in env_block
        assert f"AUTOCOMPLETER_PROXY_API_KEY={body['install_key']}" in env_block
        assert "AUTOCOMPLETER_PROXY_BASE_URL=https://proxy.example/v1" in env_block
        assert "AUTOCOMPLETER_BETA_MODE=1" in env_block

        # Install is linked to a real, active install record and application row.
        assert body["install_id"] in store.installs
        application = next(iter(store.applications.values()))
        assert application.email == "ada@example.com"
        assert application.email_normalized == "ada@example.com"
        assert application.install_id == body["install_id"]
        assert application.status == "granted"
        assert application.granted_at is not None

    def test_email_is_normalized_to_lowercase(self):
        app, store = _make_app()
        with TestClient(app, base_url="https://proxy.example") as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(email="  MIXED@Case.COM "),
            )
        assert response.status_code == 201
        application = next(iter(store.applications.values()))
        assert application.email == "MIXED@Case.COM"
        assert application.email_normalized == "mixed@case.com"

    def test_cors_preflight_succeeds_for_allowed_origin(self):
        app, _ = _make_app()
        with TestClient(app, base_url="https://proxy.example") as client:
            response = client.options(
                "/v1/beta/applications",
                headers={
                    "Origin": "https://autocompleter.dev",
                    "Access-Control-Request-Method": "POST",
                },
            )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://autocompleter.dev"

    def test_cors_preflight_succeeds_for_preview_origin_regex(self):
        app, _ = _make_app()
        with TestClient(app, base_url="https://proxy.example") as client:
            response = client.options(
                "/v1/beta/applications",
                headers={
                    "Origin": "https://7afa4c9d.autocompleter-259.pages.dev",
                    "Access-Control-Request-Method": "POST",
                },
            )
        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"]
            == "https://7afa4c9d.autocompleter-259.pages.dev"
        )


class TestBetaApplicationsDuplicateEmail:
    def test_duplicate_email_returns_409_and_does_not_mint_new_install(self):
        app, store = _make_app()
        with TestClient(app) as client:
            first = client.post("/v1/beta/applications", json=_valid_payload())
            assert first.status_code == 201
            install_count_after_first = len(store.installs)

            second = client.post(
                "/v1/beta/applications",
                json=_valid_payload(name="Somebody Else"),
            )

        assert second.status_code == 409
        body = second.json()
        assert "already" in body["detail"].lower()
        # Crucially: no new install key is reissued.
        assert "install_key" not in body
        assert "email" not in body
        assert len(store.installs) == install_count_after_first
        assert len(store.applications) == 1

    def test_duplicate_email_is_case_insensitive(self):
        app, store = _make_app()
        with TestClient(app) as client:
            client.post("/v1/beta/applications", json=_valid_payload())
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(email="ADA@example.com"),
            )
        assert response.status_code == 409


class TestBetaApplicationsHoneypot:
    def test_honeypot_filled_returns_422_and_does_not_mint(self):
        app, store = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(company="bot corp"),
            )
        assert response.status_code == 422
        assert "install_key" not in response.json()
        assert store.installs == {}
        assert store.applications == {}

    def test_empty_honeypot_whitespace_is_not_treated_as_bot(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(company="   "),
            )
        assert response.status_code == 201


class TestBetaApplicationsInvalidPayload:
    def test_missing_required_field_rejected(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            payload = _valid_payload()
            payload.pop("email")
            response = client.post("/v1/beta/applications", json=payload)
        assert response.status_code == 422

    def test_invalid_email_shape_rejected(self):
        app, store = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(email="not-an-email"),
            )
        assert response.status_code == 422
        assert store.installs == {}
        assert store.applications == {}

    def test_blank_required_string_rejected(self):
        app, _ = _make_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(name=""),
            )
        assert response.status_code == 422


class TestBetaApplicationsNoPartialGrantOnFailure:
    def test_install_is_revoked_when_application_insert_fails(self):
        store = InMemoryStore()
        store.application_insert_should_fail = True
        app, _ = _make_app(store)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(),
            )
        # Unhandled upstream error surfaces as 500 via FastAPI default handler.
        assert response.status_code == 500
        # The install that was minted before the failure must be revoked,
        # so no orphaned active install_id exists.
        assert len(store.installs) == 1
        install = next(iter(store.installs.values()))
        assert install.status == "revoked"
        assert install.revoked_at is not None
        assert store.applications == {}

    def test_race_on_duplicate_email_is_recovered_as_409(self):
        """If two submits race, the loser should see 409 not 500."""
        store = InMemoryStore()
        store.application_insert_simulates_race = True
        app, _ = _make_app(store)
        with TestClient(app) as client:
            response = client.post(
                "/v1/beta/applications",
                json=_valid_payload(),
            )
        assert response.status_code == 409
        body = response.json()
        assert "install_key" not in body
        # The loser's install is revoked; only the racer's record stands.
        install = next(iter(store.installs.values()))
        assert install.status == "revoked"
        assert len(store.applications) == 1


class TestBetaApplicationsDoNotLeakPII:
    def test_install_notes_do_not_contain_email(self):
        app, store = _make_app()
        with TestClient(app) as client:
            response = client.post("/v1/beta/applications", json=_valid_payload())
        assert response.status_code == 201
        install = next(iter(store.installs.values()))
        # Email lives in beta_applications only — not leaked into
        # beta_installs.notes.
        assert install.notes is None or "@" not in (install.notes or "")
