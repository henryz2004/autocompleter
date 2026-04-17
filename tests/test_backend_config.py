"""Tests for beta backend configuration loading."""

from __future__ import annotations

from backend.config import load_backend_config


class TestBackendConfig:
    def test_prefers_supabase_secret_key_env(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_URL", "https://supabase.example.co")
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_SECRET_KEY", "sb_secret_new")
        monkeypatch.setenv(
            "AUTOCOMPLETER_SUPABASE_SERVICE_ROLE_KEY",
            "sb_secret_legacy",
        )

        config = load_backend_config()

        assert config.supabase_secret_key == "sb_secret_new"

    def test_falls_back_to_legacy_service_role_env(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_URL", "https://supabase.example.co")
        monkeypatch.delenv("AUTOCOMPLETER_SUPABASE_SECRET_KEY", raising=False)
        monkeypatch.setenv(
            "AUTOCOMPLETER_SUPABASE_SERVICE_ROLE_KEY",
            "sb_secret_legacy",
        )

        config = load_backend_config()

        assert config.supabase_secret_key == "sb_secret_legacy"

    def test_loads_public_landing_env_settings(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_URL", "https://supabase.example.co")
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_SECRET_KEY", "sb_secret_new")
        monkeypatch.setenv(
            "AUTOCOMPLETER_PUBLIC_ALLOWED_ORIGINS",
            "https://autocompleter.dev,http://localhost:4321",
        )
        monkeypatch.setenv(
            "AUTOCOMPLETER_PUBLIC_ALLOWED_ORIGIN_REGEX",
            r"^https://[a-z0-9-]+\.autocompleter-259\.pages\.dev$",
        )
        monkeypatch.setenv(
            "AUTOCOMPLETER_PUBLIC_INSTALL_DOCS_URL",
            "https://example.com/docs/friend-beta",
        )

        config = load_backend_config()

        assert config.public_cors_origins == [
            "https://autocompleter.dev",
            "http://localhost:4321",
        ]
        assert (
            config.public_cors_origin_regex
            == r"^https://[a-z0-9-]+\.autocompleter-259\.pages\.dev$"
        )
        assert config.public_install_docs_url == "https://example.com/docs/friend-beta"

    def test_defaults_include_local_dev_and_pages_preview_cors(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_URL", "https://supabase.example.co")
        monkeypatch.setenv("AUTOCOMPLETER_SUPABASE_SECRET_KEY", "sb_secret_new")
        monkeypatch.delenv("AUTOCOMPLETER_PUBLIC_ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("AUTOCOMPLETER_PUBLIC_ALLOWED_ORIGIN_REGEX", raising=False)

        config = load_backend_config()

        assert "http://127.0.0.1:4321" in config.public_cors_origins
        assert "http://localhost:4321" in config.public_cors_origins
        assert "https://autocompleter.dev" in config.public_cors_origins
        assert "autocompleter-259\\.pages\\.dev" in config.public_cors_origin_regex
