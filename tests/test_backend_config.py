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
