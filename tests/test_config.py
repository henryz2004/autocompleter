"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from autocompleter.config import load_config


class TestFollowupAfterAcceptConfig:
    def test_followup_after_accept_defaults_on(self, monkeypatch):
        monkeypatch.delenv("AUTOCOMPLETER_FOLLOWUP_AFTER_ACCEPT", raising=False)
        cfg = load_config()
        assert cfg.followup_after_accept_enabled is True

    def test_followup_after_accept_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETER_FOLLOWUP_AFTER_ACCEPT", "false")
        cfg = load_config()
        assert cfg.followup_after_accept_enabled is False

    def test_help_and_report_hotkeys_can_be_overridden(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_HELP_HOTKEY", "cmd+/")
        monkeypatch.setenv("AUTOCOMPLETER_REPORT_HOTKEY", "cmd+shift+b")

        cfg = load_config()

        assert cfg.help_hotkey == "cmd+/"
        assert cfg.report_hotkey == "cmd+shift+b"
        assert cfg.feedback_dir == Path(tmp_path) / ".autocompleter" / "feedback"


class TestBetaProxyConfig:
    def test_proxy_enabled_overrides_effective_inference(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_BASE_URL", "https://proxy.example/v1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_API_KEY", "proxy-key")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_BASE_URL", "https://ignored.example/v1")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_MODEL", "beta-model")

        cfg = load_config()

        assert cfg.proxy_enabled is True
        assert cfg.effective_llm_provider == "openai"
        assert cfg.effective_llm_base_url == "https://proxy.example/v1"
        assert cfg.effective_openai_api_key == "proxy-key"
        assert cfg.effective_llm_model == "beta-model"
        assert cfg.effective_fallback_api_key == ""

    def test_byo_mode_keeps_existing_provider_fields(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("AUTOCOMPLETER_PROXY_ENABLED", raising=False)
        # Avoid inheriting repository .env defaults during tests.
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "0")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_BASE_URL", "https://api.example/v1")

        cfg = load_config()

        assert cfg.proxy_enabled is False
        assert cfg.effective_llm_provider == "anthropic"
        assert cfg.effective_llm_base_url == "https://api.example/v1"

    def test_telemetry_opt_out_does_not_disable_proxy_mode(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_BASE_URL", "https://proxy.example/v1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_API_KEY", "proxy-key")
        monkeypatch.setenv("AUTOCOMPLETER_TELEMETRY_ENABLED", "0")
        monkeypatch.setenv("AUTOCOMPLETER_TELEMETRY_URL", "https://telemetry.example/events")

        cfg = load_config()

        assert cfg.proxy_enabled is True
        assert cfg.telemetry_enabled is False
        assert cfg.telemetry_active is False
        assert cfg.effective_llm_base_url == "https://proxy.example/v1"

    def test_telemetry_auth_defaults_to_proxy_install_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_API_KEY", "proxy-key")

        cfg = load_config()

        assert cfg.effective_telemetry_api_key == "proxy-key"

    def test_install_id_is_created_and_reused(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("AUTOCOMPLETER_INSTALL_ID", raising=False)
        # Ensure repository default install id from .env does not leak into tests.
        monkeypatch.setenv("AUTOCOMPLETER_INSTALL_ID", "")

        cfg1 = load_config()
        cfg2 = load_config()

        install_id_path = Path(tmp_path) / ".autocompleter" / "install_id"
        assert cfg1.install_id
        assert cfg1.install_id == cfg2.install_id
        assert install_id_path.read_text(encoding="utf-8").strip() == cfg1.install_id
