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
        assert cfg.effective_llm_model == ""
        assert cfg.effective_model_label == "backend-default"
        assert cfg.effective_request_route == "proxy"
        assert cfg.effective_fallback_api_key == ""

    def test_byo_mode_keeps_existing_provider_fields(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("AUTOCOMPLETER_PROXY_ENABLED", raising=False)
        # Avoid inheriting repository .env defaults during tests.
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "0")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_BASE_URL", "https://api.example/v1")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_MODEL", "qwen/qwen3-32b")

        cfg = load_config()

        assert cfg.proxy_enabled is False
        assert cfg.effective_llm_provider == "anthropic"
        assert cfg.effective_llm_base_url == "https://api.example/v1"
        assert cfg.effective_llm_model == "qwen/qwen3-32b"
        assert cfg.effective_request_route == "direct"

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

    def test_byo_mode_prefers_provider_specific_key_for_primary(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "0")
        monkeypatch.setenv("AUTOCOMPLETER_LLM_BASE_URL", "https://api.groq.com/openai/v1")
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")

        cfg = load_config()

        assert cfg.effective_openai_api_key == "groq-key"

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


class TestDebugCaptureConfig:
    def test_debug_capture_defaults_off(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_MODE", "")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_PROFILE", "")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_SUCCESS", "")

        cfg = load_config()

        assert cfg.debug_capture_mode == "off"
        assert cfg.debug_capture_profile == "normal"
        assert cfg.debug_capture_success is False
        assert cfg.debug_capture_active is False

    def test_debug_capture_mode_is_normalized_and_uses_proxy_url(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_BETA_MODE", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_BASE_URL", "https://proxy.example/v1/")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_API_KEY", "proxy-key")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_MODE", "BOTH")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_PROFILE", "AGGRESSIVE")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_SUCCESS", "1")

        cfg = load_config()

        assert cfg.debug_capture_mode == "both"
        assert cfg.debug_capture_profile == "aggressive"
        assert cfg.debug_capture_success is True
        assert cfg.debug_capture_active is True
        assert cfg.debug_capture_url == "https://proxy.example/v1/debug-artifacts"
        assert cfg.debug_capture_failures_enabled is True
        assert cfg.debug_capture_manual_enabled is True
        assert cfg.debug_capture_aggressive_enabled is True
        assert cfg.debug_capture_success_enabled is True

    def test_invalid_debug_capture_mode_falls_back_to_off(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_BETA_MODE", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_BASE_URL", "https://proxy.example/v1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_API_KEY", "proxy-key")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_MODE", "loud")

        cfg = load_config()

        assert cfg.debug_capture_mode == "off"
        assert cfg.debug_capture_active is False

    def test_invalid_debug_capture_profile_falls_back_to_normal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AUTOCOMPLETER_BETA_MODE", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_ENABLED", "1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_BASE_URL", "https://proxy.example/v1")
        monkeypatch.setenv("AUTOCOMPLETER_PROXY_API_KEY", "proxy-key")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_MODE", "both")
        monkeypatch.setenv("AUTOCOMPLETER_DEBUG_CAPTURE_PROFILE", "loud")

        cfg = load_config()

        assert cfg.debug_capture_profile == "normal"
        assert cfg.debug_capture_aggressive_enabled is False
