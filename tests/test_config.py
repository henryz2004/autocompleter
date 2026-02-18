"""Tests for configuration."""

import os
from unittest.mock import patch

from autocompleter.config import Config, load_config


class TestConfig:
    def test_default_config(self, tmp_path):
        config = Config(data_dir=tmp_path / "autocompleter")
        assert config.llm_provider == "anthropic"
        assert config.num_suggestions == 3
        assert config.debounce_ms == 500
        assert config.hotkey == "ctrl+space"

    def test_db_path(self, tmp_path):
        config = Config(data_dir=tmp_path / "autocompleter")
        assert config.db_path == tmp_path / "autocompleter" / "context.db"

    def test_data_dir_created(self, tmp_path):
        data_dir = tmp_path / "autocompleter" / "nested"
        Config(data_dir=data_dir)
        assert data_dir.exists()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"})
    def test_api_key_from_env(self, tmp_path):
        config = Config(data_dir=tmp_path / "autocompleter")
        assert config.anthropic_api_key == "env-key"

    @patch.dict(
        os.environ,
        {
            "AUTOCOMPLETER_LLM_PROVIDER": "openai",
            "AUTOCOMPLETER_HOTKEY": "cmd+j",
        },
    )
    def test_load_config_from_env(self):
        config = load_config()
        assert config.llm_provider == "openai"
        assert config.hotkey == "cmd+j"
