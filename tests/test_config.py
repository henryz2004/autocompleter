"""Tests for configuration."""

import os
from unittest.mock import patch

from autocompleter.config import Config, load_config


class TestConfig:
    def test_data_dir_created(self, tmp_path):
        data_dir = tmp_path / "autocompleter" / "nested"
        Config(data_dir=data_dir)
        assert data_dir.exists()

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
