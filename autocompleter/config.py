"""Configuration management for the autocompleter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: Path = field(
        default_factory=lambda: Path.home() / ".autocompleter"
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "context.db"

    # LLM API
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 150
    temperature: float = 0.7

    # Suggestion behavior
    num_suggestions: int = 3
    debounce_ms: int = 500
    hotkey: str = "ctrl+space"

    # Context store
    max_context_age_hours: int = 72
    max_context_entries: int = 5000
    context_window_chars: int = 4000

    # Overlay
    overlay_width: int = 400
    overlay_max_height: int = 200
    overlay_font_size: int = 13
    overlay_opacity: float = 0.95

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")


def _load_dotenv() -> None:
    """Load .env file from the project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_config() -> Config:
    """Load configuration, using .env file, environment variables, and defaults."""
    _load_dotenv()
    config = Config(
        llm_provider=os.environ.get("AUTOCOMPLETER_LLM_PROVIDER", "anthropic"),
        llm_model=os.environ.get(
            "AUTOCOMPLETER_LLM_MODEL", "claude-sonnet-4-20250514"
        ),
        hotkey=os.environ.get("AUTOCOMPLETER_HOTKEY", "ctrl+space"),
    )
    return config
