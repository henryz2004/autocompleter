"""Configuration management for the autocompleter."""

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


def load_config() -> Config:
    """Load configuration, using environment variables and defaults."""
    config = Config(
        llm_provider=os.environ.get("AUTOCOMPLETER_LLM_PROVIDER", "anthropic"),
        llm_model=os.environ.get(
            "AUTOCOMPLETER_LLM_MODEL", "claude-sonnet-4-20250514"
        ),
        hotkey=os.environ.get("AUTOCOMPLETER_HOTKEY", "ctrl+space"),
    )
    return config
