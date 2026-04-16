"""Configuration management for the autocompleter."""

from __future__ import annotations

import os
import uuid
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

    @property
    def feedback_dir(self) -> Path:
        return self.data_dir / "feedback"

    # LLM API
    llm_provider: str = "openai"  # "anthropic" or "openai" (cerebras/groq/etc. use "openai")
    llm_base_url: str = "https://api.cerebras.ai/v1"  # empty = SDK default
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "qwen-3-235b-a22b-instruct-2507"
    max_tokens: int = 200
    temperature: float = 0.7

    # Beta proxy
    beta_mode: bool = False
    proxy_enabled: bool = False
    proxy_base_url: str = ""
    proxy_api_key: str = ""

    # Fallback provider (used when primary doesn't respond in time)
    fallback_provider: str = "openai"  # same convention as llm_provider
    fallback_base_url: str = "https://api.groq.com/openai/v1"
    fallback_model: str = "qwen/qwen3-32b"
    fallback_api_key: str = "__UNSET__"
    escalation_timeout_ms: int = 400  # ms to wait before firing fallback

    # Per-mode overrides (continuation = low entropy, reply = higher entropy)
    continuation_temperature: float = 0.35
    reply_temperature: float = 0.8

    # Suggestion behavior
    num_suggestions: int = 3
    debounce_ms: int = 500
    hotkey: str = "ctrl+space"
    regenerate_hotkey: str = "ctrl+r"
    help_hotkey: str = "ctrl+/"
    report_hotkey: str = "ctrl+shift+b"
    followup_after_accept_enabled: bool = True

    # Context store
    context_window_chars: int = 4000

    # Overlay
    overlay_width: int = 500
    overlay_max_height: int = 300
    overlay_font_size: int = 13
    overlay_opacity: float = 0.95

    # Multi-line suggestions
    max_suggestion_lines: int = 4

    # Auto-trigger
    auto_trigger_enabled: bool = False
    auto_trigger_delay_ms: int = 3000
    auto_trigger_cooldown_ms: int = 5000  # cooldown after dismiss

    # Long-term memory (mem0)
    memory_enabled: bool = False
    memory_llm_provider: str = "groq"        # LLM used by mem0 for fact extraction
    memory_llm_model: str = "qwen/qwen3-32b"
    memory_embedder_provider: str = "openai"  # "openai" or "huggingface"
    memory_embedder_model: str = "text-embedding-3-small"
    memory_decay_rate: float = 0.01  # Exponential decay λ (half-life ~69h). 0 = no decay.

    # Telemetry
    telemetry_enabled: bool = False
    telemetry_url: str = ""
    telemetry_api_key: str = ""
    install_id: str = ""

    # Sentinel to distinguish "not provided" from "explicitly set to empty"
    _UNSET: str = "__UNSET__"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.openai_api_key:
            # Try provider-specific keys before falling back to OPENAI_API_KEY
            for env_var in ("CEREBRAS_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
                key = os.environ.get(env_var, "")
                if key:
                    self.openai_api_key = key
                    break
        if self.fallback_api_key == self._UNSET:
            # Not explicitly set — auto-resolve from env vars
            self.fallback_api_key = ""
            fallback_key_map = {
                "https://api.groq.com/openai/v1": "GROQ_API_KEY",
                "https://api.cerebras.ai/v1": "CEREBRAS_API_KEY",
                "https://api.sambanova.ai/v1": "SAMBANOVA_API_KEY",
                "https://api.together.xyz/v1": "TOGETHER_API_KEY",
                "https://api.fireworks.ai/inference/v1": "FIREWORKS_API_KEY",
                "https://api.deepinfra.com/v1/openai": "DEEPINFRA_API_KEY",
            }
            env_var = fallback_key_map.get(self.fallback_base_url)
            if env_var:
                self.fallback_api_key = os.environ.get(env_var, "")
            if not self.fallback_api_key:
                self.fallback_api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.install_id:
            self.install_id = self._resolve_install_id()

    @property
    def effective_llm_provider(self) -> str:
        if self.proxy_enabled:
            return "openai"
        return self.llm_provider

    @property
    def effective_llm_base_url(self) -> str:
        if self.proxy_enabled:
            return self.proxy_base_url
        return self.llm_base_url

    @property
    def effective_openai_api_key(self) -> str:
        if self.proxy_enabled:
            return self.proxy_api_key
        return self.openai_api_key

    @property
    def effective_llm_model(self) -> str:
        return self.llm_model

    @property
    def effective_fallback_provider(self) -> str:
        if self.proxy_enabled:
            return ""
        return self.fallback_provider

    @property
    def effective_fallback_base_url(self) -> str:
        if self.proxy_enabled:
            return ""
        return self.fallback_base_url

    @property
    def effective_fallback_model(self) -> str:
        if self.proxy_enabled:
            return ""
        return self.fallback_model

    @property
    def effective_fallback_api_key(self) -> str:
        if self.proxy_enabled:
            return ""
        return self.fallback_api_key

    @property
    def telemetry_active(self) -> bool:
        return self.telemetry_enabled and bool(self.telemetry_url.strip())

    @property
    def effective_telemetry_api_key(self) -> str:
        if self.telemetry_api_key:
            return self.telemetry_api_key
        if self.proxy_api_key:
            return self.proxy_api_key
        return ""

    def _resolve_install_id(self) -> str:
        install_id_path = self.data_dir / "install_id"
        try:
            if install_id_path.is_file():
                existing = install_id_path.read_text(encoding="utf-8").strip()
                if existing:
                    return existing
            generated = str(uuid.uuid4())
            install_id_path.write_text(generated, encoding="utf-8")
            return generated
        except Exception:
            return str(uuid.uuid4())


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

    def _env_bool(name: str, default: str = "") -> bool:
        return os.environ.get(name, default).lower() in ("1", "true")

    config = Config(
        llm_provider=os.environ.get("AUTOCOMPLETER_LLM_PROVIDER", "openai"),
        llm_base_url=os.environ.get(
            "AUTOCOMPLETER_LLM_BASE_URL", "https://api.cerebras.ai/v1"
        ),
        llm_model=os.environ.get(
            "AUTOCOMPLETER_LLM_MODEL", "qwen-3-235b-a22b-instruct-2507"
        ),
        beta_mode=_env_bool("AUTOCOMPLETER_BETA_MODE"),
        proxy_enabled=_env_bool("AUTOCOMPLETER_PROXY_ENABLED"),
        proxy_base_url=os.environ.get("AUTOCOMPLETER_PROXY_BASE_URL", ""),
        proxy_api_key=os.environ.get("AUTOCOMPLETER_PROXY_API_KEY", ""),
        fallback_provider=os.environ.get(
            "AUTOCOMPLETER_FALLBACK_PROVIDER", "openai"
        ),
        fallback_base_url=os.environ.get(
            "AUTOCOMPLETER_FALLBACK_BASE_URL", "https://api.groq.com/openai/v1"
        ),
        fallback_model=os.environ.get(
            "AUTOCOMPLETER_FALLBACK_MODEL", "qwen/qwen3-32b"
        ),
        escalation_timeout_ms=int(os.environ.get(
            "AUTOCOMPLETER_ESCALATION_TIMEOUT_MS", "400"
        )),
        hotkey=os.environ.get("AUTOCOMPLETER_HOTKEY", "ctrl+space"),
        regenerate_hotkey=os.environ.get("AUTOCOMPLETER_REGENERATE_HOTKEY", "ctrl+r"),
        help_hotkey=os.environ.get("AUTOCOMPLETER_HELP_HOTKEY", "ctrl+/"),
        report_hotkey=os.environ.get("AUTOCOMPLETER_REPORT_HOTKEY", "ctrl+shift+b"),
        followup_after_accept_enabled=os.environ.get(
            "AUTOCOMPLETER_FOLLOWUP_AFTER_ACCEPT", "1"
        ).lower() in ("1", "true"),
        auto_trigger_enabled=_env_bool("AUTOCOMPLETER_AUTO_TRIGGER"),
        memory_enabled=_env_bool("AUTOCOMPLETER_MEMORY"),
        memory_llm_provider=os.environ.get(
            "AUTOCOMPLETER_MEMORY_LLM_PROVIDER", "groq"
        ),
        memory_llm_model=os.environ.get(
            "AUTOCOMPLETER_MEMORY_LLM_MODEL", "qwen/qwen3-32b"
        ),
        memory_embedder_provider=os.environ.get(
            "AUTOCOMPLETER_MEMORY_EMBEDDER_PROVIDER", "openai"
        ),
        memory_embedder_model=os.environ.get(
            "AUTOCOMPLETER_MEMORY_EMBEDDER_MODEL", "text-embedding-3-small"
        ),
        memory_decay_rate=float(os.environ.get(
            "AUTOCOMPLETER_MEMORY_DECAY_RATE", "0.01"
        )),
        telemetry_enabled=_env_bool("AUTOCOMPLETER_TELEMETRY_ENABLED"),
        telemetry_url=os.environ.get("AUTOCOMPLETER_TELEMETRY_URL", ""),
        telemetry_api_key=os.environ.get("AUTOCOMPLETER_TELEMETRY_API_KEY", ""),
        install_id=os.environ.get("AUTOCOMPLETER_INSTALL_ID", ""),
    )
    return config
