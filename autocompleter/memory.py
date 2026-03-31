"""Long-term memory powered by mem0.

Persists user knowledge, writing preferences, and conversation patterns
across sessions.  Uses FAISS for local vector storage (no external server)
and a configurable LLM (Groq by default) for fact extraction.

The module exposes a thin async-friendly wrapper around ``mem0.Memory`` so
that memory operations never block the hotkey / LLM pipeline.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

# Sentinel: mem0 is an optional dependency.  If not installed we degrade
# gracefully — MemoryStore.search() returns [] and .add() is a no-op.
try:
    from mem0 import Memory as Mem0Memory

    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False


# ---- Custom categories for autocompleter memories ----
MEMORY_CATEGORIES = [
    {
        "personal_info": (
            "Basic information about the user — name, job title, company, "
            "projects, contacts, locations."
        ),
    },
    {
        "writing_style": (
            "User's writing tone, vocabulary preferences, common phrases, "
            "formatting habits.  Includes per-app style variations."
        ),
    },
    {
        "frequent_topics": (
            "Topics, technologies, or subjects the user discusses regularly.  "
            "Tracks which topics appear in which apps."
        ),
    },
    {
        "app_preferences": (
            "Per-app behavior patterns — preferred response length, formality "
            "level, use of emoji, signature styles."
        ),
    },
]

# Default user id for single-user desktop autocompleter.
_DEFAULT_USER_ID = "default_user"


class MemoryStore:
    """Wrapper around mem0 ``Memory`` with async-friendly helpers.

    All public methods are safe to call from any thread.  Heavy operations
    (``add``, ``search``) can optionally run on a background thread via
    ``add_async`` to avoid blocking the hotkey pipeline.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._mem: Any | None = None  # mem0 Memory instance
        self._lock = threading.Lock()
        self._initialized = False
        # Bounded thread pool for async memory writes (max 2 concurrent).
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mem0-add",
        )

        if not config.memory_enabled:
            logger.info("Memory system disabled (set AUTOCOMPLETER_MEMORY=1 to enable)")
            return

        if not HAS_MEM0:
            logger.warning(
                "Memory enabled in config but mem0 is not installed.  "
                "Run: pip install mem0ai faiss-cpu"
            )
            return

        self._init_mem0()

    # ---- Initialization ----

    def _init_mem0(self) -> None:
        """Build the mem0 ``Memory`` instance from autocompleter config."""
        cfg = self._config

        # Resolve API keys from environment (same precedence as main config).
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openai_key = (
            os.environ.get("OPENAI_API_KEY", "")
            or cfg.openai_api_key
        )

        # Validate required API keys before attempting initialization.
        if cfg.memory_llm_provider == "groq" and not groq_key:
            logger.warning(
                "Memory LLM provider is 'groq' but GROQ_API_KEY is not set — "
                "memory disabled.  Set GROQ_API_KEY or change "
                "AUTOCOMPLETER_MEMORY_LLM_PROVIDER."
            )
            return
        if cfg.memory_embedder_provider == "openai" and not openai_key:
            logger.warning(
                "Memory embedder is 'openai' but OPENAI_API_KEY is not set — "
                "memory disabled.  Set OPENAI_API_KEY or change "
                "AUTOCOMPLETER_MEMORY_EMBEDDER_PROVIDER."
            )
            return

        mem0_config: dict[str, Any] = {
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "autocompleter_memories",
                    "path": str(cfg.data_dir / "memories"),
                },
            },
            "llm": {
                "provider": cfg.memory_llm_provider,
                "config": {
                    "model": cfg.memory_llm_model,
                    "temperature": 0.1,
                    "max_tokens": 1500,
                },
            },
            "embedder": {
                "provider": cfg.memory_embedder_provider,
                "config": {
                    "model": cfg.memory_embedder_model,
                },
            },
            "history_db_path": str(cfg.data_dir / "memory_history.db"),
            "version": "v1.1",
        }

        try:
            self._mem = Mem0Memory.from_config(mem0_config)
            self._initialized = True
            logger.info(
                "Memory system initialized "
                f"(llm={cfg.memory_llm_provider}/{cfg.memory_llm_model}, "
                f"embedder={cfg.memory_embedder_provider}/{cfg.memory_embedder_model}, "
                f"store=faiss @ {cfg.data_dir / 'memories'})"
            )
        except Exception:
            logger.exception("Failed to initialize mem0 — memory disabled")
            self._mem = None
            self._initialized = False

    # ---- Public API ----

    @property
    def enabled(self) -> bool:
        """True when mem0 is fully initialized and ready."""
        return self._initialized and self._mem is not None

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        user_id: str = _DEFAULT_USER_ID,
    ) -> list[str]:
        """Search memories relevant to *query*.

        Returns a list of memory strings, most relevant first.
        Thread-safe and non-blocking for short queries.
        Returns ``[]`` when memory is disabled or on error.
        """
        if not self.enabled or not query.strip():
            return []

        try:
            results = self._mem.search(  # type: ignore[union-attr]
                query=query,
                user_id=user_id,
                limit=limit,
            )
            # mem0 returns {"results": [{"memory": "...", "score": 0.9}, ...]}
            memories: list[str] = []
            entries = results.get("results", []) if isinstance(results, dict) else results
            for entry in entries:
                text = entry.get("memory", "") if isinstance(entry, dict) else str(entry)
                if text.strip():
                    memories.append(text.strip())
            if memories:
                logger.debug(
                    f"[MEM] search({query[:60]!r}) -> {len(memories)} memories"
                )
            return memories
        except Exception:
            logger.debug("Memory search failed", exc_info=True)
            return []

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str = _DEFAULT_USER_ID,
        metadata: dict[str, Any] | None = None,
    ) -> dict | None:
        """Add messages to memory (blocking).

        ``messages`` is a list of OpenAI-format chat dicts:
        ``[{"role": "user", "content": "..."}, ...]``

        Returns the mem0 result dict, or None on error.
        """
        if not self.enabled:
            return None

        try:
            result = self._mem.add(  # type: ignore[union-attr]
                messages,
                user_id=user_id,
                metadata=metadata,
            )
            logger.debug(f"[MEM] add() -> {result}")
            return result
        except Exception:
            logger.debug("Memory add failed", exc_info=True)
            return None

    def add_async(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str = _DEFAULT_USER_ID,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fire-and-forget memory addition via bounded thread pool.

        Use this from the hotkey / accept path to avoid blocking.
        Uses a ``ThreadPoolExecutor(max_workers=2)`` to prevent unbounded
        thread spawning on rapid accepts.
        """
        if not self.enabled:
            return

        self._executor.submit(
            self.add, messages, user_id=user_id, metadata=metadata,
        )

    def get_all(
        self,
        *,
        user_id: str = _DEFAULT_USER_ID,
        limit: int = 100,
    ) -> list[str]:
        """Return all stored memories for the user."""
        if not self.enabled:
            return []

        try:
            results = self._mem.get_all(user_id=user_id, limit=limit)  # type: ignore[union-attr]
            entries = results.get("results", []) if isinstance(results, dict) else results
            return [
                entry.get("memory", "") if isinstance(entry, dict) else str(entry)
                for entry in entries
                if (entry.get("memory", "") if isinstance(entry, dict) else str(entry)).strip()
            ]
        except Exception:
            logger.debug("Memory get_all failed", exc_info=True)
            return []

    def format_for_context(self, memories: list[str], max_chars: int = 600) -> str:
        """Format a list of memory strings into a context block for the LLM.

        Returns an empty string if no memories are provided.
        """
        if not memories:
            return ""

        parts: list[str] = []
        total = 0
        for mem in memories:
            if total + len(mem) > max_chars:
                break
            parts.append(f"- {mem}")
            total += len(mem)

        return "User memories:\n" + "\n".join(parts)
