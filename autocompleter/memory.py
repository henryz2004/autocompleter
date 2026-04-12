"""Long-term memory powered by mem0.

Persists user knowledge, writing preferences, and conversation patterns
across sessions.  Uses FAISS for local vector storage (no external server)
and a configurable LLM (Groq by default) for fact extraction.

The module exposes a thin async-friendly wrapper around ``mem0.Memory`` so
that memory operations never block the hotkey / LLM pipeline.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

    The **pre-warm / cache** pattern keeps embedding API calls off the
    latency-critical trigger path:

    1. The observer loop calls ``pre_warm()`` every poll cycle with
       context signals (app, window title, visible text).
    2. ``pre_warm()`` builds a composite query, checks if it differs
       from the cached query, and if so fires a background search.
    3. At trigger time, ``get_cached_context()`` returns the pre-warmed
       formatted memory string instantly (no API call).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._mem: Any | None = None  # mem0 Memory instance
        self._lock = threading.Lock()
        self._initialized = False
        self._vector_count: int = 0  # track FAISS index size for short-circuit
        # Bounded thread pool for async memory ops (max 2 concurrent).
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mem0",
        )

        # ---- Pre-warm LRU cache ----
        self._cache_lock = threading.Lock()
        # OrderedDict[query -> (results_list, formatted_str)], most recent last.
        self._cache: OrderedDict[str, tuple[list[str], str]] = OrderedDict()
        self._cache_max_size: int = 8
        self._warm_in_flight_keys: set[str] = set()  # per-query dedup

        # ---- Decay ----
        self._decay_rate: float = getattr(config, "memory_decay_rate", 0.01)

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
            # Capture vector count for short-circuit optimization.
            try:
                vs = self._mem.vector_store  # type: ignore[union-attr]
                self._vector_count = getattr(vs, "index", None)
                if self._vector_count is not None:
                    self._vector_count = getattr(
                        self._vector_count, "ntotal", 0
                    )
                else:
                    self._vector_count = 0
            except Exception:
                self._vector_count = 0
            logger.info(
                "Memory system initialized "
                f"(llm={cfg.memory_llm_provider}/{cfg.memory_llm_model}, "
                f"embedder={cfg.memory_embedder_provider}/{cfg.memory_embedder_model}, "
                f"store=faiss @ {cfg.data_dir / 'memories'}, "
                f"vectors={self._vector_count})"
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

    @staticmethod
    def _apply_decay(
        entries: list[dict],
        decay_rate: float,
        now: datetime | None = None,
    ) -> list[dict]:
        """Apply exponential time-decay to search results and re-sort.

        Each entry dict must have ``score`` (float) and ``updated_at``
        (ISO 8601 string).  Returns entries with ``decayed_score`` added,
        sorted by decayed_score descending.

        ``decay_rate`` is the λ parameter: ``decayed = score × e^(-λ × hours)``.
        A rate of 0 disables decay (scores unchanged).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        for entry in entries:
            score = entry.get("score", 0.0) or 0.0
            if decay_rate <= 0:
                entry["decayed_score"] = score
                continue

            updated_at_str = entry.get("updated_at") or entry.get("created_at", "")
            if updated_at_str:
                try:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    hours_elapsed = max(0, (now - updated_at).total_seconds() / 3600)
                except (ValueError, TypeError):
                    hours_elapsed = 0
            else:
                hours_elapsed = 0

            decay_factor = math.exp(-decay_rate * hours_elapsed)
            entry["decayed_score"] = score * decay_factor

        entries.sort(key=lambda e: e.get("decayed_score", 0), reverse=True)
        return entries

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        user_id: str = _DEFAULT_USER_ID,
    ) -> list[str]:
        """Search memories relevant to *query*.

        Returns a list of memory strings, most relevant first (with
        time-decay applied when ``memory_decay_rate > 0``).
        Thread-safe and non-blocking for short queries.
        Returns ``[]`` when memory is disabled or on error.
        """
        if not self.enabled or not query.strip():
            return []

        # Short-circuit: skip the expensive embedding API call when the
        # FAISS index is empty — there's nothing to match against.
        if self._vector_count == 0:
            return []

        try:
            results = self._mem.search(  # type: ignore[union-attr]
                query=query,
                user_id=user_id,
                limit=limit,
            )
            # mem0 returns {"results": [{"memory": "...", "score": 0.9,
            #   "created_at": "...", "updated_at": "..."}, ...]}
            entries = results.get("results", []) if isinstance(results, dict) else results

            if self._decay_rate > 0:
                self._apply_decay(entries, self._decay_rate)

            memories: list[str] = []
            for entry in entries:
                text = entry.get("memory", "") if isinstance(entry, dict) else str(entry)
                if text.strip():
                    memories.append(text.strip())
                    if self._decay_rate > 0 and isinstance(entry, dict):
                        raw = entry.get("score", 0)
                        decayed = entry.get("decayed_score", raw)
                        age_h = 0
                        ts = entry.get("updated_at") or entry.get("created_at", "")
                        if ts:
                            try:
                                dt = datetime.fromisoformat(ts)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                            except (ValueError, TypeError):
                                pass
                        logger.debug(
                            f"[MEM] decay: {text[:50]!r} "
                            f"score={raw:.2f} → {decayed:.2f} (age={age_h:.0f}h)"
                        )
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
            # Update vector count so short-circuit reflects new memories.
            if result:
                self._vector_count += 1
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

    def get_all_with_ids(
        self,
        *,
        user_id: str = _DEFAULT_USER_ID,
        limit: int = 1000,
    ) -> list[dict]:
        """Return all stored memories with IDs for consolidation.

        Returns ``[{"id": "...", "memory": "...", "created_at": "...",
        "updated_at": "..."}, ...]``.
        """
        if not self.enabled:
            return []

        try:
            results = self._mem.get_all(user_id=user_id, limit=limit)  # type: ignore[union-attr]
            entries = results.get("results", []) if isinstance(results, dict) else results
            out: list[dict] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                mem_text = entry.get("memory", "")
                if mem_text.strip():
                    out.append({
                        "id": entry.get("id", ""),
                        "memory": mem_text.strip(),
                        "created_at": entry.get("created_at", ""),
                        "updated_at": entry.get("updated_at", ""),
                    })
            return out
        except Exception:
            logger.debug("Memory get_all_with_ids failed", exc_info=True)
            return []

    def delete(self, memory_id: str) -> bool:
        """Delete a single memory by ID.

        Returns True on success, False on error.
        """
        if not self.enabled:
            return False

        try:
            self._mem.delete(memory_id)  # type: ignore[union-attr]
            self._vector_count = max(0, self._vector_count - 1)
            logger.debug(f"[MEM] deleted memory {memory_id}")
            return True
        except Exception:
            logger.debug(f"Memory delete failed for {memory_id}", exc_info=True)
            return False

    def get_full_memory_context(self, max_chars: int = 1200) -> str:
        """Combine instructions file, memory file, and FAISS cache.

        Priority: instructions.md (always full) > memory.md (always full)
        > FAISS cached results (fills remaining budget).

        Returns empty string if all sources are empty.
        """
        parts: list[str] = []

        # 1. User instructions (never truncated).
        instructions_path = self._config.data_dir / "instructions.md"
        try:
            instructions = instructions_path.read_text(encoding="utf-8").strip()
            if instructions:
                parts.append(f"User instructions:\n{instructions}")
        except (FileNotFoundError, OSError):
            pass

        # 2. Long-term memory file (never truncated).
        memory_path = self._config.data_dir / "memory.md"
        try:
            memory_md = memory_path.read_text(encoding="utf-8").strip()
            if memory_md:
                parts.append(f"User profile:\n{memory_md}")
        except (FileNotFoundError, OSError):
            pass

        # 3. FAISS cached results (fill remaining budget).
        faiss_formatted = self.get_cached_context()
        if faiss_formatted:
            current_len = sum(len(p) for p in parts)
            remaining = max_chars - current_len
            if remaining > 50:  # worth including if >50 chars available
                # Replace "User memories:" header with "Recent context:"
                faiss_block = faiss_formatted.replace(
                    "User memories:", "Recent context:", 1
                )
                if len(faiss_block) > remaining:
                    faiss_block = faiss_block[:remaining].rsplit("\n", 1)[0]
                parts.append(faiss_block)

        return "\n\n".join(parts) if parts else ""

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

    # ---- Pre-warm / cache API ----

    @staticmethod
    def build_query(
        app_name: str = "",
        window_title: str = "",
        visible_snippet: str = "",
    ) -> str:
        """Build a composite memory search query from context signals.

        Combines app name, window title (often contains the contact /
        channel name), and a snippet of visible text to give the
        embedding model richer retrieval signal than cursor text alone.
        """
        parts: list[str] = []
        if app_name:
            parts.append(app_name)
        if window_title:
            parts.append(window_title)
        if visible_snippet:
            # Take last ~200 chars — most recent content is most relevant.
            parts.append(visible_snippet.strip()[-200:])
        return " | ".join(parts)

    def pre_warm(
        self,
        query: str,
        *,
        limit: int = 3,
        user_id: str = _DEFAULT_USER_ID,
    ) -> None:
        """Pre-warm the memory cache in the background.

        Called by the observer loop every poll cycle.  Uses an LRU cache
        (size ``_cache_max_size``) so switching back to a previously-seen
        app/tab reuses cached results instead of hitting the embedding API.

        If *query* is already cached, it's promoted to most-recent (no API
        call).  If a warm for this query is already in flight, this is a
        no-op.
        """
        if not self.enabled:
            return

        with self._cache_lock:
            if query in self._cache:
                # LRU hit — promote to most-recent.
                self._cache.move_to_end(query)
                return
            if query in self._warm_in_flight_keys:
                return
            self._warm_in_flight_keys.add(query)

        def _do_warm() -> None:
            try:
                memories = self.search(query, limit=limit, user_id=user_id)
                formatted = self.format_for_context(memories)
                with self._cache_lock:
                    self._cache[query] = (memories, formatted)
                    # Evict oldest entries beyond max size.
                    while len(self._cache) > self._cache_max_size:
                        self._cache.popitem(last=False)
                if formatted:
                    logger.debug(
                        f"[MEM] pre-warm: {len(memories)} memories "
                        f"({len(formatted)} chars) for query {query[:60]!r}"
                    )
            except Exception:
                logger.debug("Memory pre-warm failed", exc_info=True)
            finally:
                with self._cache_lock:
                    self._warm_in_flight_keys.discard(query)

        self._executor.submit(_do_warm)

    def get_cached_context(self, query: str = "") -> str:
        """Return the pre-warmed formatted memory string.

        This is the hot-path method called at trigger time.  If *query*
        is provided and found in the LRU cache, returns that entry.
        Otherwise returns the most recently cached entry (last item in
        the OrderedDict).

        No API calls, no blocking.
        """
        with self._cache_lock:
            if not self._cache:
                return ""
            if query and query in self._cache:
                self._cache.move_to_end(query)
                _, formatted = self._cache[query]
                return formatted
            # Return the most recently inserted/accessed entry.
            _, formatted = next(reversed(self._cache.values()))
            return formatted

    def get_cached_results(self, query: str = "") -> list[str]:
        """Return the raw pre-warmed memory list."""
        with self._cache_lock:
            if not self._cache:
                return []
            if query and query in self._cache:
                self._cache.move_to_end(query)
                results, _ = self._cache[query]
                return list(results)
            results, _ = next(reversed(self._cache.values()))
            return list(results)
