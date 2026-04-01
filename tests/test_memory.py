"""Tests for the MemoryStore wrapper (autocompleter.memory)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.config import Config
from autocompleter.memory import MemoryStore, HAS_MEM0, _DEFAULT_USER_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(memory_enabled: bool = True, **overrides) -> Config:
    """Create a Config with memory enabled and temp data dir."""
    import tempfile
    data_dir = Path(tempfile.mkdtemp())
    defaults = dict(
        data_dir=data_dir,
        memory_enabled=memory_enabled,
        memory_llm_provider="groq",
        memory_llm_model="qwen/qwen3-32b",
        memory_embedder_provider="openai",
        memory_embedder_model="text-embedding-3-small",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# Disabled / no-op tests
# ---------------------------------------------------------------------------

class TestMemoryDisabled:
    """When memory_enabled=False, everything is a safe no-op."""

    def test_disabled_by_config(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        assert not store.enabled
        assert store.search("hello") == []
        assert store.add([{"role": "user", "content": "hi"}]) is None
        assert store.get_all() == []

    @patch.dict("autocompleter.memory.__dict__", {"HAS_MEM0": False})
    def test_disabled_when_mem0_missing(self):
        """If mem0 is not installed, gracefully degrade."""
        import autocompleter.memory as mod
        original = mod.HAS_MEM0
        mod.HAS_MEM0 = False
        try:
            store = MemoryStore(_make_config(memory_enabled=True))
            assert not store.enabled
        finally:
            mod.HAS_MEM0 = original


# ---------------------------------------------------------------------------
# Mock-based tests (no real mem0 calls)
# ---------------------------------------------------------------------------

class TestMemorySearch:
    """Test search() with a mocked mem0 instance."""

    def _make_store_with_mock(self):
        """Create a MemoryStore with a mocked mem0 Memory."""
        store = MemoryStore(_make_config(memory_enabled=False))
        # Force-enable by injecting a mock
        store._mem = MagicMock()
        store._initialized = True
        store._vector_count = 10  # non-zero so search isn't short-circuited
        return store

    def test_search_returns_memories(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {
            "results": [
                {"memory": "User's name is Henry", "score": 0.95},
                {"memory": "User works on autocompleter project", "score": 0.8},
            ]
        }
        results = store.search("who am I")
        assert len(results) == 2
        assert results[0] == "User's name is Henry"
        assert results[1] == "User works on autocompleter project"
        store._mem.search.assert_called_once_with(
            query="who am I",
            user_id=_DEFAULT_USER_ID,
            limit=5,
        )

    def test_search_empty_query_returns_empty(self):
        store = self._make_store_with_mock()
        results = store.search("   ")
        assert results == []
        store._mem.search.assert_not_called()

    def test_search_handles_exception(self):
        store = self._make_store_with_mock()
        store._mem.search.side_effect = RuntimeError("network error")
        results = store.search("test query")
        assert results == []

    def test_search_respects_limit(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {"results": []}
        store.search("test", limit=3)
        store._mem.search.assert_called_once_with(
            query="test",
            user_id=_DEFAULT_USER_ID,
            limit=3,
        )

    def test_search_custom_user_id(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {"results": []}
        store.search("test", user_id="custom_user")
        store._mem.search.assert_called_once_with(
            query="test",
            user_id="custom_user",
            limit=5,
        )

    def test_search_filters_blank_memories(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {
            "results": [
                {"memory": "Valid memory", "score": 0.9},
                {"memory": "  ", "score": 0.5},
                {"memory": "", "score": 0.3},
            ]
        }
        results = store.search("test")
        assert results == ["Valid memory"]


class TestMemoryAdd:
    """Test add() and add_async() with a mocked mem0 instance."""

    def _make_store_with_mock(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        store._mem = MagicMock()
        store._initialized = True
        return store

    def test_add_returns_result(self):
        store = self._make_store_with_mock()
        store._mem.add.return_value = {"id": "mem_123"}
        messages = [{"role": "user", "content": "I prefer concise responses"}]
        result = store.add(messages)
        assert result == {"id": "mem_123"}
        store._mem.add.assert_called_once_with(
            messages,
            user_id=_DEFAULT_USER_ID,
            metadata=None,
        )

    def test_add_with_metadata(self):
        store = self._make_store_with_mock()
        store._mem.add.return_value = {}
        messages = [{"role": "user", "content": "test"}]
        store.add(messages, metadata={"app": "Slack"})
        store._mem.add.assert_called_once_with(
            messages,
            user_id=_DEFAULT_USER_ID,
            metadata={"app": "Slack"},
        )

    def test_add_handles_exception(self):
        store = self._make_store_with_mock()
        store._mem.add.side_effect = RuntimeError("API error")
        result = store.add([{"role": "user", "content": "test"}])
        assert result is None

    def test_add_when_disabled_returns_none(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        result = store.add([{"role": "user", "content": "test"}])
        assert result is None

    def test_add_async_runs_on_background_thread(self):
        store = self._make_store_with_mock()
        event = threading.Event()

        def signal_add(*args, **kwargs):
            event.set()
            return {"id": "mem_async"}

        store._mem.add.side_effect = signal_add
        store.add_async([{"role": "user", "content": "test"}])
        # Should complete within 2 seconds
        assert event.wait(timeout=2.0), "add_async did not run"
        store._mem.add.assert_called_once()

    def test_add_async_when_disabled_is_noop(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        # Should not raise
        store.add_async([{"role": "user", "content": "test"}])


class TestMemoryGetAll:
    """Test get_all() with a mocked mem0 instance."""

    def _make_store_with_mock(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        store._mem = MagicMock()
        store._initialized = True
        return store

    def test_get_all_returns_memories(self):
        store = self._make_store_with_mock()
        store._mem.get_all.return_value = {
            "results": [
                {"memory": "Memory one"},
                {"memory": "Memory two"},
            ]
        }
        results = store.get_all()
        assert len(results) == 2
        assert results[0] == "Memory one"

    def test_get_all_when_disabled(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        assert store.get_all() == []

    def test_get_all_handles_exception(self):
        store = self._make_store_with_mock()
        store._mem.get_all.side_effect = RuntimeError("error")
        assert store.get_all() == []


class TestFormatForContext:
    """Test format_for_context() output formatting."""

    def test_formats_memories_as_bullet_list(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        memories = ["User is named Henry", "User works on macOS autocompleter"]
        result = store.format_for_context(memories)
        assert result.startswith("User memories:\n")
        assert "- User is named Henry" in result
        assert "- User works on macOS autocompleter" in result

    def test_empty_memories_returns_empty_string(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        assert store.format_for_context([]) == ""

    def test_respects_max_chars(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        memories = ["A" * 300, "B" * 300, "C" * 300]
        result = store.format_for_context(memories, max_chars=600)
        assert "- " + "A" * 300 in result
        assert "- " + "B" * 300 in result
        # Third one should be truncated by budget
        assert "C" * 300 not in result


class TestContextStoreIntegration:
    """Test that context_store properly includes memory_context."""

    def test_continuation_context_includes_memory(self):
        from autocompleter.context_store import ContextStore
        import tempfile
        db_path = Path(tempfile.mkdtemp()) / "test.db"
        cs = ContextStore(db_path)
        cs.open()
        try:
            context = cs.get_continuation_context(
                before_cursor="Hello ",
                after_cursor="",
                source_app="TestApp",
                memory_context="User memories:\n- User prefers formal tone",
            )
            assert "User memories:" in context
            assert "User prefers formal tone" in context
        finally:
            cs.close()

    def test_continuation_context_without_memory(self):
        from autocompleter.context_store import ContextStore
        import tempfile
        db_path = Path(tempfile.mkdtemp()) / "test.db"
        cs = ContextStore(db_path)
        cs.open()
        try:
            context = cs.get_continuation_context(
                before_cursor="Hello ",
                after_cursor="",
                source_app="TestApp",
                memory_context="",
            )
            assert "User memories:" not in context
        finally:
            cs.close()

    def test_reply_context_includes_memory(self):
        from autocompleter.context_store import ContextStore
        import tempfile
        db_path = Path(tempfile.mkdtemp()) / "test.db"
        cs = ContextStore(db_path)
        cs.open()
        try:
            context = cs.get_reply_context(
                conversation_turns=[
                    {"speaker": "Alice", "text": "How's the project?"},
                ],
                source_app="TestApp",
                memory_context="User memories:\n- User's project is called Autocompleter",
            )
            assert "User memories:" in context
            assert "User's project is called Autocompleter" in context
            # Conversation should also be present
            assert "Alice" in context
        finally:
            cs.close()

    def test_reply_context_without_memory(self):
        from autocompleter.context_store import ContextStore
        import tempfile
        db_path = Path(tempfile.mkdtemp()) / "test.db"
        cs = ContextStore(db_path)
        cs.open()
        try:
            context = cs.get_reply_context(
                conversation_turns=[
                    {"speaker": "Alice", "text": "Hello"},
                ],
                source_app="TestApp",
            )
            assert "User memories:" not in context
        finally:
            cs.close()


# ---------------------------------------------------------------------------
# Build query tests
# ---------------------------------------------------------------------------

class TestBuildQuery:
    """Test MemoryStore.build_query() composite query construction."""

    def test_all_signals(self):
        q = MemoryStore.build_query(
            app_name="Discord",
            window_title="@Bankim - Discord",
            visible_snippet="hey, how's the project going?",
        )
        assert "Discord" in q
        assert "@Bankim" in q
        assert "project going?" in q

    def test_app_and_title_only(self):
        q = MemoryStore.build_query(app_name="Messages", window_title="Alice")
        assert "Messages" in q
        assert "Alice" in q

    def test_empty_inputs(self):
        q = MemoryStore.build_query()
        assert q == ""

    def test_snippet_truncated(self):
        long_text = "x" * 500
        q = MemoryStore.build_query(visible_snippet=long_text)
        # Last 200 chars
        assert len(q) == 200


# ---------------------------------------------------------------------------
# Short-circuit tests
# ---------------------------------------------------------------------------

class TestShortCircuitEmptyIndex:
    """Test that search() skips the API call when vector count is 0."""

    def test_search_skipped_when_zero_vectors(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        store._mem = MagicMock()
        store._initialized = True
        store._vector_count = 0
        results = store.search("test query")
        assert results == []
        # mem0.search should NOT have been called
        store._mem.search.assert_not_called()

    def test_search_proceeds_when_vectors_exist(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        store._mem = MagicMock()
        store._initialized = True
        store._vector_count = 5
        store._mem.search.return_value = {"results": []}
        store.search("test query")
        # mem0.search SHOULD have been called
        store._mem.search.assert_called_once()


# ---------------------------------------------------------------------------
# Pre-warm / cache tests
# ---------------------------------------------------------------------------

class TestPreWarmCache:
    """Test the pre-warm caching layer."""

    def _make_store_with_mock(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        store._mem = MagicMock()
        store._initialized = True
        store._vector_count = 10  # non-zero so search proceeds
        return store

    def test_get_cached_context_empty_initially(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        assert store.get_cached_context() == ""
        assert store.get_cached_results() == []

    def test_pre_warm_populates_cache(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {
            "results": [
                {"memory": "User likes Python", "score": 0.9},
            ]
        }
        store.pre_warm("Discord | @Bankim | hey")
        # Wait for background thread
        store._executor.shutdown(wait=True)

        cached = store.get_cached_context()
        assert "User likes Python" in cached
        assert store.get_cached_results() == ["User likes Python"]

    def test_pre_warm_noop_for_same_query(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {"results": []}

        store.pre_warm("same query")
        store._executor.shutdown(wait=True)

        # Reset mock to track second call
        store._mem.search.reset_mock()

        store.pre_warm("same query")
        store._executor.shutdown(wait=True)

        # Should NOT have searched again — query didn't change
        store._mem.search.assert_not_called()

    def test_pre_warm_refreshes_on_new_query(self):
        store = self._make_store_with_mock()
        store._mem.search.return_value = {
            "results": [{"memory": "Memory A", "score": 0.9}]
        }

        store.pre_warm("query 1")
        store._executor.shutdown(wait=True)
        assert "Memory A" in store.get_cached_context()

        # Recreate executor after shutdown so we can submit again.
        from concurrent.futures import ThreadPoolExecutor
        store._executor = ThreadPoolExecutor(max_workers=2)

        store._mem.search.return_value = {
            "results": [{"memory": "Memory B", "score": 0.8}]
        }
        store.pre_warm("query 2")
        store._executor.shutdown(wait=True)
        assert "Memory B" in store.get_cached_context()
        assert "Memory A" not in store.get_cached_context()

    def test_pre_warm_when_disabled_is_noop(self):
        store = MemoryStore(_make_config(memory_enabled=False))
        store.pre_warm("anything")  # should not raise

    def test_add_increments_vector_count(self):
        store = self._make_store_with_mock()
        store._vector_count = 0
        store._mem.add.return_value = {"id": "mem_123"}
        store.add([{"role": "user", "content": "test"}])
        assert store._vector_count == 1
