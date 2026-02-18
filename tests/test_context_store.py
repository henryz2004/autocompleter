"""Tests for the context store."""

import tempfile
import time
from pathlib import Path

import pytest

from autocompleter.context_store import ContextStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_context.db"
    s = ContextStore(db_path)
    s.open()
    yield s
    s.close()


class TestContextStore:
    def test_add_and_retrieve(self, store):
        entry_id = store.add_entry(
            source_app="Safari",
            content="Hello world",
            entry_type="visible_text",
            source_url="https://example.com",
        )
        assert entry_id > 0

        entries = store.get_recent(limit=10)
        assert len(entries) == 1
        assert entries[0].content == "Hello world"
        assert entries[0].source_app == "Safari"
        assert entries[0].source_url == "https://example.com"
        assert entries[0].entry_type == "visible_text"

    def test_deduplication(self, store):
        store.add_entry("Safari", "Same content", "visible_text")
        store.add_entry("Safari", "Same content", "visible_text")

        entries = store.get_recent(limit=10)
        assert len(entries) == 1

    def test_deduplication_allows_different_apps(self, store):
        store.add_entry("Safari", "Same content", "visible_text")
        store.add_entry("Chrome", "Same content", "visible_text")

        entries = store.get_recent(limit=10)
        assert len(entries) == 2

    def test_get_by_source(self, store):
        store.add_entry("Safari", "Safari content", "visible_text")
        store.add_entry("Chrome", "Chrome content", "visible_text")

        safari_entries = store.get_by_source("Safari")
        assert len(safari_entries) == 1
        assert safari_entries[0].source_app == "Safari"

    def test_search(self, store):
        store.add_entry("Safari", "Python is great", "visible_text")
        store.add_entry("Safari", "JavaScript rocks", "visible_text")

        results = store.search("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_get_sliced_context(self, store):
        store.add_entry("Safari", "First message from Safari", "visible_text")
        store.add_entry("Chrome", "Message from Chrome", "visible_text")

        context = store.get_sliced_context("Safari", max_chars=4000)
        assert "First message from Safari" in context
        assert "Chrome" in context  # Cross-app context included

    def test_get_sliced_context_respects_max_chars(self, store):
        # Add entries that exceed max_chars
        for i in range(50):
            store.add_entry(
                "Safari",
                f"Entry number {i} with some padding text to fill space " * 3,
                "visible_text",
                timestamp=time.time() + i * 10,  # Avoid dedup
            )

        context = store.get_sliced_context("Safari", max_chars=200)
        assert len(context) <= 400  # Some slack for the last entry

    def test_prune_by_age(self, store):
        old_time = time.time() - 100 * 3600  # 100 hours ago
        store.add_entry("Safari", "Old entry", "visible_text", timestamp=old_time)
        store.add_entry("Safari", "New entry", "visible_text")

        removed = store.prune(max_age_hours=72)
        assert removed == 1

        entries = store.get_recent()
        assert len(entries) == 1
        assert entries[0].content == "New entry"

    def test_prune_by_count(self, store):
        for i in range(10):
            store.add_entry(
                "Safari",
                f"Entry {i}",
                "visible_text",
                timestamp=time.time() + i * 10,
            )

        removed = store.prune(max_age_hours=9999, max_entries=5)
        assert removed == 5
        assert store.entry_count() == 5

    def test_entry_count(self, store):
        assert store.entry_count() == 0
        store.add_entry("Safari", "One", "visible_text")
        assert store.entry_count() == 1

    def test_open_close_reopen(self, tmp_path):
        db_path = tmp_path / "persist.db"
        store = ContextStore(db_path)
        store.open()
        store.add_entry("Safari", "Persisted", "visible_text")
        store.close()

        store2 = ContextStore(db_path)
        store2.open()
        entries = store2.get_recent()
        assert len(entries) == 1
        assert entries[0].content == "Persisted"
        store2.close()

    def test_not_opened_raises(self):
        store = ContextStore(Path("/tmp/unused.db"))
        with pytest.raises(RuntimeError, match="not open"):
            store.get_recent()
