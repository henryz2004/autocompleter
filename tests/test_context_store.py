"""Tests for the context store."""

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

    def test_add_with_window_title(self, store):
        entry_id = store.add_entry(
            source_app="Slack",
            content="Message text",
            entry_type="visible_text",
            window_title="#general - Slack",
        )
        assert entry_id > 0
        entries = store.get_recent(limit=1)
        assert entries[0].window_title == "#general - Slack"

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

    # ---- Continuation context tests ----

    def test_continuation_context_includes_cursor_state(self, store):
        store.add_entry("Slack", "Some visible text", "visible_text")
        context = store.get_continuation_context(
            before_cursor="Hello, I wanted to ",
            after_cursor="you about it.",
            source_app="Slack",
        )
        assert "Hello, I wanted to " in context
        assert "you about it." in context
        assert "Text before cursor:" in context
        assert "Text after cursor:" in context

    def test_continuation_context_includes_metadata(self, store):
        context = store.get_continuation_context(
            before_cursor="test",
            after_cursor="",
            source_app="Slack",
            window_title="#general",
            source_url="https://slack.com",
        )
        assert "App: Slack" in context
        assert "Window: #general" in context
        assert "URL: https://slack.com" in context

    def test_continuation_context_includes_local_context(self, store):
        store.add_entry("Slack", "Recent chat about project deadline", "visible_text")
        context = store.get_continuation_context(
            before_cursor="The deadline is ",
            after_cursor="",
            source_app="Slack",
        )
        assert "Recent context:" in context
        assert "project deadline" in context

    def test_continuation_context_skips_user_input_entries(self, store):
        store.add_entry("Slack", "visible stuff", "visible_text")
        store.add_entry("Slack", "user typed something", "user_input")
        context = store.get_continuation_context(
            before_cursor="test",
            after_cursor="",
            source_app="Slack",
        )
        assert "user typed something" not in context

    def test_continuation_context_omits_empty_after_cursor(self, store):
        context = store.get_continuation_context(
            before_cursor="Hello",
            after_cursor="",
            source_app="TextEdit",
        )
        assert "Text after cursor:" not in context

    # ---- Reply context tests ----

    def test_reply_context_with_structured_turns(self, store):
        turns = [
            {"speaker": "Alice", "text": "Hey, how's the PR?"},
            {"speaker": "Bob", "text": "Almost done, fixing tests"},
        ]
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Slack",
            window_title="#dev",
        )
        assert "- Alice: Hey, how's the PR?" in context
        assert "- Bob: Almost done, fixing tests" in context
        assert "Conversation:" in context
        assert "Channel: #dev" in context

    def test_reply_context_with_draft(self, store):
        turns = [{"speaker": "Alice", "text": "Hello"}]
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Slack",
            draft_text="Thanks for",
        )
        assert "Draft so far:" in context
        assert "Thanks for" in context

    def test_reply_context_falls_back_to_flat_text(self, store):
        store.add_entry("Slack", "Some visible conversation text", "visible_text")
        context = store.get_reply_context(
            conversation_turns=[],
            source_app="Slack",
        )
        assert "Recent visible text:" in context
        assert "Some visible conversation text" in context

    def test_reply_context_limits_turns(self, store):
        turns = [
            {"speaker": f"User{i}", "text": f"Message {i}"}
            for i in range(20)
        ]
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Slack",
            max_turns=3,
        )
        # Should only have the last 3 turns
        assert "User17:" in context
        assert "User18:" in context
        assert "User19:" in context
        assert "User0:" not in context

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
