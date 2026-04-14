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
    # ---- Continuation context tests ----

    def test_continuation_context_includes_cursor_state(self, store):
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

    def test_continuation_context_includes_subtree(self, store):
        subtree = "<StaticText>Recent chat about project deadline</StaticText>"
        context = store.get_continuation_context(
            before_cursor="The deadline is ",
            after_cursor="",
            source_app="Slack",
            subtree_context=subtree,
        )
        assert "Nearby content:" in context
        assert "project deadline" in context

    def test_continuation_context_omits_empty_after_cursor(self, store):
        context = store.get_continuation_context(
            before_cursor="Hello",
            after_cursor="",
            source_app="TextEdit",
        )
        assert "Text after cursor:" not in context

    def test_continuation_context_includes_cross_app(self, store):
        context = store.get_continuation_context(
            before_cursor="test",
            after_cursor="",
            source_app="Slack",
            cross_app_context="Recent context from Safari: looking at docs",
        )
        assert "Recent context from Safari" in context

    def test_continuation_context_includes_memory(self, store):
        context = store.get_continuation_context(
            before_cursor="test",
            after_cursor="",
            source_app="Slack",
            memory_context="User prefers concise replies",
        )
        assert "User prefers concise replies" in context

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

    def test_reply_context_falls_back_to_subtree(self, store):
        subtree_xml = "<StaticText>Welcome to the app</StaticText>"
        context = store.get_reply_context(
            conversation_turns=[],
            source_app="UnknownApp",
            subtree_context=subtree_xml,
        )
        assert "Nearby content:" in context
        assert subtree_xml in context
        assert "Conversation:" not in context

    def test_reply_context_no_turns_no_subtree(self, store):
        context = store.get_reply_context(
            conversation_turns=[],
            source_app="Slack",
        )
        assert "Conversation:" not in context
        assert "Nearby content:" not in context

    def test_reply_context_prioritizes_turns_over_subtree(self, store):
        """When both conversation_turns and subtree_context are provided,
        conversation_turns (with speaker labels) should take priority.
        """
        turns = [
            {"speaker": "Claude", "text": "What would you like help with?"},
            {"speaker": "User", "text": "I need to debug my code"},
        ]
        subtree_xml = "<StaticText>What would you like help with?</StaticText>"
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Claude",
            subtree_context=subtree_xml,
        )
        assert "Conversation:" in context
        assert "- Claude: What would you like help with?" in context
        assert "- User: I need to debug my code" in context
        assert "Nearby content:" not in context

    def test_reply_context_one_sided_supplements_with_subtree(self, store):
        """One-sided conversation should supplement with subtree context."""
        turns = [
            {"speaker": "You", "text": "What's the weather?"},
            {"speaker": "You", "text": "Also check the news"},
        ]
        subtree = "<StaticText>The weather is sunny today</StaticText>"
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Gemini",
            subtree_context=subtree,
        )
        assert "Conversation:" in context
        assert "Additional visible content" in context
        assert subtree in context

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

    def test_reply_context_includes_timestamps(self):
        """get_reply_context should format timestamps in turn lines."""
        store = ContextStore(Path("/tmp/test_ts.db"))
        store.open()
        try:
            turns = [
                {"speaker": "Alice", "text": "hello", "timestamp": "10:05 PM"},
                {"speaker": "Bob", "text": "hi there"},  # no timestamp
                {"speaker": "Alice", "text": "how are you", "timestamp": "Yesterday at 8:47 PM"},
            ]
            ctx = store.get_reply_context(
                conversation_turns=turns,
                source_app="Discord",
            )
            assert "Alice [10:05 PM]: hello" in ctx
            assert "- Bob: hi there" in ctx  # no bracket for missing timestamp
            assert "Alice [Yesterday at 8:47 PM]: how are you" in ctx
        finally:
            store.close()

    # ---- Lifecycle tests ----

    def test_open_close_reopen(self, tmp_path):
        db_path = tmp_path / "persist.db"
        store = ContextStore(db_path)
        store.open()
        store.record_feedback("Safari", "continuation", "test", "accepted")
        store.close()

        store2 = ContextStore(db_path)
        store2.open()
        stats = store2.get_feedback_stats(source_app="Safari")
        assert stats["total_accepted"] == 1
        store2.close()

    def test_not_opened_raises(self):
        store = ContextStore(Path("/tmp/unused.db"))
        with pytest.raises(RuntimeError, match="not open"):
            store.get_feedback_stats()
