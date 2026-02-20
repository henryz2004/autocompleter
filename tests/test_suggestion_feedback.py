"""Tests for the suggestion feedback loop.

Covers feedback recording, statistics, dismissed patterns,
temperature adjustment, and system prompt integration.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from autocompleter.config import Config
from autocompleter.context_store import ContextStore
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    Suggestion,
    SuggestionEngine,
    adjust_temperature,
)


# ---- Fixtures ----


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_feedback.db"
    s = ContextStore(db_path)
    s.open()
    yield s
    s.close()


@pytest.fixture
def config():
    return Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        num_suggestions=3,
        debounce_ms=100,
        max_tokens=150,
    )


@pytest.fixture
def engine(config):
    return SuggestionEngine(config)


# ---- Schema migration tests ----


class TestSchemaMigration:
    def test_feedback_table_created_on_open(self, tmp_path):
        """Opening the DB for the first time should create the suggestion_feedback table."""
        db_path = tmp_path / "fresh.db"
        store = ContextStore(db_path)
        store.open()
        # Verify the table exists by inserting directly
        conn = store._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='suggestion_feedback'"
        )
        assert cursor.fetchone() is not None
        store.close()

    def test_feedback_table_survives_reopen(self, tmp_path):
        """Table should persist across close/reopen cycles."""
        db_path = tmp_path / "persist.db"
        store = ContextStore(db_path)
        store.open()
        store.record_feedback(
            source_app="Safari",
            mode="continuation",
            suggestion_text="test",
            action="accepted",
        )
        store.close()

        store2 = ContextStore(db_path)
        store2.open()
        stats = store2.get_feedback_stats()
        assert stats["total_accepted"] == 1
        store2.close()

    def test_existing_db_gets_feedback_table(self, tmp_path):
        """An existing DB without the feedback table should get it on open."""
        import sqlite3

        db_path = tmp_path / "old.db"
        # Create a DB with only the context_entries table
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE context_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_app TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                entry_type TEXT NOT NULL,
                window_title TEXT DEFAULT ''
            )"""
        )
        conn.commit()
        conn.close()

        store = ContextStore(db_path)
        store.open()
        # Should be able to record feedback now
        fid = store.record_feedback(
            source_app="Safari",
            mode="reply",
            suggestion_text="hello",
            action="dismissed",
        )
        assert fid > 0
        store.close()


# ---- record_feedback tests ----


class TestRecordFeedback:
    def test_stores_feedback_correctly(self, store):
        fid = store.record_feedback(
            source_app="Slack",
            mode="continuation",
            suggestion_text="Hello there",
            action="accepted",
            suggestion_index=1,
            total_suggestions=3,
            latency_ms=1500.5,
        )
        assert fid > 0

        # Verify by reading directly
        conn = store._get_conn()
        cursor = conn.execute(
            "SELECT source_app, mode, suggestion_text, action, "
            "suggestion_index, total_suggestions, latency_ms "
            "FROM suggestion_feedback WHERE id = ?",
            (fid,),
        )
        row = cursor.fetchone()
        assert row[0] == "Slack"
        assert row[1] == "continuation"
        assert row[2] == "Hello there"
        assert row[3] == "accepted"
        assert row[4] == 1
        assert row[5] == 3
        assert abs(row[6] - 1500.5) < 0.01

    def test_stores_with_none_optionals(self, store):
        fid = store.record_feedback(
            source_app="Safari",
            mode="reply",
            suggestion_text="test text",
            action="dismissed",
        )
        assert fid > 0

        conn = store._get_conn()
        cursor = conn.execute(
            "SELECT suggestion_index, total_suggestions, latency_ms "
            "FROM suggestion_feedback WHERE id = ?",
            (fid,),
        )
        row = cursor.fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

    def test_stores_multiple_feedback_entries(self, store):
        for i in range(5):
            store.record_feedback(
                source_app="Slack",
                mode="continuation",
                suggestion_text=f"suggestion {i}",
                action="accepted" if i % 2 == 0 else "dismissed",
                suggestion_index=i,
                total_suggestions=5,
            )

        conn = store._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM suggestion_feedback")
        assert cursor.fetchone()[0] == 5

    def test_records_regenerated_action(self, store):
        fid = store.record_feedback(
            source_app="VSCode",
            mode="continuation",
            suggestion_text="old suggestion",
            action="regenerated",
        )
        assert fid > 0

    def test_feedback_records_timestamp(self, store):
        before = time.time()
        store.record_feedback(
            source_app="Slack",
            mode="reply",
            suggestion_text="test",
            action="accepted",
        )
        after = time.time()

        conn = store._get_conn()
        cursor = conn.execute("SELECT timestamp FROM suggestion_feedback")
        ts = cursor.fetchone()[0]
        assert before <= ts <= after

    def test_latency_calculation(self, store):
        """Feedback with a known latency should store it correctly."""
        store.record_feedback(
            source_app="Slack",
            mode="continuation",
            suggestion_text="test",
            action="accepted",
            latency_ms=2345.67,
        )
        stats = store.get_feedback_stats()
        assert abs(stats["avg_latency_ms"] - 2345.67) < 0.01


# ---- get_feedback_stats tests ----


class TestGetFeedbackStats:
    def test_empty_stats(self, store):
        stats = store.get_feedback_stats()
        assert stats["total_shown"] == 0
        assert stats["total_accepted"] == 0
        assert stats["total_dismissed"] == 0
        assert stats["accept_rate"] == 0.0
        assert stats["avg_accepted_index"] == 0.0
        assert stats["avg_latency_ms"] == 0.0

    def test_all_accepted(self, store):
        for i in range(5):
            store.record_feedback(
                source_app="Slack",
                mode="continuation",
                suggestion_text=f"s{i}",
                action="accepted",
                suggestion_index=i,
                total_suggestions=5,
                latency_ms=1000.0 + i * 100,
            )
        stats = store.get_feedback_stats()
        assert stats["total_shown"] == 5
        assert stats["total_accepted"] == 5
        assert stats["total_dismissed"] == 0
        assert stats["accept_rate"] == 1.0
        assert stats["avg_accepted_index"] == 2.0  # avg of 0,1,2,3,4
        assert abs(stats["avg_latency_ms"] - 1200.0) < 0.01  # avg of 1000..1400

    def test_all_dismissed(self, store):
        for i in range(3):
            store.record_feedback(
                source_app="Slack",
                mode="reply",
                suggestion_text=f"s{i}",
                action="dismissed",
                suggestion_index=i,
                total_suggestions=3,
            )
        stats = store.get_feedback_stats()
        assert stats["total_shown"] == 3
        assert stats["total_accepted"] == 0
        assert stats["total_dismissed"] == 3
        assert stats["accept_rate"] == 0.0

    def test_mixed_actions(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s1", action="accepted",
            suggestion_index=0, total_suggestions=3,
        )
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s2", action="dismissed",
            suggestion_index=1, total_suggestions=3,
        )
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s3", action="dismissed",
            suggestion_index=2, total_suggestions=3,
        )
        stats = store.get_feedback_stats()
        assert stats["total_shown"] == 3
        assert stats["total_accepted"] == 1
        assert stats["total_dismissed"] == 2
        assert abs(stats["accept_rate"] - 1 / 3) < 0.01

    def test_filtered_by_source_app(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s1", action="accepted",
        )
        store.record_feedback(
            source_app="Safari", mode="continuation",
            suggestion_text="s2", action="dismissed",
        )
        store.record_feedback(
            source_app="Slack", mode="reply",
            suggestion_text="s3", action="dismissed",
        )

        slack_stats = store.get_feedback_stats(source_app="Slack")
        assert slack_stats["total_shown"] == 2
        assert slack_stats["total_accepted"] == 1
        assert slack_stats["total_dismissed"] == 1

        safari_stats = store.get_feedback_stats(source_app="Safari")
        assert safari_stats["total_shown"] == 1
        assert safari_stats["total_accepted"] == 0
        assert safari_stats["total_dismissed"] == 1

    def test_filtered_by_mode(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s1", action="accepted",
        )
        store.record_feedback(
            source_app="Slack", mode="reply",
            suggestion_text="s2", action="dismissed",
        )

        cont_stats = store.get_feedback_stats(mode="continuation")
        assert cont_stats["total_shown"] == 1
        assert cont_stats["total_accepted"] == 1

        reply_stats = store.get_feedback_stats(mode="reply")
        assert reply_stats["total_shown"] == 1
        assert reply_stats["total_dismissed"] == 1

    def test_filtered_by_app_and_mode(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s1", action="accepted",
        )
        store.record_feedback(
            source_app="Slack", mode="reply",
            suggestion_text="s2", action="dismissed",
        )
        store.record_feedback(
            source_app="Safari", mode="continuation",
            suggestion_text="s3", action="accepted",
        )

        stats = store.get_feedback_stats(source_app="Slack", mode="continuation")
        assert stats["total_shown"] == 1
        assert stats["total_accepted"] == 1

    def test_respects_hours_window(self, store):
        # Record old feedback (beyond the window)
        conn = store._get_conn()
        old_time = time.time() - 48 * 3600  # 48 hours ago
        conn.execute(
            """INSERT INTO suggestion_feedback
               (timestamp, source_app, mode, suggestion_text, action,
                suggestion_index, total_suggestions, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (old_time, "Slack", "continuation", "old suggestion", "accepted", 0, 3, 500.0),
        )
        conn.commit()

        # Record recent feedback
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="recent", action="dismissed",
            suggestion_index=1, total_suggestions=3,
        )

        # With 24-hour window, should only see the recent entry
        stats = store.get_feedback_stats(hours=24)
        assert stats["total_shown"] == 1
        assert stats["total_dismissed"] == 1
        assert stats["total_accepted"] == 0

        # With 72-hour window, should see both
        stats = store.get_feedback_stats(hours=72)
        assert stats["total_shown"] == 2
        assert stats["total_accepted"] == 1
        assert stats["total_dismissed"] == 1

    def test_avg_accepted_index(self, store):
        """avg_accepted_index should average only accepted entries."""
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s1", action="accepted",
            suggestion_index=0, total_suggestions=3,
        )
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s2", action="dismissed",
            suggestion_index=1, total_suggestions=3,
        )
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s3", action="accepted",
            suggestion_index=2, total_suggestions=3,
        )
        stats = store.get_feedback_stats()
        # avg of index 0 and 2 = 1.0
        assert stats["avg_accepted_index"] == 1.0

    def test_avg_latency_ms_all_entries(self, store):
        """avg_latency_ms should average across all entries with latency."""
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s1", action="accepted",
            latency_ms=1000.0,
        )
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="s2", action="dismissed",
            latency_ms=2000.0,
        )
        stats = store.get_feedback_stats()
        assert abs(stats["avg_latency_ms"] - 1500.0) < 0.01


# ---- get_recent_dismissed_patterns tests ----


class TestGetRecentDismissedPatterns:
    def test_empty_returns_empty(self, store):
        assert store.get_recent_dismissed_patterns() == []

    def test_returns_dismissed_only(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="accepted text", action="accepted",
        )
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="dismissed text", action="dismissed",
        )
        patterns = store.get_recent_dismissed_patterns()
        assert patterns == ["dismissed text"]

    def test_returns_in_recency_order(self, store):
        for i in range(5):
            store.record_feedback(
                source_app="Slack", mode="continuation",
                suggestion_text=f"pattern {i}", action="dismissed",
            )
        patterns = store.get_recent_dismissed_patterns()
        # Most recent first
        assert patterns[0] == "pattern 4"
        assert patterns[-1] == "pattern 0"

    def test_respects_limit(self, store):
        for i in range(10):
            store.record_feedback(
                source_app="Slack", mode="continuation",
                suggestion_text=f"pattern {i}", action="dismissed",
            )
        patterns = store.get_recent_dismissed_patterns(limit=3)
        assert len(patterns) == 3
        # Most recent 3
        assert patterns == ["pattern 9", "pattern 8", "pattern 7"]

    def test_filtered_by_source_app(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="slack pattern", action="dismissed",
        )
        store.record_feedback(
            source_app="Safari", mode="continuation",
            suggestion_text="safari pattern", action="dismissed",
        )

        slack_patterns = store.get_recent_dismissed_patterns(source_app="Slack")
        assert slack_patterns == ["slack pattern"]

        safari_patterns = store.get_recent_dismissed_patterns(source_app="Safari")
        assert safari_patterns == ["safari pattern"]

    def test_no_app_filter_returns_all(self, store):
        store.record_feedback(
            source_app="Slack", mode="continuation",
            suggestion_text="from slack", action="dismissed",
        )
        store.record_feedback(
            source_app="Safari", mode="continuation",
            suggestion_text="from safari", action="dismissed",
        )
        patterns = store.get_recent_dismissed_patterns()
        assert len(patterns) == 2


# ---- adjust_temperature tests ----


class TestAdjustTemperature:
    def test_low_accept_rate_lowers_temp(self):
        # accept_rate < 0.3 -> lower by 0.1
        result = adjust_temperature(0.5, 0.2)
        assert abs(result - 0.4) < 0.001

    def test_high_accept_rate_raises_temp(self):
        # accept_rate > 0.7 -> raise by 0.05
        result = adjust_temperature(0.5, 0.8)
        assert abs(result - 0.55) < 0.001

    def test_mid_accept_rate_unchanged(self):
        # 0.3 <= accept_rate <= 0.7 -> no change
        result = adjust_temperature(0.5, 0.5)
        assert abs(result - 0.5) < 0.001

    def test_boundary_0_3_unchanged(self):
        # Exactly 0.3 should NOT lower
        result = adjust_temperature(0.5, 0.3)
        assert abs(result - 0.5) < 0.001

    def test_boundary_0_7_unchanged(self):
        # Exactly 0.7 should NOT raise
        result = adjust_temperature(0.5, 0.7)
        assert abs(result - 0.5) < 0.001

    def test_clamp_lower_bound(self):
        # Even with very low base temp, should not go below 0.1
        result = adjust_temperature(0.1, 0.1)
        assert abs(result - 0.1) < 0.001

    def test_clamp_lower_bound_extreme(self):
        # 0.05 - 0.1 = -0.05, should clamp to 0.1
        result = adjust_temperature(0.05, 0.1)
        assert abs(result - 0.1) < 0.001

    def test_clamp_upper_bound(self):
        # 1.0 + 0.05 = 1.05, should clamp to 1.0
        result = adjust_temperature(1.0, 0.9)
        assert abs(result - 1.0) < 0.001

    def test_clamp_upper_bound_extreme(self):
        result = adjust_temperature(0.98, 0.8)
        assert abs(result - 1.0) < 0.001

    def test_zero_accept_rate(self):
        result = adjust_temperature(0.5, 0.0)
        assert abs(result - 0.4) < 0.001

    def test_full_accept_rate(self):
        result = adjust_temperature(0.5, 1.0)
        assert abs(result - 0.55) < 0.001


# ---- System prompt negative patterns integration tests ----


class TestNegativePatternsInPrompt:
    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_negative_patterns_appended_to_system_prompt(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="new suggestion", index=0)]

        engine.generate_suggestions(
            current_input="Hello world test",
            context="some context",
            mode=AutocompleteMode.CONTINUATION,
            negative_patterns=["bad suggestion 1", "bad suggestion 2"],
        )

        mock_call.assert_called_once()
        system_prompt = mock_call.call_args[0][0]
        assert "Avoid generating similar completions" in system_prompt
        assert "- bad suggestion 1" in system_prompt
        assert "- bad suggestion 2" in system_prompt

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_no_negative_patterns_no_change(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="suggestion", index=0)]

        engine.generate_suggestions(
            current_input="Hello world test",
            context="some context",
            mode=AutocompleteMode.CONTINUATION,
            negative_patterns=None,
        )

        mock_call.assert_called_once()
        system_prompt = mock_call.call_args[0][0]
        assert "Avoid generating similar completions" not in system_prompt

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_empty_negative_patterns_no_change(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="suggestion", index=0)]

        engine.generate_suggestions(
            current_input="Hello world test",
            context="some context",
            mode=AutocompleteMode.CONTINUATION,
            negative_patterns=[],
        )

        mock_call.assert_called_once()
        system_prompt = mock_call.call_args[0][0]
        assert "Avoid generating similar completions" not in system_prompt

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_reply_mode_negative_patterns(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="reply", index=0)]

        engine.generate_suggestions(
            current_input="",
            context="some conversation",
            mode=AutocompleteMode.REPLY,
            negative_patterns=["unwanted reply"],
        )

        system_prompt = mock_call.call_args[0][0]
        assert "- unwanted reply" in system_prompt


# ---- Feedback stats temperature integration ----


class TestFeedbackStatsTemperatureIntegration:
    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_low_accept_rate_lowers_temperature(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="s", index=0)]

        feedback_stats = {"accept_rate": 0.1, "total_shown": 10}

        engine.generate_suggestions(
            current_input="Hello world test",
            context="ctx",
            mode=AutocompleteMode.CONTINUATION,
            feedback_stats=feedback_stats,
        )

        _, kwargs = mock_call.call_args
        # continuation_temperature is 0.3, lowered by 0.1 -> 0.2
        assert abs(kwargs["temperature"] - 0.2) < 0.001

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_high_accept_rate_raises_temperature(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="s", index=0)]

        feedback_stats = {"accept_rate": 0.85}

        engine.generate_suggestions(
            current_input="Hello world test",
            context="ctx",
            mode=AutocompleteMode.CONTINUATION,
            feedback_stats=feedback_stats,
        )

        _, kwargs = mock_call.call_args
        # continuation_temperature is 0.3, raised by 0.05 -> 0.35
        assert abs(kwargs["temperature"] - 0.35) < 0.001

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_no_feedback_stats_uses_base_temperature(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="s", index=0)]

        engine.generate_suggestions(
            current_input="Hello world test",
            context="ctx",
            mode=AutocompleteMode.CONTINUATION,
            feedback_stats=None,
        )

        _, kwargs = mock_call.call_args
        assert abs(kwargs["temperature"] - engine.config.continuation_temperature) < 0.001
