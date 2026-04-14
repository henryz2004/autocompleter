"""Tests for the cross-app context trail."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from autocompleter.context_trail import AppSnapshot, ContextTrail
from autocompleter.context_store import ContextStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trail():
    return ContextTrail(maxlen=10, text_summary_chars=500)


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_context.db"
    s = ContextStore(db_path)
    s.open()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# ContextTrail.record tests
# ---------------------------------------------------------------------------

class TestRecord:
    def test_single_app_no_trail_entry(self, trail):
        """Recording from a single app should not create any trail entries."""
        trail.record(app_name="Chrome", window_title="Docs")
        trail.record(app_name="Chrome", window_title="Docs")
        trail.record(app_name="Chrome", window_title="Docs")

        snapshots = trail.get_recent_cross_app_context("Chrome")
        assert len(snapshots) == 0

    def test_app_switch_creates_snapshot(self, trail):
        """Switching A -> B should create a snapshot of A."""
        trail.record(
            app_name="Chrome",
            window_title="React Docs",
            subtree_xml="<text>useState allows you to add state</text>",
        )
        trail.record(
            app_name="VS Code",
            window_title="App.tsx",
            subtree_xml="<text>import React from 'react'</text>",
        )

        snapshots = trail.get_recent_cross_app_context("VS Code")
        assert len(snapshots) == 1
        assert snapshots[0].app_name == "Chrome"
        assert snapshots[0].window_title == "React Docs"
        assert "useState" in snapshots[0].text_summary

    def test_multiple_switches(self, trail):
        """A -> B -> C -> A should create snapshots for A, B, C."""
        for app, text in [("Chrome", "Chrome text"), ("VS Code", "VS Code text"),
                          ("Slack", "Slack text"), ("Chrome", "Chrome text 2")]:
            trail.record(app_name=app, window_title="w", subtree_xml=f"<t>{text}</t>")

        snapshots = trail.get_recent_cross_app_context("Chrome")
        assert len(snapshots) == 2
        app_names = [s.app_name for s in snapshots]
        assert "VS Code" in app_names
        assert "Slack" in app_names

    def test_deduplication_consecutive_same_app_window(self, trail):
        """Non-consecutive same app+window entries are NOT deduplicated."""
        trail.record(app_name="Chrome", window_title="Docs", subtree_xml="<t>First visit</t>")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")
        trail.record(app_name="Chrome", window_title="Docs", subtree_xml="<t>Second visit</t>")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        all_snapshots = trail.get_recent_cross_app_context(
            "NONE", max_age_seconds=9999, max_entries=20,
        )
        chrome_entries = [s for s in all_snapshots if s.app_name == "Chrome"]
        assert len(chrome_entries) == 2

    def test_deduplication_truly_consecutive(self, trail):
        """Consecutive snapshots for same app+window are updated in place."""
        trail._trail.append(AppSnapshot(
            app_name="Chrome",
            window_title="Docs",
            text_summary="old text",
            timestamp=time.time() - 10,
        ))
        trail._current_app = "Chrome"
        trail._current_window = "Docs"
        trail._current_subtree = "<t>updated text</t>"
        trail._current_url = ""

        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        all_snapshots = trail.get_recent_cross_app_context(
            "NONE", max_age_seconds=9999, max_entries=20,
        )
        chrome_entries = [s for s in all_snapshots if s.app_name == "Chrome"]
        assert len(chrome_entries) == 1
        assert "updated text" in chrome_entries[0].text_summary


# ---------------------------------------------------------------------------
# get_recent_cross_app_context tests
# ---------------------------------------------------------------------------

class TestGetRecentCrossAppContext:
    def test_excludes_current_app(self, trail):
        trail.record(app_name="Chrome", window_title="w", subtree_xml="<t>chrome</t>")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        snapshots = trail.get_recent_cross_app_context("Chrome")
        for s in snapshots:
            assert s.app_name != "Chrome"

    def test_respects_max_age(self, trail):
        trail.record(app_name="Chrome", window_title="w", subtree_xml="<t>old</t>")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        if trail._trail:
            trail._trail[0] = AppSnapshot(
                app_name=trail._trail[0].app_name,
                window_title=trail._trail[0].window_title,
                text_summary=trail._trail[0].text_summary,
                timestamp=time.time() - 120,
                source_url=trail._trail[0].source_url,
            )

        snapshots = trail.get_recent_cross_app_context("VS Code", max_age_seconds=60)
        assert len(snapshots) == 0

        snapshots = trail.get_recent_cross_app_context("VS Code", max_age_seconds=300)
        assert len(snapshots) == 1

    def test_respects_max_entries(self, trail):
        apps = ["Chrome", "VS Code", "Slack", "Terminal", "Safari"]
        for app in apps:
            trail.record(app_name=app, window_title="w", subtree_xml=f"<t>text from {app}</t>")
        trail.record(app_name="Notes", window_title="w", subtree_xml="<t>notes</t>")

        snapshots = trail.get_recent_cross_app_context(
            "Notes", max_age_seconds=9999, max_entries=2,
        )
        assert len(snapshots) == 2

    def test_returns_most_recent_first(self, trail):
        for app in ["Chrome", "VS Code", "Slack", "Terminal"]:
            trail.record(app_name=app, window_title="w", subtree_xml=f"<t>{app}</t>")

        snapshots = trail.get_recent_cross_app_context(
            "Terminal", max_age_seconds=9999, max_entries=10,
        )
        assert len(snapshots) == 3
        assert snapshots[0].app_name == "Slack"
        assert snapshots[1].app_name == "VS Code"
        assert snapshots[2].app_name == "Chrome"


# ---------------------------------------------------------------------------
# format_cross_app_context tests
# ---------------------------------------------------------------------------

class TestFormatCrossAppContext:
    def test_format_produces_expected_output(self):
        snapshots = [
            AppSnapshot(
                app_name="Chrome",
                window_title="React Hooks Documentation",
                text_summary="useState allows you to add state to functional components",
                timestamp=time.time(),
            ),
            AppSnapshot(
                app_name="VS Code",
                window_title="App.tsx",
                text_summary="import { useState, useEffect } from 'react';",
                timestamp=time.time(),
            ),
        ]

        result = ContextTrail.format_cross_app_context(snapshots)
        assert "[Recent activity from other apps]" in result
        assert '- Chrome ("React Hooks Documentation"): useState allows' in result
        assert '- VS Code ("App.tsx"): import { useState' in result

    def test_format_no_window_title(self):
        snapshots = [
            AppSnapshot(
                app_name="Terminal",
                window_title="",
                text_summary="$ python script.py",
                timestamp=time.time(),
            ),
        ]
        result = ContextTrail.format_cross_app_context(snapshots)
        assert "- Terminal: $ python script.py" in result
        assert '("")' not in result

    def test_format_truncates_long_summary(self):
        long_text = "x" * 300
        snapshots = [
            AppSnapshot(
                app_name="Chrome",
                window_title="Page",
                text_summary=long_text,
                timestamp=time.time(),
            ),
        ]
        result = ContextTrail.format_cross_app_context(snapshots)
        assert "..." in result


# ---------------------------------------------------------------------------
# Trail maxlen and text_summary truncation
# ---------------------------------------------------------------------------

class TestTrailLimits:
    def test_maxlen_evicts_oldest(self):
        trail = ContextTrail(maxlen=3)
        for app in ["A", "B", "C", "D", "E", "F"]:
            trail.record(app_name=app, window_title="w", subtree_xml=f"<t>{app}</t>")

        all_snapshots = trail.get_recent_cross_app_context(
            "F", max_age_seconds=9999, max_entries=20,
        )
        assert len(all_snapshots) <= 3
        names = [s.app_name for s in all_snapshots]
        assert "A" not in names
        assert "B" not in names

    def test_text_summary_truncation(self):
        trail = ContextTrail(text_summary_chars=50)
        long_xml = "<text>" + "a" * 200 + "</text>"
        trail.record(app_name="Chrome", window_title="w", subtree_xml=long_xml)
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        snapshots = trail.get_recent_cross_app_context("VS Code", max_age_seconds=9999)
        assert len(snapshots) == 1
        assert len(snapshots[0].text_summary) == 50


# ---------------------------------------------------------------------------
# clear() test
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_trail(self, trail):
        trail.record(app_name="Chrome", window_title="w")
        trail.record(app_name="VS Code", window_title="w")
        trail.clear()

        assert trail.get_recent_cross_app_context("VS Code") == []
        assert trail._current_app == ""


# ---------------------------------------------------------------------------
# Integration with context_store
# ---------------------------------------------------------------------------

class TestContextStoreIntegration:
    def test_continuation_context_includes_cross_app(self, store):
        cross_app = (
            "[Recent activity from other apps]\n"
            '- Chrome ("React Docs"): useState allows you to add state'
        )
        context = store.get_continuation_context(
            before_cursor="const [count, setCount] = ",
            after_cursor="",
            source_app="VS Code",
            window_title="App.tsx",
            cross_app_context=cross_app,
        )
        assert "[Recent activity from other apps]" in context
        assert "useState" in context
        assert "App: VS Code" in context
        assert "const [count, setCount] =" in context

    def test_continuation_context_cross_app_position(self, store):
        cross_app = "[Recent activity from other apps]\n- Chrome: docs"
        context = store.get_continuation_context(
            before_cursor="typing here",
            after_cursor="",
            source_app="VS Code",
            window_title="file.py",
            subtree_context="<text>def hello():</text>",
            cross_app_context=cross_app,
        )
        meta_idx = context.index("App: VS Code")
        cross_idx = context.index("[Recent activity from other apps]")
        nearby_idx = context.index("Nearby content:")
        cursor_idx = context.index("Text before cursor:")

        assert meta_idx < cross_idx < nearby_idx < cursor_idx

    def test_continuation_context_empty_cross_app(self, store):
        context = store.get_continuation_context(
            before_cursor="test",
            after_cursor="",
            source_app="VS Code",
            cross_app_context="",
        )
        assert "[Recent activity from other apps]" not in context

    def test_reply_context_includes_cross_app(self, store):
        cross_app = (
            "[Recent activity from other apps]\n"
            '- VS Code ("main.py"): def process_data(input):'
        )
        context = store.get_reply_context(
            conversation_turns=[
                {"speaker": "Alice", "text": "Can you update the data pipeline?"},
            ],
            source_app="Slack",
            window_title="#engineering",
            cross_app_context=cross_app,
        )
        assert "[Recent activity from other apps]" in context
        assert "def process_data" in context
        assert "Channel: #engineering" in context
        assert "Alice:" in context

    def test_reply_context_empty_cross_app(self, store):
        context = store.get_reply_context(
            conversation_turns=[],
            source_app="Slack",
            cross_app_context="",
        )
        assert "[Recent activity from other apps]" not in context

    def test_reply_context_cross_app_position(self, store):
        cross_app = "[Recent activity from other apps]\n- Chrome: docs"
        context = store.get_reply_context(
            conversation_turns=[
                {"speaker": "Bob", "text": "Hello"},
            ],
            source_app="Slack",
            window_title="#general",
            cross_app_context=cross_app,
        )
        meta_idx = context.index("App: Slack")
        cross_idx = context.index("[Recent activity from other apps]")
        conv_idx = context.index("Conversation:")

        assert meta_idx < cross_idx < conv_idx


# ---------------------------------------------------------------------------
# End-to-end: ContextTrail -> format -> context_store
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_trail_to_context_store_pipeline(self, store):
        trail = ContextTrail()

        trail.record(
            app_name="Chrome",
            window_title="React Hooks Docs",
            subtree_xml="<text>useState hook lets you add state to function components</text>",
            url="https://react.dev/hooks",
        )
        trail.record(
            app_name="VS Code",
            window_title="App.tsx",
            subtree_xml="<text>import React from 'react'</text>",
        )

        snapshots = trail.get_recent_cross_app_context("VS Code")
        cross_app_str = ContextTrail.format_cross_app_context(snapshots)

        context = store.get_continuation_context(
            before_cursor="const [items, setItems] = ",
            after_cursor="",
            source_app="VS Code",
            window_title="App.tsx",
            cross_app_context=cross_app_str,
        )

        assert "React Hooks Docs" in context
        assert "useState" in context
        assert "const [items, setItems] =" in context


# ---------------------------------------------------------------------------
# _build_text_summary tests
# ---------------------------------------------------------------------------

class TestBuildTextSummary:
    def test_strips_xml_tags(self):
        trail = ContextTrail()
        summary = trail._build_text_summary('<text role="heading">Hello World</text>')
        assert "<" not in summary
        assert "Hello World" in summary

    def test_empty_xml(self):
        trail = ContextTrail()
        assert trail._build_text_summary("") == ""


# ---------------------------------------------------------------------------
# Noisy cross-app source filtering tests
# ---------------------------------------------------------------------------

class TestNoisyCrossAppFiltering:
    def test_skips_notification_center(self):
        trail = ContextTrail()
        trail.record(app_name="Chrome", window_title="w", subtree_xml="<t>React docs</t>")
        trail.record(app_name="Notification Center", window_title="w", subtree_xml="<t>New message</t>")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        snapshots = trail.get_recent_cross_app_context("VS Code")
        app_names = [s.app_name for s in snapshots]
        assert "Notification Center" not in app_names
        assert "Chrome" in app_names

    def test_skips_system_ui_server(self):
        trail = ContextTrail()
        trail.record(app_name="Chrome", window_title="w", subtree_xml="<t>docs</t>")
        trail.record(app_name="SystemUIServer", window_title="w", subtree_xml="<t>Battery</t>")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        snapshots = trail.get_recent_cross_app_context("VS Code")
        app_names = [s.app_name for s in snapshots]
        assert "SystemUIServer" not in app_names

    def test_skips_empty_summaries(self):
        trail = ContextTrail()
        trail.record(app_name="Discord", window_title="w", subtree_xml="")
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        snapshots = trail.get_recent_cross_app_context("VS Code")
        for s in snapshots:
            assert s.text_summary.strip() != ""

    def test_allows_substantive_apps(self):
        trail = ContextTrail()
        trail.record(
            app_name="Safari",
            window_title="w",
            subtree_xml="<t>Python documentation: list comprehensions</t>",
        )
        trail.record(app_name="VS Code", window_title="w", subtree_xml="<t>code</t>")

        snapshots = trail.get_recent_cross_app_context("VS Code")
        assert len(snapshots) == 1
        assert snapshots[0].app_name == "Safari"
