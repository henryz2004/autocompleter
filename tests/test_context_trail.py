"""Tests for the cross-app context trail."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from autocompleter.context_trail import AppSnapshot, ContextTrail
from autocompleter.context_store import ContextStore
from autocompleter.input_observer import VisibleContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_visible_content(
    app_name: str = "TestApp",
    window_title: str = "Test Window",
    text_elements: list[str] | None = None,
    url: str = "",
    app_pid: int = 1234,
) -> VisibleContent:
    """Create a mock VisibleContent for testing."""
    return VisibleContent(
        app_name=app_name,
        app_pid=app_pid,
        window_title=window_title,
        text_elements=text_elements or ["Some visible text"],
        url=url,
    )


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
        """Recording from a single app should not create any trail entries
        because there is no app switch."""
        content = _make_visible_content(app_name="Chrome")
        trail.record(content)
        trail.record(content)
        trail.record(content)

        snapshots = trail.get_recent_cross_app_context("Chrome")
        assert len(snapshots) == 0

    def test_app_switch_creates_snapshot(self, trail):
        """Switching A -> B should create a snapshot of A."""
        content_a = _make_visible_content(
            app_name="Chrome",
            window_title="React Docs",
            text_elements=["useState allows you to add state"],
        )
        content_b = _make_visible_content(
            app_name="VS Code",
            window_title="App.tsx",
            text_elements=["import React from 'react'"],
        )

        trail.record(content_a)
        trail.record(content_b)  # Switch: should snapshot Chrome

        # From VS Code's perspective, Chrome should appear
        snapshots = trail.get_recent_cross_app_context("VS Code")
        assert len(snapshots) == 1
        assert snapshots[0].app_name == "Chrome"
        assert snapshots[0].window_title == "React Docs"
        assert "useState" in snapshots[0].text_summary

    def test_multiple_switches(self, trail):
        """A -> B -> C -> A should create snapshots for A, B, C."""
        apps = [
            _make_visible_content(app_name="Chrome", text_elements=["Chrome text"]),
            _make_visible_content(app_name="VS Code", text_elements=["VS Code text"]),
            _make_visible_content(app_name="Slack", text_elements=["Slack text"]),
            _make_visible_content(app_name="Chrome", text_elements=["Chrome text 2"]),
        ]

        for content in apps:
            trail.record(content)

        # From Chrome's perspective: should see VS Code and Slack
        snapshots = trail.get_recent_cross_app_context("Chrome")
        assert len(snapshots) == 2
        app_names = [s.app_name for s in snapshots]
        assert "VS Code" in app_names
        assert "Slack" in app_names

    def test_deduplication_consecutive_same_app_window(self, trail):
        """Consecutive snapshots for the same app+window should be deduplicated.

        Scenario: Chrome -> VS Code -> Chrome (same window) -> VS Code
        This creates trail entries [Chrome, VS Code, Chrome] -- not consecutive
        dupes, so all three are kept.

        But if Chrome appears as the last entry and we try to push another
        Chrome snapshot, the dedup fires.
        """
        # Test true consecutive dedup: A -> B -> A -> B should create
        # [A, B, A] but if the second push of A matches the last entry,
        # it updates in place instead of appending.
        chrome = _make_visible_content(
            app_name="Chrome",
            window_title="Docs",
            text_elements=["First visit"],
        )
        vscode = _make_visible_content(app_name="VS Code", text_elements=["code"])
        chrome_2 = _make_visible_content(
            app_name="Chrome",
            window_title="Docs",
            text_elements=["Second visit"],
        )

        # A -> B: trail = [Chrome(Docs)]
        trail.record(chrome)
        trail.record(vscode)

        # B -> A: trail = [Chrome(Docs), VS Code]
        trail.record(chrome_2)

        # A -> B again: tries to push Chrome(Docs) -- but last entry is VS Code,
        # so NOT consecutive. Trail = [Chrome(Docs), VS Code, Chrome(Docs)]
        trail.record(vscode)

        all_snapshots = trail.get_recent_cross_app_context(
            "NONE", max_age_seconds=9999, max_entries=20,
        )
        # Non-consecutive: both Chrome entries should exist
        chrome_entries = [s for s in all_snapshots if s.app_name == "Chrome"]
        assert len(chrome_entries) == 2

    def test_deduplication_truly_consecutive(self, trail):
        """When the last trail entry matches app+window, it should be updated
        in place rather than creating a duplicate.

        This happens when: A -> B -> A -> B, and before the second A->B switch,
        the last trail entry is already A (from the first A->B switch). But
        since we interleave with B, the second snapshot of A won't be consecutive.

        A true consecutive scenario requires: We push A, then try to push A again
        without any intervening entries. This happens when record() is called with
        the same previous-app state -- which requires manually constructing the
        scenario.
        """
        # Directly manipulate to test the dedup path
        from autocompleter.context_trail import AppSnapshot
        import time

        trail._trail.append(AppSnapshot(
            app_name="Chrome",
            window_title="Docs",
            text_summary="old text",
            timestamp=time.time() - 10,
        ))
        trail._current_app = "Chrome"
        trail._current_window = "Docs"
        trail._current_content = _make_visible_content(
            app_name="Chrome",
            window_title="Docs",
            text_elements=["updated text"],
        )

        # Now simulate switching to VS Code -- this would push a Chrome snapshot,
        # but since the last trail entry is already Chrome+Docs, it should dedup
        trail.record(_make_visible_content(app_name="VS Code", text_elements=["code"]))

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
        """Snapshots from the current app should be excluded."""
        trail.record(_make_visible_content(app_name="Chrome", text_elements=["chrome"]))
        trail.record(_make_visible_content(app_name="VS Code", text_elements=["code"]))

        snapshots = trail.get_recent_cross_app_context("Chrome")
        for s in snapshots:
            assert s.app_name != "Chrome"

    def test_respects_max_age(self, trail):
        """Old entries should be excluded based on max_age_seconds."""
        trail.record(_make_visible_content(app_name="Chrome", text_elements=["old"]))
        trail.record(_make_visible_content(app_name="VS Code", text_elements=["code"]))

        # Manually age the Chrome snapshot
        if trail._trail:
            trail._trail[0] = AppSnapshot(
                app_name=trail._trail[0].app_name,
                window_title=trail._trail[0].window_title,
                text_summary=trail._trail[0].text_summary,
                timestamp=time.time() - 120,  # 2 minutes ago
                source_url=trail._trail[0].source_url,
            )

        # With max_age=60, the Chrome entry should be excluded
        snapshots = trail.get_recent_cross_app_context(
            "VS Code", max_age_seconds=60,
        )
        assert len(snapshots) == 0

        # With max_age=300, it should be included
        snapshots = trail.get_recent_cross_app_context(
            "VS Code", max_age_seconds=300,
        )
        assert len(snapshots) == 1

    def test_respects_max_entries(self, trail):
        """Should return at most max_entries snapshots."""
        # Create many app switches
        apps = ["Chrome", "VS Code", "Slack", "Terminal", "Safari"]
        for i, app in enumerate(apps):
            trail.record(_make_visible_content(
                app_name=app, text_elements=[f"text from {app}"],
            ))
        # One more to push the last app's snapshot
        trail.record(_make_visible_content(app_name="Notes", text_elements=["notes"]))

        snapshots = trail.get_recent_cross_app_context(
            "Notes", max_age_seconds=9999, max_entries=2,
        )
        assert len(snapshots) == 2

    def test_returns_most_recent_first(self, trail):
        """Snapshots should be ordered newest first."""
        trail.record(_make_visible_content(app_name="Chrome", text_elements=["first"]))
        trail.record(_make_visible_content(app_name="VS Code", text_elements=["second"]))
        trail.record(_make_visible_content(app_name="Slack", text_elements=["third"]))
        trail.record(_make_visible_content(app_name="Terminal", text_elements=["fourth"]))

        snapshots = trail.get_recent_cross_app_context(
            "Terminal", max_age_seconds=9999, max_entries=10,
        )
        # Should be Slack, VS Code, Chrome (most recent switch first)
        assert len(snapshots) == 3
        assert snapshots[0].app_name == "Slack"
        assert snapshots[1].app_name == "VS Code"
        assert snapshots[2].app_name == "Chrome"

    def test_empty_trail(self, trail):
        """Empty trail should return empty list."""
        snapshots = trail.get_recent_cross_app_context("Chrome")
        assert snapshots == []


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

    def test_format_empty_list(self):
        result = ContextTrail.format_cross_app_context([])
        assert result == ""

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
        # Should NOT have empty quotes
        assert '("")' not in result

    def test_format_truncates_long_summary(self):
        """Summaries longer than 200 chars should be truncated in format output."""
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
        # The formatted line should contain "..." and not the full 300 chars
        assert "..." in result


# ---------------------------------------------------------------------------
# Trail maxlen and text_summary truncation
# ---------------------------------------------------------------------------

class TestTrailLimits:
    def test_maxlen_evicts_oldest(self):
        """When trail is full, oldest entries should be evicted."""
        trail = ContextTrail(maxlen=3)

        # Create 5 app switches: A -> B -> C -> D -> E -> F
        apps = ["A", "B", "C", "D", "E", "F"]
        for app in apps:
            trail.record(_make_visible_content(
                app_name=app, text_elements=[f"text from {app}"],
            ))

        # Trail should contain at most 3 entries
        all_snapshots = trail.get_recent_cross_app_context(
            "F", max_age_seconds=9999, max_entries=20,
        )
        assert len(all_snapshots) <= 3

        # Oldest entries (A, B) should have been evicted
        names = [s.app_name for s in all_snapshots]
        assert "A" not in names
        assert "B" not in names

    def test_text_summary_truncation(self):
        """Long visible content should be capped at text_summary_chars."""
        trail = ContextTrail(text_summary_chars=50)

        long_text = "a" * 200
        content_a = _make_visible_content(
            app_name="Chrome", text_elements=[long_text],
        )
        content_b = _make_visible_content(app_name="VS Code", text_elements=["code"])

        trail.record(content_a)
        trail.record(content_b)  # triggers snapshot of Chrome

        snapshots = trail.get_recent_cross_app_context(
            "VS Code", max_age_seconds=9999,
        )
        assert len(snapshots) == 1
        assert len(snapshots[0].text_summary) == 50


# ---------------------------------------------------------------------------
# clear() test
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_trail(self, trail):
        trail.record(_make_visible_content(app_name="Chrome"))
        trail.record(_make_visible_content(app_name="VS Code"))
        trail.clear()

        assert trail.get_recent_cross_app_context("VS Code") == []
        assert trail._current_app == ""


# ---------------------------------------------------------------------------
# Integration with context_store
# ---------------------------------------------------------------------------

class TestContextStoreIntegration:
    def test_continuation_context_includes_cross_app(self, store):
        """Cross-app context should appear in the assembled continuation context."""
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
        """Cross-app context should appear between metadata and visible text."""
        cross_app = "[Recent activity from other apps]\n- Chrome: docs"
        context = store.get_continuation_context(
            before_cursor="typing here",
            after_cursor="",
            source_app="VS Code",
            window_title="file.py",
            visible_text=["def hello():"],
            cross_app_context=cross_app,
        )
        # Metadata should come first
        meta_idx = context.index("App: VS Code")
        cross_idx = context.index("[Recent activity from other apps]")
        visible_idx = context.index("Visible context:")
        cursor_idx = context.index("Text before cursor:")

        assert meta_idx < cross_idx < visible_idx < cursor_idx

    def test_continuation_context_empty_cross_app(self, store):
        """Empty cross_app_context should not add any section."""
        context = store.get_continuation_context(
            before_cursor="test",
            after_cursor="",
            source_app="VS Code",
            cross_app_context="",
        )
        assert "[Recent activity from other apps]" not in context

    def test_reply_context_includes_cross_app(self, store):
        """Cross-app context should appear in the assembled reply context."""
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
        """Empty cross_app_context should not add any section."""
        context = store.get_reply_context(
            conversation_turns=[],
            source_app="Slack",
            cross_app_context="",
        )
        assert "[Recent activity from other apps]" not in context

    def test_reply_context_cross_app_position(self, store):
        """Cross-app context should appear between metadata and conversation."""
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
        """Full pipeline: record app switches, format, pass to context assembly."""
        trail = ContextTrail()

        # Simulate: user reads docs in Chrome, then switches to VS Code
        trail.record(_make_visible_content(
            app_name="Chrome",
            window_title="React Hooks Docs",
            text_elements=["useState hook lets you add state to function components"],
            url="https://react.dev/hooks",
        ))
        trail.record(_make_visible_content(
            app_name="VS Code",
            window_title="App.tsx",
            text_elements=["import React from 'react'"],
        ))

        # Get cross-app context from trail
        snapshots = trail.get_recent_cross_app_context("VS Code")
        cross_app_str = ContextTrail.format_cross_app_context(snapshots)

        # Pass to context_store
        context = store.get_continuation_context(
            before_cursor="const [items, setItems] = ",
            after_cursor="",
            source_app="VS Code",
            window_title="App.tsx",
            cross_app_context=cross_app_str,
        )

        # Verify the Chrome docs context made it through
        assert "React Hooks Docs" in context
        assert "useState" in context
        assert "const [items, setItems] =" in context
