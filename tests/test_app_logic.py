"""Tests for pure helpers in app.py."""

from __future__ import annotations

from unittest.mock import Mock
from types import SimpleNamespace

import pytest

# Import the class to access its static methods.
# We need to be careful because app.py imports heavy macOS deps, but
# these are guarded by try/except, so the import should succeed.
import autocompleter.app as app_module
from autocompleter.app import Autocompleter
from autocompleter.input_observer import VisibleContent
from autocompleter.trigger_dump import TriggerSnapshot
from autocompleter.input_observer import FocusedElement


# ---------------------------------------------------------------------------
# _extract_first_segment tests
# ---------------------------------------------------------------------------

class TestExtractFirstSegment:
    def test_newline_boundary(self):
        """Text with newline returns only the first line."""
        result = Autocompleter._extract_first_segment("Hello world\nMore text here")
        assert result == "Hello world"

    def test_period_boundary(self):
        """Text with '. ' returns up to and including the period."""
        result = Autocompleter._extract_first_segment("First sentence. Second sentence")
        assert result == "First sentence."

    def test_no_boundary_returns_all(self):
        """Text with no sentence boundary returns the full text."""
        result = Autocompleter._extract_first_segment("Hello world")
        assert result == "Hello world"

    def test_empty_string(self):
        result = Autocompleter._extract_first_segment("")
        assert result == ""

    def test_newline_takes_precedence_over_period(self):
        """When both newline and period exist, newline wins."""
        result = Autocompleter._extract_first_segment("First. And more\nSecond line")
        assert result == "First. And more"


class TestPrepareInjectedText:
    def test_strips_colliding_edge_spaces(self):
        result = Autocompleter._prepare_injected_text(
            " hello ",
            "already trailing ",
            " next",
        )
        assert result == "hello"

    def test_only_trims_newlines_when_no_space_collision(self):
        result = Autocompleter._prepare_injected_text(
            "hello there\n",
            "abc",
            "",
        )
        assert result == "hello there"


# ---------------------------------------------------------------------------
# _hash_content tests
# ---------------------------------------------------------------------------

class TestHashContent:
    def test_deterministic(self):
        """Same input always produces the same hash."""
        h1 = Autocompleter._hash_content("hello world")
        h2 = Autocompleter._hash_content("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = Autocompleter._hash_content("hello")
        h2 = Autocompleter._hash_content("world")
        assert h1 != h2

    def test_unicode(self):
        """Unicode text should hash without errors."""
        h = Autocompleter._hash_content("Bonjour le monde")
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest length


class _Focused:
    def __init__(self, app_name: str, before_cursor: str = "", placeholder_detected: bool = False):
        self.app_name = app_name
        self.before_cursor = before_cursor
        self.placeholder_detected = placeholder_detected


class TestVisibleRefreshDecision:
    def test_should_revalidate_cached_visible_content_for_codex(self):
        assert Autocompleter._should_revalidate_cached_visible_content(
            "Codex",
            {"visible_source": "cache", "visible_cache_age_ms": 300.0},
            conversation_turns=None,
        )

    def test_should_not_revalidate_non_codex_cache(self):
        assert not Autocompleter._should_revalidate_cached_visible_content(
            "Slack",
            {"visible_source": "cache", "visible_cache_age_ms": 300.0},
            conversation_turns=None,
        )

    def test_should_not_revalidate_fresh_source(self):
        assert not Autocompleter._should_revalidate_cached_visible_content(
            "Codex",
            {"visible_source": "fresh", "visible_cache_age_ms": 300.0},
            conversation_turns=None,
        )


class TestWorkerVisibleRefresh:
    def test_worker_refresh_swaps_in_fresh_visible_content(self):
        app = object.__new__(Autocompleter)
        app.observer = Mock()
        app._last_trigger_args = {}

        cached = VisibleContent(
            app_name="Codex",
            app_pid=1,
            window_title="Codex",
            text_elements=["Old thread context", "Older message"],
            url="",
        )
        fresh = VisibleContent(
            app_name="Codex",
            app_pid=1,
            window_title="Codex",
            text_elements=[
                "New thread context from the current Codex chat",
                "Latest visible message in the current thread",
            ],
            url="",
        )
        app._last_visible_content = cached
        app._last_visible_content_time = 0.0
        app.observer.get_visible_content.return_value = fresh

        snapshot = TriggerSnapshot()
        focused = _Focused("Codex")
        refreshed, meta = app._maybe_refresh_visible_content_for_worker(
            focused=focused,
            window_title="Codex",
            visible_text_elements=list(cached.text_elements),
            visible_meta={"visible_source": "cache", "visible_cache_age_ms": 300.0},
            conversation_turns=None,
            snapshot=snapshot,
        )

        assert refreshed == fresh.text_elements
        assert meta["visible_source"] == "worker_refresh"
        assert meta["visible_content_changed"] is True
        assert snapshot.visible_source == "worker_refresh"
        assert snapshot.visible_text_elements == fresh.text_elements

    def test_worker_refresh_keeps_cached_visible_content_when_unchanged(self):
        app = object.__new__(Autocompleter)
        app.observer = Mock()
        app._last_trigger_args = {}

        cached = VisibleContent(
            app_name="Codex",
            app_pid=1,
            window_title="Codex",
            text_elements=["Same thread context", "Same message"],
            url="",
        )
        app._last_visible_content = cached
        app._last_visible_content_time = 0.0
        app.observer.get_visible_content.return_value = cached

        focused = _Focused("Codex")
        refreshed, meta = app._maybe_refresh_visible_content_for_worker(
            focused=focused,
            window_title="Codex",
            visible_text_elements=list(cached.text_elements),
            visible_meta={"visible_source": "cache", "visible_cache_age_ms": 300.0},
            conversation_turns=None,
        )

        assert refreshed == cached.text_elements
        assert meta["visible_source"] == "cache"
        assert meta["visible_content_changed"] is False


class TestReplyVisibleFiltering:
    def test_filters_recent_user_prompt_from_visible_text(self):
        app = object.__new__(Autocompleter)
        app.context_store = Mock()
        app.context_store.get_by_source.return_value = [
            Mock(entry_type="user_input", content="just invoked it again, can you check the logs now?"),
            Mock(entry_type="visible_text", content="other"),
        ]
        focused = _Focused("Codex", before_cursor="", placeholder_detected=True)

        result = app._filter_recent_user_inputs_from_visible_text(
            "Codex",
            [
                "just invoked it again, can you check the logs now?",
                "let's fix. also make sure the testing logic exactly mirrors the live logic",
            ],
            focused,
        )

        assert result == [
            "let's fix. also make sure the testing logic exactly mirrors the live logic",
        ]

class TestPostAcceptFollowup:
    def test_reuses_saved_context_and_refreshes_focus_state(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        focused = FocusedElement(
            app_name="Codex",
            app_pid=123,
            role="AXTextArea",
            value="hello world there",
            selected_text="",
            position=(50.0, 60.0),
            size=(200.0, 20.0),
            insertion_point=17,
        )

        overlay_calls = []
        captured = {}

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                captured["target"] = target
                captured["args"] = args
                captured["kwargs"] = kwargs or {}
                captured["daemon"] = daemon

            def start(self):
                captured["started"] = True

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(app_module, "_get_caret_screen_position", lambda: None)

        app.config = SimpleNamespace(followup_after_accept_enabled=True)
        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        app.overlay = SimpleNamespace(
            show=lambda suggestions, x, y, caret_height=20.0: overlay_calls.append(
                (suggestions[0].text, x, y, caret_height)
            )
        )
        app._dumper = None
        app._generation_id = 4
        app._trigger_time = None
        app._trigger_before_cursor = ""
        app._trigger_after_cursor = ""
        app._trigger_mode = ""
        app._trigger_app = ""
        app._replace_on_inject = False
        app._latency_tracker = SimpleNamespace(
            start=lambda generation_id=0: captured.setdefault("latency_start", generation_id),
            mark=lambda stage: captured.setdefault("latency_marks", []).append(stage),
        )
        app._generate_and_show_streaming = lambda *args, **kwargs: None
        app._last_trigger_args = {
            "focused": object(),
            "x": 1.0,
            "y": 2.0,
            "caret_height": 20.0,
            "mode": app_module.AutocompleteMode.REPLY,
            "window_title": "Codex",
            "source_url": "",
            "conversation_turns": [{"speaker": "User", "text": "hi"}],
            "visible_text_elements": ["visible context"],
            "cross_app_context": "[Recent activity from other apps]\n- Terminal: ...",
            "subtree_context": "<context><TextArea>old</TextArea></context>",
            "visible_meta": {"visible_source": "cache"},
            "trigger_type": "manual",
        }

        app._start_post_accept_followup()

        assert app._generation_id == 5
        assert app._trigger_before_cursor == "hello world there"
        assert app._trigger_mode == "continuation"
        assert overlay_calls == [("Generating...", 50.0, 80.0, 20.0)]
        assert app._last_trigger_args["focused"] is focused
        assert app._last_trigger_args["visible_text_elements"] == ["visible context"]
        assert app._last_trigger_args["cross_app_context"].startswith("[Recent activity")
        assert app._last_trigger_args["trigger_type"] == "post_accept"
        assert captured["latency_start"] == 5
        assert captured["started"] is True
        assert captured["args"][0] is focused
        assert captured["args"][4] == app_module.AutocompleteMode.CONTINUATION
        assert captured["args"][7] == [{"speaker": "User", "text": "hi"}]
        assert captured["args"][8] == ["visible context"]
        assert captured["args"][10].startswith("[Recent activity")
        assert captured["args"][12] == "<context><TextArea>old</TextArea></context>"
        assert captured["args"][16] == "post_accept"
