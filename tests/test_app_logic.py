"""Tests for app.py helpers and orchestration glue."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import the class to access its static methods.
# We need to be careful because app.py imports heavy macOS deps, but
# these are guarded by try/except, so the import should succeed.
import autocompleter.app as app_module
from autocompleter.app import Autocompleter
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


class TestPostAcceptFollowup:
    def test_build_post_accept_focused_state_empty_draft(self):
        """Follow-up always starts with an empty draft — the accepted text is committed."""
        live = FocusedElement(
            app_name="Codex",
            app_pid=123,
            role="AXTextArea",
            value="transient",
            selected_text="",
            position=(10.0, 20.0),
            size=(300.0, 40.0),
            insertion_point=4,
            placeholder_detected=False,
        )

        result = Autocompleter._build_post_accept_focused_state(
            live_focused=live,
        )

        assert result.before_cursor == ""
        assert result.after_cursor == ""
        assert result.insertion_point == 0
        assert result.value == ""
        assert result.position == live.position

    def test_reuses_saved_context_and_refreshes_focus_state(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        focused = FocusedElement(
            app_name="Codex",
            app_pid=123,
            role="AXTextArea",
            value="transient partial",
            selected_text="",
            position=(50.0, 60.0),
            size=(200.0, 20.0),
            insertion_point=10,
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
        app._trigger_before_cursor = "hello world"
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
            "cross_app_context": "[Recent activity from other apps]\n- Terminal: ...",
            "subtree_context": "<context><TextArea>old</TextArea></context>",
            "trigger_type": "manual",
        }

        app._start_post_accept_followup(" there")

        assert app._generation_id == 5
        # Follow-up always starts with empty draft
        assert app._trigger_before_cursor == ""
        assert app._trigger_mode == "reply"
        assert overlay_calls == [("Generating...", 50.0, 80.0, 20.0)]
        assert app._last_trigger_args["focused"] is not focused
        assert app._last_trigger_args["focused"].before_cursor == ""
        assert app._last_trigger_args["cross_app_context"].startswith("[Recent activity")
        assert app._last_trigger_args["subtree_context"] == "<context><TextArea>old</TextArea></context>"
        assert app._last_trigger_args["trigger_type"] == "post_accept"
        # Conversation turns should include the committed text as a new "You" turn
        updated_turns = app._last_trigger_args["conversation_turns"]
        assert len(updated_turns) == 2
        assert updated_turns[0] == {"speaker": "User", "text": "hi"}
        assert updated_turns[1].speaker == "You"
        assert updated_turns[1].text == "hello world there"
        assert captured["latency_start"] == 5
        assert captured["started"] is True
        assert captured["args"][0].before_cursor == ""
        assert captured["args"][4] == app_module.AutocompleteMode.REPLY
        # Generation thread also gets updated conversation turns
        assert len(captured["args"][7]) == 2
        assert captured["args"][7][1].speaker == "You"
        assert captured["args"][9].startswith("[Recent activity")
        assert captured["args"][10] is None
        assert captured["args"][11] == "<context><TextArea>old</TextArea></context>"
        assert captured["args"][14] == "post_accept"

    def test_followup_no_conversation_turns_stays_empty(self, monkeypatch):
        """When no conversation turns exist (non-chat app), list stays empty."""
        app = Autocompleter.__new__(Autocompleter)
        focused = FocusedElement(
            app_name="TextEdit",
            app_pid=456,
            role="AXTextArea",
            value="some text",
            selected_text="",
            position=(50.0, 60.0),
            size=(200.0, 20.0),
            insertion_point=9,
        )

        captured = {}

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                captured["args"] = args
            def start(self):
                captured["started"] = True

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(app_module, "_get_caret_screen_position", lambda: None)

        app.config = SimpleNamespace(followup_after_accept_enabled=True)
        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        app.overlay = SimpleNamespace(show=lambda *a, **kw: None)
        app._dumper = None
        app._generation_id = 1
        app._trigger_time = None
        app._trigger_before_cursor = "some text"
        app._trigger_after_cursor = ""
        app._trigger_mode = ""
        app._trigger_app = ""
        app._replace_on_inject = False
        app._latency_tracker = SimpleNamespace(
            start=lambda **kw: None, mark=lambda s: None,
        )
        app._generate_and_show_streaming = lambda *args, **kwargs: None
        app._last_trigger_args = {
            "focused": object(),
            "x": 1.0, "y": 2.0, "caret_height": 20.0,
            "mode": app_module.AutocompleteMode.CONTINUATION,
            "window_title": "Untitled",
            "source_url": "",
            "conversation_turns": [],
            "cross_app_context": "",
            "subtree_context": None,
            "trigger_type": "manual",
        }

        app._start_post_accept_followup(" continued")

        # No conversation turns → stays empty, no "You" turn appended
        assert app._last_trigger_args["conversation_turns"] == []
        assert captured["args"][7] == []
        # Draft is still empty for follow-ups
        assert app._trigger_before_cursor == ""


class TestRegenerateDiversity:
    def test_regenerate_uses_higher_temperature_boost(self, monkeypatch):
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

        captured = {}

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                captured["target"] = target
                captured["args"] = args
                captured["kwargs"] = kwargs or {}

            def start(self):
                captured["started"] = True

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        app.overlay = SimpleNamespace(is_visible=True, show=lambda *args, **kwargs: None)
        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        def _fake_new_snapshot_for_focus(**kwargs):
            captured["snapshot_kwargs"] = kwargs
            return "snapshot"
        app._new_snapshot_for_focus = _fake_new_snapshot_for_focus
        app._last_trigger_args = {
            "focused": focused,
            "x": 1.0,
            "y": 2.0,
            "caret_height": 20.0,
            "mode": app_module.AutocompleteMode.CONTINUATION,
            "window_title": "Codex",
            "source_url": "",
            "conversation_turns": None,
            "cross_app_context": "",
            "subtree_context": "<context></context>",
            "trigger_type": "manual",
        }
        app._current_suggestions = [SimpleNamespace(text="old one"), SimpleNamespace(text="old two")]
        app._generation_id = 0
        app._run_on_main = lambda fn: None
        app._auto_trigger_debouncer = SimpleNamespace(cancel=lambda: None)
        app._latency_tracker = SimpleNamespace(start=lambda **kwargs: None, mark=lambda *args, **kwargs: None)
        app._capture_live_trigger_context = lambda focused, trigger_type: (
            app_module.AutocompleteMode.CONTINUATION,
            10.0, 20.0, 20.0,
            "Codex", "", None, "", "<context></context>",
        )

        assert app._on_regenerate() is True
        assert captured["kwargs"]["temperature_boost"] == 0.5
        assert captured["args"][10] == "snapshot"
        assert captured["snapshot_kwargs"]["trigger_type"] == "regenerate"
        assert captured["snapshot_kwargs"]["generation_id"] == 1


class TestTelemetryHooks:
    def test_start_emits_app_started(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        events = []

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self.target = target

            def start(self):
                return None

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(app_module, "HAS_APPKIT", False)

        app.config = SimpleNamespace(
            db_path="/tmp/context.db",
            hotkey="ctrl+space",
            regenerate_hotkey="ctrl+r",
            llm_provider="openai",
            llm_model="ignored",
            effective_llm_provider="openai",
            effective_llm_model="beta-model",
        )
        app.observer = SimpleNamespace(check_accessibility_permissions=lambda: True)
        app.context_store = SimpleNamespace(open=lambda: None, close=lambda: None)
        app.memory = SimpleNamespace(enabled=False)
        app.overlay = SimpleNamespace(set_dismiss_callback=lambda cb: None, hide=lambda: None)
        app.hotkey_listener = SimpleNamespace(
            set_unhandled_key_callback=lambda cb: None,
            register=lambda *args, **kwargs: None,
            start=lambda: None,
            stop=lambda: None,
        )
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)), stop=lambda: None)
        app._auto_trigger_enabled = False
        app._auto_trigger_debouncer = SimpleNamespace(stop=lambda: None, cancel=lambda: None, poke=lambda: None)
        app._observe_loop = lambda: None
        app._run_simple_loop = lambda: None

        app.start()

        assert ("app_started", {}) in events

    def test_emit_trigger_telemetry_buckets_app_category(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))

        app._emit_trigger_telemetry(
            mode=app_module.AutocompleteMode.REPLY,
            trigger_type="manual",
            app_name="Slack",
        )

        assert events == [(
            "trigger_fired",
            {
                "mode": "reply",
                "trigger_type": "manual",
                "app_category": "chat",
            },
        )]

    def test_accept_selected_suggestion_emits_accepted_event(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        followup = []
        focused = FocusedElement(
            app_name="Slack",
            app_pid=1,
            role="AXTextArea",
            value="hello",
            selected_text="",
            position=None,
            size=None,
            insertion_point=5,
        )

        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        app.overlay = SimpleNamespace(accept_selection=lambda: app_module.Suggestion(text=" world", index=1))
        app.injector = SimpleNamespace(inject=lambda text: True)
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app.memory = SimpleNamespace(enabled=False)
        app.context_store = SimpleNamespace(record_feedback=lambda **kwargs: None)
        app._trigger_before_cursor = "hello"
        app._trigger_after_cursor = ""
        app._trigger_mode = "reply"
        app._trigger_app = "Slack"
        app._current_suggestions = [app_module.Suggestion(text="a", index=0), app_module.Suggestion(text=" world", index=1)]
        app._trigger_time = None
        app._start_post_accept_followup = lambda text: followup.append(text)

        app._accept_selected_suggestion()

        assert followup == [" world"]
        assert events == [(
            "suggestion_accepted",
            {
                "suggestion_rank": 2,
                "mode": "reply",
                "accepted_length_bucket": "1-10",
            },
        )]

    def test_partial_accept_emits_partial_event(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        injected = []
        focused = FocusedElement(
            app_name="Slack",
            app_pid=1,
            role="AXTextArea",
            value="hello",
            selected_text="",
            position=None,
            size=None,
            insertion_point=5,
        )

        app.overlay = SimpleNamespace(
            is_visible=True,
            accept_selection=lambda: app_module.Suggestion(text="Hello world. Next sentence", index=0),
        )
        app.injector = SimpleNamespace(inject=lambda text: injected.append(text) or True)
        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app._trigger_before_cursor = ""
        app._trigger_after_cursor = ""
        app._run_on_main = lambda fn: fn()

        assert app._on_partial_accept() is True
        assert injected == ["Hello world."]
        assert events == [(
            "partial_accept_used",
            {
                "suggestion_rank": 1,
                "accepted_length_bucket": "11-30",
            },
        )]

    def test_dismiss_emits_single_event(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        hidden = []

        app.overlay = SimpleNamespace(is_visible=True, hide=lambda: hidden.append(True))
        app._auto_trigger_debouncer = SimpleNamespace(cancel=lambda: None)
        app._generation_id = 0
        app._last_trigger_args = {"x": 1}
        app._current_suggestions = [
            app_module.Suggestion(text="one", index=0),
            app_module.Suggestion(text="two", index=1),
        ]
        app._trigger_mode = "reply"
        app._trigger_app = "Slack"
        app._trigger_time = None
        app.context_store = SimpleNamespace(record_feedback=lambda **kwargs: None)
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app._run_on_main = lambda fn: fn()

        assert app._on_nav_dismiss() is True
        assert hidden == [True]
        assert events == [(
            "suggestion_dismissed",
            {
                "count_shown": 2,
                "mode": "reply",
            },
        )]

    def test_streaming_worker_error_emits_error_event(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        hidden = []

        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app.overlay = SimpleNamespace(hide=lambda: hidden.append(True))
        app._run_on_main = lambda fn: fn()
        app._generate_and_show_streaming_inner = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom"))

        app._generate_and_show_streaming(SimpleNamespace(app_name="Slack"), 1.0, 2.0)

        assert hidden == [True]
        assert events == [(
            "error_occurred",
            {
                "component": "streaming_worker",
                "error_type": "ValueError",
            },
        )]
