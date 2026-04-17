"""Tests for app.py helpers and orchestration glue."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import the class to access its static methods.
# We need to be careful because app.py imports heavy macOS deps, but
# these are guarded by try/except, so the import should succeed.
import autocompleter.app as app_module
from autocompleter.app import Autocompleter
from autocompleter.feedback import FeedbackContext
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
    def test_build_post_accept_focused_state_uses_synthetic_inserted_text(self):
        """Follow-up should synthesize the post-accept cursor state immediately."""
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
            accepted_text=" world",
            before_cursor="hello",
            after_cursor="!",
        )

        assert result.before_cursor == "hello world"
        assert result.after_cursor == "!"
        assert result.insertion_point == 11
        assert result.value == "hello world!"
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
        assert app._trigger_before_cursor == "hello world there"
        assert app._trigger_mode == "continuation"
        assert overlay_calls == [("Generating...", 50.0, 80.0, 20.0)]
        assert app._last_trigger_args["focused"] is not focused
        assert app._last_trigger_args["focused"].before_cursor == "hello world there"
        assert app._last_trigger_args["focused"].value == "hello world there"
        assert app._last_trigger_args["cross_app_context"].startswith("[Recent activity")
        assert app._last_trigger_args["subtree_context"] == "<context><TextArea>old</TextArea></context>"
        assert app._last_trigger_args["trigger_type"] == "post_accept"
        updated_turns = app._last_trigger_args["conversation_turns"]
        assert len(updated_turns) == 1
        assert updated_turns[0] == {"speaker": "User", "text": "hi"}
        assert captured["latency_start"] == 5
        assert captured["started"] is True
        assert captured["args"][0].before_cursor == "hello world there"
        assert captured["args"][4] == app_module.AutocompleteMode.CONTINUATION
        assert len(captured["args"][7]) == 1
        assert captured["args"][9].startswith("[Recent activity")
        assert captured["args"][10] is None
        assert captured["args"][11] == "<context><TextArea>old</TextArea></context>"
        assert captured["kwargs"]["trigger_type"] == "post_accept"

    def test_followup_no_conversation_turns_stays_empty(self, monkeypatch):
        """When no conversation turns exist, follow-up uses the live field state."""
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

        assert app._last_trigger_args["conversation_turns"] == []
        assert captured["args"][7] == []
        assert app._trigger_before_cursor == "some text continued"

    def test_followup_keeps_synthetic_text_even_if_live_field_is_empty(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        focused = FocusedElement(
            app_name="Codex",
            app_pid=123,
            role="AXTextArea",
            value="",
            selected_text="",
            position=(50.0, 60.0),
            size=(200.0, 20.0),
            insertion_point=0,
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
        app._trigger_before_cursor = "hello world"
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
            "mode": app_module.AutocompleteMode.REPLY,
            "window_title": "Codex",
            "source_url": "",
            "conversation_turns": [{"speaker": "User", "text": "hi"}],
            "cross_app_context": "",
            "subtree_context": None,
            "trigger_type": "manual",
        }

        app._start_post_accept_followup(" there")

        updated_turns = app._last_trigger_args["conversation_turns"]
        assert len(updated_turns) == 1
        assert updated_turns[0] == {"speaker": "User", "text": "hi"}
        assert app._last_trigger_args["focused"].before_cursor == "hello world there"


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
        app._active_invocation = None
        app._capture_live_trigger_context = lambda focused, trigger_type: (
            app_module.AutocompleteMode.CONTINUATION,
            10.0, 20.0, 20.0,
            "Codex", "", None, "", "<context></context>",
        )

        assert app._on_regenerate() is True
        assert captured["kwargs"]["temperature_boost"] == 0.5
        assert captured["kwargs"]["extra_negative_patterns"] == ["old one", "old two"]
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
            help_hotkey="ctrl+/",
            report_hotkey="ctrl+shift+b",
            llm_provider="openai",
            llm_model="ignored",
            effective_llm_provider="openai",
            effective_llm_model="beta-model",
        )
        app.observer = SimpleNamespace(check_accessibility_permissions=lambda: True)
        app.context_store = SimpleNamespace(open=lambda: None, close=lambda: None)
        app.memory = SimpleNamespace(enabled=False)
        app.overlay = SimpleNamespace(set_dismiss_callback=lambda cb: None, hide=lambda: None)
        app.help_overlay = SimpleNamespace(set_dismiss_callback=lambda cb: None, hide=lambda: None)
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
        app._active_invocation = None

        app.start()

        assert ("app_started", {}) in events

    def test_emit_trigger_telemetry_buckets_app_category(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app._active_invocation = None

        invocation_id = app._emit_trigger_telemetry(
            mode=app_module.AutocompleteMode.REPLY,
            trigger_type="manual",
            app_name="Slack",
        )

        assert invocation_id
        assert events[0][0] == "trigger_fired"
        assert events[0][1]["mode"] == "reply"
        assert events[0][1]["trigger_type"] == "manual"
        assert events[0][1]["app_category"] == "chat"
        assert events[0][1]["source_app"] == "Slack"
        assert events[0][1]["requested_route"] == "direct"
        assert events[0][1]["profile"]["requested_route"] == "direct"
        assert events[0][1]["invocation_id"] == invocation_id

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
        app.injector = SimpleNamespace(inject=lambda text, **kwargs: True)
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
        app._active_invocation = app_module._InvocationTelemetryState(
            invocation_id="inv-1",
            trigger_type="manual",
            mode="reply",
            source_app="Slack",
            app_category="chat",
            requested_route="proxy",
            started_monotonic=0.0,
            started_at="2026-04-16T00:00:00Z",
        )

        app._accept_selected_suggestion()

        assert followup == [" world"]
        assert events[0][0] == "suggestion_accepted"
        assert events[0][1]["suggestion_rank"] == 2
        assert events[0][1]["accepted_length_bucket"] == "1-10"
        assert events[0][1]["invocation_id"] == "inv-1"

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
        app.injector = SimpleNamespace(inject=lambda text, **kwargs: injected.append(text) or True)
        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app._trigger_before_cursor = ""
        app._trigger_after_cursor = ""
        app._run_on_main = lambda fn: fn()
        app._active_invocation = app_module._InvocationTelemetryState(
            invocation_id="inv-2",
            trigger_type="manual",
            mode="reply",
            source_app="Slack",
            app_category="chat",
            requested_route="proxy",
            started_monotonic=0.0,
            started_at="2026-04-16T00:00:00Z",
        )

        assert app._on_partial_accept() is True
        assert injected == ["Hello world."]
        assert events[0][0] == "partial_accept_used"
        assert events[0][1]["suggestion_rank"] == 1
        assert events[0][1]["accepted_length_bucket"] == "11-30"
        assert events[0][1]["invocation_id"] == "inv-2"

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
        app._active_invocation = app_module._InvocationTelemetryState(
            invocation_id="inv-3",
            trigger_type="manual",
            mode="reply",
            source_app="Slack",
            app_category="chat",
            requested_route="proxy",
            started_monotonic=0.0,
            started_at="2026-04-16T00:00:00Z",
        )

        assert app._on_nav_dismiss() is True
        assert hidden == [True]
        assert events[0][0] == "suggestion_dismissed"
        assert events[0][1]["count_shown"] == 2
        assert events[0][1]["invocation_id"] == "inv-3"

    def test_streaming_worker_error_emits_error_event(self):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        hidden = []

        app.telemetry = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
        app.overlay = SimpleNamespace(hide=lambda: hidden.append(True))
        app._run_on_main = lambda fn: fn()
        app._generate_and_show_streaming_inner = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom"))
        app._active_invocation = None

        app._generate_and_show_streaming(SimpleNamespace(app_name="Slack"), 1.0, 2.0)

        assert hidden == [True]
        assert events == [(
            "error_occurred",
            {
                "component": "streaming_worker",
                "error_type": "ValueError",
            },
        )]

    def test_build_feedback_context_uses_saved_trigger_metadata(self):
        app = Autocompleter.__new__(Autocompleter)
        focused = FocusedElement(
            app_name="Slack",
            app_pid=1,
            role="AXTextArea",
            value="hidden",
            selected_text="",
            position=None,
            size=None,
            insertion_point=0,
            placeholder_detected=True,
        )
        app.config = SimpleNamespace(
            effective_llm_provider="openai",
            effective_llm_model="beta-model",
            effective_fallback_provider="openai",
            effective_fallback_model="fallback-model",
        )
        app._last_trigger_args = {
            "focused": focused,
            "mode": app_module.AutocompleteMode.REPLY,
            "trigger_type": "manual",
            "source_url": "https://slack.com/app_redirect?channel=C123",
            "conversation_turns": [{"speaker": "User"}, {"speaker": "Assistant"}],
            "subtree_context": "<TextArea>hidden</TextArea>",
            "window_title": "Slack",
        }
        app._last_latency_record = SimpleNamespace(
            suggestion_count=3,
            e2e_total_ms=950.0,
            e2e_first_ms=420.0,
            use_tui=False,
            use_shell=False,
            used_subtree_context=True,
            used_semantic_context=False,
            used_memory_context=True,
            visible_source="cache",
        )
        app._last_fallback_used = True

        context = app._build_feedback_context()

        assert isinstance(context, FeedbackContext)
        assert context.app_name == "Slack"
        assert context.app_bundle_role == "AXTextArea"
        assert context.url_domain == "slack.com"
        assert context.mode == "reply"
        assert context.trigger_type == "manual"
        assert context.conversation_turns_detected == 2
        assert context.conversation_speakers == 2
        assert context.subtree_context_chars == len("<TextArea>hidden</TextArea>")
        assert context.fallback_used is True
        assert context.suggestion_count == 3

    def test_report_feedback_persists_and_emits_telemetry(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        shown = []
        focused = FocusedElement(
            app_name="Slack",
            app_pid=1,
            role="AXTextArea",
            value="hidden",
            selected_text="",
            position=(10.0, 10.0),
            size=(100.0, 20.0),
            insertion_point=0,
        )

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(app_module, "_get_caret_screen_position", lambda: (10.0, 10.0, 20.0))

        app.config = SimpleNamespace(
            install_id="install-123",
            effective_llm_provider="openai",
            effective_llm_model="beta-model",
            effective_fallback_provider="openai",
            effective_fallback_model="fallback-model",
        )
        app.feedback_reporter = SimpleNamespace(
            submit=lambda ctx, installation_id=None: {
                "report_id": "report-123",
                "installation_id": installation_id,
                "app": {"name": ctx.app_name},
            }
        )
        app.overlay = SimpleNamespace(show=lambda suggestions, x, y, caret_height=20.0: shown.append(suggestions[0].text), hide=lambda: None)
        app.observer = SimpleNamespace(get_focused_element=lambda: focused)
        app._emit_telemetry = lambda event, **payload: events.append((event, payload))
        app._run_on_main = lambda fn: fn()
        app._generation_id = 0
        app._last_latency_record = None
        app._last_fallback_used = False
        app._last_trigger_args = {
            "focused": focused,
            "mode": app_module.AutocompleteMode.REPLY,
            "trigger_type": "manual",
            "source_url": "https://slack.com/client/T123",
            "conversation_turns": [],
            "subtree_context": None,
            "window_title": "Slack",
        }

        assert app._on_report_feedback() is True
        assert shown == ["Report captured (Slack)"]
        assert events == [(
            "feedback_reported",
            {
                "report": {
                    "report_id": "report-123",
                    "installation_id": "install-123",
                    "app": {"name": "Slack"},
                },
            },
        )]

    def test_report_feedback_uploads_debug_artifact_when_manual_capture_enabled(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        events = []
        debug_artifacts = []
        cdp_probes = []
        focused = FocusedElement(
            app_name="Slack",
            app_pid=1,
            role="AXTextArea",
            value="hidden",
            selected_text="",
            position=(10.0, 10.0),
            size=(100.0, 20.0),
            insertion_point=0,
        )

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(app_module, "_get_caret_screen_position", lambda: (10.0, 10.0, 20.0))
        monkeypatch.setattr(
            app_module,
            "probe_editable_dom_state",
            lambda app_name, app_pid: cdp_probes.append((app_name, app_pid)) or {"status": "success"},
        )

        app.config = SimpleNamespace(
            install_id="install-123",
            effective_llm_provider="openai",
            effective_llm_model="beta-model",
            effective_fallback_provider="openai",
            effective_fallback_model="fallback-model",
            debug_capture_profile="aggressive",
        )
        app.feedback_reporter = SimpleNamespace(
            submit=lambda ctx, installation_id=None: {
                "report_id": "report-123",
                "installation_id": installation_id,
                "app": {"name": ctx.app_name},
            }
        )
        app.debug_artifacts = SimpleNamespace(
            enabled=True,
            manual_capture_enabled=True,
            emit_artifact=lambda *args, **kwargs: debug_artifacts.append((args, kwargs)),
        )
        app._log_buffer_handler = SimpleNamespace(snapshot=lambda limit=200: ["tail"])
        app.overlay = SimpleNamespace(show=lambda suggestions, x, y, caret_height=20.0: None, hide=lambda: None)
        app.observer = SimpleNamespace(
            get_focused_element=lambda: focused,
            get_focus_debug_info=lambda profile="normal": {
                "frontmost_app": {"name": "Slack", "pid": 1},
                "profile_used": profile,
            },
        )
        app._emit_telemetry = lambda event, **payload: events.append((event, payload))
        app._run_on_main = lambda fn: fn()
        app._generation_id = 0
        app._last_latency_record = None
        app._last_fallback_used = False
        app._last_trigger_args = {
            "focused": focused,
            "mode": app_module.AutocompleteMode.REPLY,
            "trigger_type": "manual",
            "source_url": "https://slack.com/client/T123",
            "conversation_turns": [],
            "subtree_context": None,
            "window_title": "Slack",
            "invocation_id": "inv-123",
        }

        assert app._on_report_feedback() is True
        assert events[0][0] == "feedback_reported"
        assert len(debug_artifacts) == 1
        assert debug_artifacts[0][0][0] == "manual_report"
        assert debug_artifacts[0][1]["invocation_id"] == "inv-123"
        payload = debug_artifacts[0][0][1]
        assert payload["focus_debug"]["profile_used"] == "normal"
        assert "cdp_probe" not in payload["focus_debug"]
        assert cdp_probes == []

    def test_trigger_focus_failure_uploads_debug_artifact(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        debug_artifacts = []
        profiles = []

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(
            app_module,
            "probe_editable_dom_state",
            lambda app_name, app_pid: {
                "status": "success",
                "app_name": app_name,
                "app_pid": app_pid,
                "editable_candidates": [{"tag": "textarea"}],
            },
        )

        app._auto_trigger_debouncer = SimpleNamespace(cancel=lambda: None)
        app.overlay = SimpleNamespace(is_visible=False)
        app.observer = SimpleNamespace(
            get_focused_element=lambda: None,
            get_focus_debug_info=lambda profile="normal": profiles.append(profile) or {
                "frontmost_app": {
                    "name": "Google Chrome",
                    "pid": 321,
                },
                "window_inventory": [{"title": "ChatGPT", "role": "AXWindow"}],
            },
        )
        app.debug_artifacts = SimpleNamespace(
            enabled=True,
            failure_capture_enabled=True,
            emit_artifact=lambda *args, **kwargs: debug_artifacts.append((args, kwargs)),
        )
        app._log_buffer_handler = SimpleNamespace(snapshot=lambda limit=200: ["tail"])
        app._latency_tracker = SimpleNamespace(start=lambda *args, **kwargs: None, mark=lambda *args, **kwargs: None)
        app._generation_id = 0
        app._last_trigger_args = None
        app._active_invocation = None
        app.config = SimpleNamespace(
            install_id="install-123",
            debug_capture_profile="aggressive",
        )

        assert app._on_trigger() is True
        assert len(debug_artifacts) == 1
        assert debug_artifacts[0][0][0] == "focus_failure"
        assert debug_artifacts[0][1]["trigger_type"] == "manual"
        payload = debug_artifacts[0][0][1]
        assert profiles == ["aggressive"]
        assert payload["focus_debug"]["window_inventory"][0]["title"] == "ChatGPT"
        assert payload["focus_debug"]["cdp_probe"]["status"] == "success"

    def test_focus_success_snapshot_uploads_debug_artifact_when_enabled(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        debug_artifacts = []
        profiles = []
        focused = FocusedElement(
            app_name="Codex",
            app_pid=321,
            role="AXTextArea",
            value="hello world",
            selected_text="",
            position=(10.0, 10.0),
            size=(100.0, 20.0),
            insertion_point=11,
        )

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(
            app_module,
            "probe_editable_dom_state",
            lambda app_name, app_pid: {
                "status": "success",
                "app_name": app_name,
                "app_pid": app_pid,
                "active_element": {"tag": "textarea"},
            },
        )

        app.observer = SimpleNamespace(
            get_focus_debug_info=lambda profile="normal": profiles.append(profile) or {
                "frontmost_app": {
                    "name": "Codex",
                    "pid": 321,
                },
                "window_inventory": [{"title": "Codex", "role": "AXWindow"}],
            },
        )
        app.debug_artifacts = SimpleNamespace(
            enabled=True,
            emit_artifact=lambda *args, **kwargs: debug_artifacts.append((args, kwargs)),
        )
        app._log_buffer_handler = SimpleNamespace(snapshot=lambda limit=200: ["tail"])
        app._active_invocation = None
        app._last_trigger_args = None
        app.config = SimpleNamespace(
            install_id="install-123",
            debug_capture_profile="aggressive",
            debug_capture_success_enabled=True,
        )

        app._capture_focus_success_snapshot(
            trigger_type="manual",
            focused=focused,
            invocation_id="inv-123",
        )

        assert len(debug_artifacts) == 1
        assert debug_artifacts[0][0][0] == "focus_snapshot"
        assert debug_artifacts[0][1]["trigger_type"] == "manual"
        payload = debug_artifacts[0][0][1]
        assert profiles == ["aggressive"]
        assert payload["focus_debug"]["window_inventory"][0]["title"] == "Codex"
        assert payload["focus_debug"]["cdp_probe"]["status"] == "success"
        assert payload["extra"]["focused_role"] == "AXTextArea"
        assert payload["extra"]["focused_value_length"] == 11

    def test_accept_injection_failure_uploads_debug_artifact(self, monkeypatch):
        app = Autocompleter.__new__(Autocompleter)
        debug_artifacts = []
        focused = FocusedElement(
            app_name="Codex",
            app_pid=1,
            role="AXTextArea",
            value="hello",
            selected_text="",
            position=(10.0, 10.0),
            size=(100.0, 20.0),
            insertion_point=5,
        )

        class FakeThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(app_module.threading, "Thread", FakeThread)

        app.observer = SimpleNamespace(
            get_focused_element=lambda: focused,
            get_focus_debug_info=lambda: {"frontmost_app": {"name": "Codex"}},
        )
        app.overlay = SimpleNamespace(
            _selected_index=0,
            accept_selection=lambda: SimpleNamespace(text=" world", index=0),
        )
        app.injector = SimpleNamespace(inject=lambda text, app_name, app_pid: False)
        app.debug_artifacts = SimpleNamespace(
            enabled=True,
            failure_capture_enabled=True,
            emit_artifact=lambda *args, **kwargs: debug_artifacts.append((args, kwargs)),
        )
        app._log_buffer_handler = SimpleNamespace(snapshot=lambda limit=200: ["tail"])
        app._active_invocation = SimpleNamespace(invocation_id="inv-123")
        app._last_trigger_args = {
            "focused": focused,
            "trigger_type": "manual",
            "window_title": "Codex",
            "source_url": "",
            "conversation_turns": [],
            "subtree_context": None,
            "tree_overview_context": None,
            "context_tree": None,
            "invocation_id": "inv-123",
        }
        app._trigger_before_cursor = "hello"
        app._trigger_after_cursor = ""
        app._current_suggestions = []
        app._record_accepted_suggestion = lambda *args, **kwargs: None
        app._start_post_accept_followup = lambda *args, **kwargs: None
        app.config = SimpleNamespace(install_id="install-123")

        app._accept_selected_suggestion()

        assert len(debug_artifacts) == 1
        assert debug_artifacts[0][0][0] == "injection_failure"
        assert debug_artifacts[0][1]["invocation_id"] == "inv-123"
