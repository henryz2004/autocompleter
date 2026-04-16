"""Tests for the suggestion engine."""

import time
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.config import Config
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    Suggestion,
    SuggestionEngine,
    build_messages,
    detect_mode,
    MODE_THRESHOLD_CHARS,
    postprocess_suggestion_texts,
)


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


class TestModeDetection:
    @pytest.mark.parametrize("kwargs,description", [
        (dict(before_cursor=""), "empty input"),
        (dict(before_cursor="   "), "whitespace only"),
        (dict(before_cursor="Hi"), "short input"),
        (dict(before_cursor="", current_input="This is a long paragraph"), "cursor at start of long text"),
    ])
    def test_reply_mode_inputs(self, kwargs, description):
        assert detect_mode(**kwargs) == AutocompleteMode.REPLY, description

    @pytest.mark.parametrize("kwargs,description", [
        (dict(before_cursor="x" * MODE_THRESHOLD_CHARS), "exactly at threshold"),
        (dict(before_cursor="Hello, how are you doing today?"), "long input"),
        (dict(before_cursor="abc"), "before_cursor at threshold"),
    ])
    def test_continuation_mode_inputs(self, kwargs, description):
        assert detect_mode(**kwargs) == AutocompleteMode.CONTINUATION, description


class TestSuggestionEngine:
    def test_continuation_prompt_stays_in_user_voice(self):
        system, user = build_messages(
            mode=AutocompleteMode.CONTINUATION,
            context="Text before cursor:\ni think we should ",
            num_suggestions=3,
            source_app="Codex",
        )

        assert "Write AS the user" in system
        assert "You ARE the author, not a respondent." in system
        assert "SAME voice, person, and perspective" in system
        assert "Continue writing from the cursor position as the same author" in user

    def test_reply_prompt_requires_user_side_message_not_assistant_voice(self):
        system, user = build_messages(
            mode=AutocompleteMode.REPLY,
            context=(
                "App: Codex | Channel: Codex\n\n"
                "Conversation:\n"
                "- User: can you add that\n"
                "- Codex: I can add a regression test for that.\n"
            ),
            num_suggestions=3,
            source_app="Codex",
        )

        assert "You suggest messages the user might type next." in system
        assert "Mirror how the USER actually writes, not the assistant." in system
        assert "Generate exactly 3 distinct suggestions for what the user might type next." in user
        assert "Stop at a natural endpoint." in system
        assert "Vary cadence as well as content" in system
        assert "single natural message turn" in user

    def test_debounce_blocks_rapid_requests(self, engine):
        engine._last_request_time = time.time()
        assert not engine.can_request()

    def test_debounce_allows_after_interval(self, engine):
        engine._last_request_time = time.time() - 1.0
        assert engine.can_request()

    def test_empty_input_and_empty_context_returns_empty(self, engine):
        result = engine.generate_suggestions("", "")
        assert result == []

    def test_whitespace_input_and_whitespace_context_returns_empty(self, engine):
        result = engine.generate_suggestions("   ", "   ")
        assert result == []

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_empty_input_with_context_calls_llm_reply_mode(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="Sounds good!", index=0),
        ]
        result = engine.generate_suggestions("", "some context")
        assert len(result) == 1
        assert result[0].text == "Sounds good!"
        mock_call.assert_called_once()
        # Should use reply-mode params
        _, kwargs = mock_call.call_args
        assert kwargs.get("temperature") == engine.config.reply_temperature
        assert kwargs.get("max_tokens") == engine.config.max_tokens

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_generate_continuation_mode_params(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="suggestion 1", index=0),
        ]
        result = engine.generate_suggestions(
            "Hello world this is a test", "context",
            mode=AutocompleteMode.CONTINUATION,
        )
        assert len(result) == 1
        _, kwargs = mock_call.call_args
        assert kwargs.get("temperature") == engine.config.continuation_temperature
        assert kwargs.get("max_tokens") == engine.config.max_tokens

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_generate_reply_mode_params(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="reply suggestion", index=0),
        ]
        result = engine.generate_suggestions(
            "", "context", mode=AutocompleteMode.REPLY,
        )
        assert len(result) == 1
        _, kwargs = mock_call.call_args
        assert kwargs.get("temperature") == engine.config.reply_temperature
        assert kwargs.get("max_tokens") == engine.config.max_tokens

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_mode_auto_detected_when_not_specified(self, mock_call, engine):
        mock_call.return_value = [Suggestion(text="s", index=0)]
        # Long input -> continuation
        engine.generate_suggestions("Hello world this is long", "ctx")
        _, kwargs = mock_call.call_args
        assert kwargs.get("temperature") == engine.config.continuation_temperature

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_generate_calls_llm(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="suggestion 1", index=0),
        ]

        result = engine.generate_suggestions("Hello world test", "context")
        assert len(result) == 1
        assert result[0].text == "suggestion 1"
        mock_call.assert_called_once()

    def test_postprocess_strips_repeated_prefix_and_normalizes_spacing(self):
        result = postprocess_suggestion_texts(
            [
                "just invoked it, so",
                "just invoked it, and it worked",
                "but",
            ],
            mode=AutocompleteMode.CONTINUATION,
            before_cursor="just invoked it, ",
        )

        assert result == [
            "so",
            "and it worked",
            "but",
        ]

    def test_postprocess_filters_avoided_texts(self):
        result = postprocess_suggestion_texts(
            ["it makes sense", "that seems right", "we should change it"],
            mode=AutocompleteMode.CONTINUATION,
            before_cursor="just invoked it again, can you check the logs now?",
            avoid_texts=["it makes sense", "that seems right"],
        )

        assert result == [
            "",
            "",
            " we should change it",
        ]

    def test_postprocess_filters_avoided_texts_in_reply_mode(self):
        result = postprocess_suggestion_texts(
            ["sounds good", "let's do it", "can you share the fixture names?"],
            mode=AutocompleteMode.REPLY,
            before_cursor="",
            avoid_texts=["sounds good", "let's do it"],
        )

        assert result == [
            "",
            "",
            "can you share the fixture names?",
        ]

    def test_postprocess_reply_mode_falls_back_when_all_texts_are_avoided(self):
        result = postprocess_suggestion_texts(
            ["sounds good", "let's do it"],
            mode=AutocompleteMode.REPLY,
            before_cursor="",
            avoid_texts=["sounds good", "let's do it"],
        )

        assert result == [
            "sounds good",
            "let's do it",
        ]

    def test_postprocess_reply_mode_strips_repeated_draft_prefix(self):
        result = postprocess_suggestion_texts(
            ["can we make it type them out gradually instead of all at once? maybe start with smaller batches"],
            mode=AutocompleteMode.REPLY,
            before_cursor="can we make it type them out gradually instead of all at once? ",
        )

        assert result == [
            "maybe start with smaller batches",
        ]

    def test_postprocess_reply_mode_normalizes_leading_space_against_draft(self):
        result = postprocess_suggestion_texts(
            [" what if we slow down spaces a bit"],
            mode=AutocompleteMode.REPLY,
            before_cursor="hello ",
        )

        assert result == [
            "what if we slow down spaces a bit",
        ]

    def test_postprocess_reply_mode_adds_needed_space_after_draft(self):
        result = postprocess_suggestion_texts(
            ["what if we slow down spaces a bit"],
            mode=AutocompleteMode.REPLY,
            before_cursor="hello",
        )

        assert result == [
            " what if we slow down spaces a bit",
        ]

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_continuation_postprocess_keeps_literal_completion(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="that makes sense?", index=0),
        ]

        result = engine.generate_suggestions(
            current_input="just invoked it, can you check? also, do you think ",
            context="Text before cursor:\njust invoked it, can you check? also, do you think ",
            mode=AutocompleteMode.CONTINUATION,
            before_cursor="just invoked it, can you check? also, do you think ",
            prompt_placeholder_aware=True,
        )

        assert result[0].text == "that makes sense?"

    def test_unknown_provider_returns_error_suggestion(self, config):
        """Unknown provider returns an error suggestion."""
        config.llm_provider = "unknown"
        engine = SuggestionEngine(config)
        # generate_suggestions catches exceptions and returns error suggestion
        result = engine.generate_suggestions("Hello world test", "context")
        assert len(result) == 1
        assert "error" in result[0].text.lower() or "try again" in result[0].text.lower()

    def test_unicode_suggestions_via_instructor(self):
        """Unicode text passes through the Instructor-based pipeline."""
        from autocompleter.suggestion_engine import SuggestionItem
        items = [
            SuggestionItem(text="Bonjour le monde"),
            SuggestionItem(text="\u00e9\u00e8\u00ea"),
            SuggestionItem(text="\u4f60\u597d"),
        ]
        mock_result = MagicMock()
        mock_result.suggestions = items
        engine = SuggestionEngine(Config(
            llm_provider="anthropic",
            anthropic_api_key="test-key",
        ))
        engine._client = MagicMock()
        engine._client.create.return_value = mock_result
        results = engine._call_llm("sys", "user", temperature=0.5, max_tokens=100)
        assert len(results) == 3
        assert results[0].text == "Bonjour le monde"
        assert results[2].text == "\u4f60\u597d"
