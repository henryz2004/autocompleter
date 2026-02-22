"""Tests for the suggestion engine."""

import time
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.config import Config
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    Suggestion,
    SuggestionEngine,
    detect_mode,
    MODE_THRESHOLD_CHARS,
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
        assert kwargs.get("max_tokens") == engine.config.reply_max_tokens

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
        assert kwargs.get("max_tokens") == engine.config.continuation_max_tokens

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
        assert kwargs.get("max_tokens") == engine.config.reply_max_tokens

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

    def test_unknown_provider_raises_at_client_creation(self, config):
        """Unknown provider raises when _get_client() is called."""
        config.llm_provider = "unknown"
        engine = SuggestionEngine(config)
        # generate_suggestions catches exceptions and returns []
        result = engine.generate_suggestions("Hello world test", "context")
        assert result == []

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
