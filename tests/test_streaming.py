"""Tests for streaming suggestion generation.

Covers Instructor-based structured output streaming via create_iterable,
error handling, empty suggestion filtering, backward compatibility of
the non-streaming path, and generation_id staleness checks.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.config import Config
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    Suggestion,
    SuggestionEngine,
    SuggestionItem,
    SuggestionList,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        num_suggestions=3,
        debounce_ms=0,  # disable debounce for testing
        max_tokens=150,
    )


@pytest.fixture
def engine(config):
    return SuggestionEngine(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine_stream(engine, items: list[SuggestionItem]):
    """Set up a mock Instructor client for streaming (create_partial).

    Simulates create_partial by yielding progressively larger SuggestionList
    snapshots — one new item per partial update.
    """
    def partial_generator():
        for i in range(len(items)):
            yield SuggestionList(suggestions=items[:i + 1])

    mock_client = MagicMock()
    mock_client.create_partial.return_value = partial_generator()
    engine._client = mock_client
    return mock_client


# ---------------------------------------------------------------------------
# Tests — Instructor streaming (provider-agnostic)
# ---------------------------------------------------------------------------

class TestStreamingSuggestions:
    """Test _call_llm_stream with mock Instructor client."""

    def test_basic_three_suggestions(self, engine):
        """Three suggestions stream correctly."""
        items = [
            SuggestionItem(text="First suggestion"),
            SuggestionItem(text="Second suggestion"),
            SuggestionItem(text="Third suggestion"),
        ]
        _mock_engine_stream(engine, items)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "First suggestion"
        assert results[0].index == 0
        assert results[1].text == "Second suggestion"
        assert results[1].index == 1
        assert results[2].text == "Third suggestion"
        assert results[2].index == 2

    def test_single_suggestion(self, engine):
        """A single suggestion is yielded correctly."""
        items = [SuggestionItem(text="Just one suggestion")]
        _mock_engine_stream(engine, items)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Just one suggestion"
        assert results[0].index == 0

    def test_empty_stream(self, engine):
        """Empty stream yields nothing."""
        _mock_engine_stream(engine, [])

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 0

    def test_unicode_in_suggestions(self, engine):
        """Unicode characters pass through correctly."""
        items = [
            SuggestionItem(text="Bonjour le monde"),
            SuggestionItem(text="\u00e9\u00e8\u00ea"),
            SuggestionItem(text="\u4f60\u597d\u4e16\u754c"),
        ]
        _mock_engine_stream(engine, items)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "Bonjour le monde"
        assert results[1].text == "\u00e9\u00e8\u00ea"
        assert results[2].text == "\u4f60\u597d\u4e16\u754c"

    def test_multiline_suggestion_text(self, engine):
        """Suggestions with newlines are preserved."""
        items = [
            SuggestionItem(text="Line one\nLine two\nLine three"),
            SuggestionItem(text="Single line"),
        ]
        _mock_engine_stream(engine, items)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Line one\nLine two\nLine three"
        assert results[1].text == "Single line"

    def test_error_during_iteration_raises(self, engine):
        """Exception during create_partial propagates."""
        mock_client = MagicMock()
        mock_client.create_partial.side_effect = RuntimeError("API error")
        engine._client = mock_client

        with pytest.raises(RuntimeError, match="API error"):
            list(engine._call_llm_stream("sys", "user"))


# ---------------------------------------------------------------------------
# Tests — _call_llm (non-streaming)
# ---------------------------------------------------------------------------

class TestCallLlm:
    """Test the non-streaming _call_llm method."""

    def _mock_engine_with_list(self, engine, items: list[SuggestionItem]):
        """Set up a mock Instructor client that returns a SuggestionList."""
        mock_client = MagicMock()
        mock_client.create.return_value = SuggestionList(suggestions=items)
        engine._client = mock_client
        return mock_client

    def test_basic_suggestions(self, engine):
        """_call_llm returns parsed Suggestion objects."""
        items = [
            SuggestionItem(text="First"),
            SuggestionItem(text="Second"),
        ]
        self._mock_engine_with_list(engine, items)

        results = engine._call_llm("sys", "user", temperature=0.5, max_tokens=100)
        assert len(results) == 2
        assert results[0].text == "First"
        assert results[0].index == 0
        assert results[1].text == "Second"
        assert results[1].index == 1

    def test_empty_results(self, engine):
        """Empty suggestions list returns empty list."""
        self._mock_engine_with_list(engine, [])
        results = engine._call_llm("sys", "user", temperature=0.5, max_tokens=100)
        assert results == []

    def test_passes_correct_params(self, engine):
        """Verify create receives correct parameters."""
        mock_client = self._mock_engine_with_list(engine, [])

        engine._call_llm("system prompt", "user message", temperature=0.7, max_tokens=200)

        mock_client.create.assert_called_once()
        call_kwargs = mock_client.create.call_args[1]
        assert call_kwargs["response_model"] is SuggestionList
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 200
        msgs = call_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "system prompt"}
        assert msgs[1] == {"role": "user", "content": "user message"}


# ---------------------------------------------------------------------------
# Tests — generate_suggestions_stream (high-level)
# ---------------------------------------------------------------------------

class TestGenerateSuggestionsStream:
    """Test the public generate_suggestions_stream method."""

    def test_streams_suggestions(self, engine):
        """generate_suggestions_stream yields suggestions."""
        items = [
            SuggestionItem(text="Alpha"),
            SuggestionItem(text="Beta"),
        ]
        _mock_engine_stream(engine, items)

        results = list(engine.generate_suggestions_stream(
            current_input="Hello world test",
            context="some context",
        ))
        assert len(results) == 2
        assert results[0].text == "Alpha"
        assert results[1].text == "Beta"

    def test_debounce_blocks_stream(self, config):
        """Debounce prevents streaming requests when too soon."""
        config.debounce_ms = 5000
        engine = SuggestionEngine(config)
        engine._last_request_time = time.time()

        results = list(engine.generate_suggestions_stream(
            current_input="test input",
            context="context",
        ))
        assert results == []

    def test_empty_input_and_context_returns_empty(self, engine):
        """Empty input and empty context yields nothing."""
        results = list(engine.generate_suggestions_stream(
            current_input="",
            context="",
        ))
        assert results == []

    def test_whitespace_only_returns_empty(self, engine):
        """Whitespace-only input and context yields nothing."""
        results = list(engine.generate_suggestions_stream(
            current_input="   ",
            context="   ",
        ))
        assert results == []

    def test_continuation_mode_params(self, engine):
        """Continuation mode uses correct temperature and max_tokens."""
        mock_client = _mock_engine_stream(engine, [SuggestionItem(text="s")])

        list(engine.generate_suggestions_stream(
            current_input="Hello world this is a test",
            context="context",
            mode=AutocompleteMode.CONTINUATION,
        ))

        call_kwargs = mock_client.create_partial.call_args[1]
        assert call_kwargs["temperature"] == engine.config.continuation_temperature
        assert call_kwargs["max_tokens"] == engine.config.continuation_max_tokens

    def test_reply_mode_params(self, engine):
        """Reply mode uses correct temperature and max_tokens."""
        mock_client = _mock_engine_stream(engine, [SuggestionItem(text="s")])

        list(engine.generate_suggestions_stream(
            current_input="",
            context="context",
            mode=AutocompleteMode.REPLY,
        ))

        call_kwargs = mock_client.create_partial.call_args[1]
        assert call_kwargs["temperature"] == engine.config.reply_temperature
        assert call_kwargs["max_tokens"] == engine.config.reply_max_tokens

    def test_exception_during_stream_yields_nothing(self, engine):
        """Top-level exception in the provider call yields nothing gracefully."""
        mock_client = MagicMock()
        mock_client.create_partial.side_effect = RuntimeError("API down")
        engine._client = mock_client

        results = list(engine.generate_suggestions_stream(
            current_input="test input",
            context="context",
        ))
        assert results == []


# ---------------------------------------------------------------------------
# Tests — backward compatibility (non-streaming path)
# ---------------------------------------------------------------------------

class TestNonStreamingBackwardCompatibility:
    """Ensure the original generate_suggestions still works unchanged."""

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_llm")
    def test_non_streaming_still_works(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="suggestion 1", index=0),
            Suggestion(text="suggestion 2", index=1),
        ]
        result = engine.generate_suggestions("Hello world test", "context")
        assert len(result) == 2
        assert result[0].text == "suggestion 1"
        assert result[1].text == "suggestion 2"
        mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — generation_id staleness check in app streaming path
# ---------------------------------------------------------------------------

class TestGenerationIdStaleness:
    """Test that stale streams are abandoned when generation_id changes.

    We test this by exercising the generate_suggestions_stream generator
    and simulating what _generate_and_show_streaming does with the
    generation_id check.
    """

    def test_stale_stream_abandoned(self, engine):
        """Simulates the app abandoning a stream when generation_id changes."""
        items = [
            SuggestionItem(text="First"),
            SuggestionItem(text="Second"),
            SuggestionItem(text="Third"),
        ]
        _mock_engine_stream(engine, items)

        # Simulate what _generate_and_show_streaming does:
        # iterate the generator but check generation_id each time
        generation_id = 1
        gen_id_holder = [1]

        collected = []
        for suggestion in engine.generate_suggestions_stream(
            current_input="test",
            context="context",
        ):
            if gen_id_holder[0] != generation_id:
                break  # Stale — abandon
            collected.append(suggestion)
            # After the first suggestion, simulate a new trigger
            if len(collected) == 1:
                gen_id_holder[0] = 2  # Newer generation

        # Only the first suggestion should have been collected
        assert len(collected) == 1
        assert collected[0].text == "First"

    def test_current_stream_not_abandoned(self, engine):
        """When generation_id stays the same, all suggestions are collected."""
        items = [
            SuggestionItem(text="First"),
            SuggestionItem(text="Second"),
            SuggestionItem(text="Third"),
        ]
        _mock_engine_stream(engine, items)

        generation_id = 1
        gen_id_holder = [1]

        collected = []
        for suggestion in engine.generate_suggestions_stream(
            current_input="test",
            context="context",
        ):
            if gen_id_holder[0] != generation_id:
                break
            collected.append(suggestion)

        assert len(collected) == 3
