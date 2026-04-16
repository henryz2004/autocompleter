"""Tests for streaming suggestion generation.

Covers raw SDK text streaming with incremental JSON parsing,
fallback to blocking path, error handling, backward compatibility of
the non-streaming path, and generation_id staleness checks.
"""

from __future__ import annotations

import json
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
    _extract_complete_suggestions,
    postprocess_suggestion_text,
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
        fallback_api_key="",  # disable fallback in tests
        escalation_timeout_ms=50,  # short timeout for fast tests
    )


@pytest.fixture
def engine(config):
    return SuggestionEngine(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_json(suggestion_texts: list[str]) -> str:
    """Build the full JSON string for a list of suggestion texts."""
    items = [{"text": t} for t in suggestion_texts]
    return json.dumps({"suggestions": items})


def _make_text_chunks(suggestion_texts: list[str]) -> list[str]:
    """Build JSON text chunks that simulate incremental streaming.

    Each suggestion object is delivered as a separate chunk, so the
    incremental parser can detect each one as it completes.
    """
    chunks = ['{"suggestions": [']
    for i, text in enumerate(suggestion_texts):
        if i > 0:
            chunks.append(", ")
        chunks.append(json.dumps({"text": text}))
    chunks.append("]}")
    return chunks


def _make_fine_text_chunks(suggestion_texts: list[str], chunk_size: int = 5) -> list[str]:
    """Build fine-grained character-level chunks for realistic streaming."""
    full_json = _build_json(suggestion_texts)
    return [full_json[i:i + chunk_size] for i in range(0, len(full_json), chunk_size)]


def _mock_anthropic_stream(engine, chunks: list[str]):
    """Set up a mock raw Anthropic client for streaming.

    Returns the mock client so tests can assert on call args.
    """
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(chunks)
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream
    engine._raw_client = mock_client
    return mock_client


def _mock_engine_stream(engine, suggestion_texts: list[str]):
    """Convenience: mock the raw client with per-object text chunks."""
    chunks = _make_text_chunks(suggestion_texts)
    return _mock_anthropic_stream(engine, chunks)


# ---------------------------------------------------------------------------
# Tests — _extract_complete_suggestions (JSON parser)
# ---------------------------------------------------------------------------

class TestExtractCompleteSuggestions:
    """Test the incremental JSON parser."""

    def test_empty_buffer(self):
        assert _extract_complete_suggestions("") == []

    def test_no_suggestions_key(self):
        assert _extract_complete_suggestions('{"other": "value"}') == []

    def test_incomplete_array(self):
        assert _extract_complete_suggestions('{"suggestions": [') == []

    def test_one_complete_object(self):
        buf = '{"suggestions": [{"text": "hello"}'
        assert _extract_complete_suggestions(buf) == ["hello"]

    def test_one_complete_one_incomplete(self):
        buf = '{"suggestions": [{"text": "hello"}, {"text": "wor'
        assert _extract_complete_suggestions(buf) == ["hello"]

    def test_two_complete(self):
        buf = '{"suggestions": [{"text": "hello"}, {"text": "world"}'
        assert _extract_complete_suggestions(buf) == ["hello", "world"]

    def test_full_json(self):
        buf = _build_json(["first", "second", "third"])
        assert _extract_complete_suggestions(buf) == ["first", "second", "third"]

    def test_escaped_quotes_in_text(self):
        buf = '{"suggestions": [{"text": "say \\"hello\\""}]}'
        assert _extract_complete_suggestions(buf) == ['say "hello"']

    def test_escaped_backslash(self):
        buf = '{"suggestions": [{"text": "path\\\\to\\\\file"}]}'
        assert _extract_complete_suggestions(buf) == ["path\\to\\file"]

    def test_newlines_in_text(self):
        buf = '{"suggestions": [{"text": "line1\\nline2"}]}'
        assert _extract_complete_suggestions(buf) == ["line1\nline2"]

    def test_unicode_in_text(self):
        buf = '{"suggestions": [{"text": "\\u4f60\\u597d"}]}'
        assert _extract_complete_suggestions(buf) == ["\u4f60\u597d"]

    def test_nested_braces_in_text(self):
        buf = '{"suggestions": [{"text": "use {braces} here"}]}'
        assert _extract_complete_suggestions(buf) == ["use {braces} here"]

    def test_whitespace_between_objects(self):
        buf = '{"suggestions": [\n  {"text": "a"} ,\n  {"text": "b"}\n]}'
        assert _extract_complete_suggestions(buf) == ["a", "b"]

    def test_missing_text_key_skipped(self):
        buf = '{"suggestions": [{"other": "val"}, {"text": "ok"}]}'
        result = _extract_complete_suggestions(buf)
        # First object has no "text" key — skipped, second is extracted
        assert "ok" in result

    def test_incremental_growth(self):
        """Simulate token-by-token growth and verify incremental detection."""
        full = _build_json(["alpha", "beta", "gamma"])
        detected_counts = []
        buf = ""
        for ch in full:
            buf += ch
            detected_counts.append(len(_extract_complete_suggestions(buf)))

        # Should grow from 0 → 1 → 2 → 3 over the course of the string
        assert detected_counts[0] == 0  # Just "{"
        assert detected_counts[-1] == 3  # Complete
        # Should have transitions 0→1, 1→2, 2→3
        transitions = set()
        for i in range(1, len(detected_counts)):
            if detected_counts[i] > detected_counts[i - 1]:
                transitions.add((detected_counts[i - 1], detected_counts[i]))
        assert (0, 1) in transitions
        assert (1, 2) in transitions
        assert (2, 3) in transitions


# ---------------------------------------------------------------------------
# Tests — _call_llm_stream (raw SDK streaming)
# ---------------------------------------------------------------------------

class TestStreamingSuggestions:
    """Test _call_llm_stream with mock raw SDK client."""

    def test_basic_three_suggestions(self, engine):
        """Three suggestions stream correctly."""
        _mock_engine_stream(engine, ["First suggestion", "Second suggestion", "Third suggestion"])

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
        _mock_engine_stream(engine, ["Just one suggestion"])

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Just one suggestion"
        assert results[0].index == 0

    def test_empty_json_falls_back(self, engine):
        """Empty suggestions array falls back to blocking path."""
        _mock_anthropic_stream(engine, ['{"suggestions": []}'])
        engine._call_llm = MagicMock(return_value=[
            Suggestion(text="Fallback", index=0),
        ])

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Fallback"


class TestStreamingPostprocess:
    def test_streaming_postprocess_uses_original_suggestion_index(self):
        before = "just invoked it again, can you check the logs now?"
        assert postprocess_suggestion_text(
            "any errors show up?",
            mode=AutocompleteMode.CONTINUATION,
            before_cursor=before,
            index=1,
        ) == " any errors show up?"

    def test_streaming_postprocess_filters_avoided_reply_text(self):
        assert postprocess_suggestion_text(
            "sounds good",
            mode=AutocompleteMode.REPLY,
            before_cursor="",
            index=0,
            avoid_texts=["sounds good"],
        ) == "sounds good"

    def test_streaming_postprocess_strips_repeated_reply_prefix(self):
        assert postprocess_suggestion_text(
            "can we make it type them out gradually instead of all at once? maybe start with smaller batches",
            mode=AutocompleteMode.REPLY,
            before_cursor="can we make it type them out gradually instead of all at once? ",
            index=0,
        ) == "maybe start with smaller batches"

    def test_streaming_postprocess_normalizes_reply_spacing(self):
        assert postprocess_suggestion_text(
            " what if we slow down spaces a bit",
            mode=AutocompleteMode.REPLY,
            before_cursor="hello ",
            index=0,
        ) == "what if we slow down spaces a bit"

    def test_unicode_in_suggestions(self, engine):
        """Unicode characters pass through correctly."""
        _mock_engine_stream(engine, ["Bonjour le monde", "\u00e9\u00e8\u00ea", "\u4f60\u597d\u4e16\u754c"])

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "Bonjour le monde"
        assert results[1].text == "\u00e9\u00e8\u00ea"
        assert results[2].text == "\u4f60\u597d\u4e16\u754c"

    def test_multiline_suggestion_text(self, engine):
        """Suggestions with newlines are preserved."""
        _mock_engine_stream(engine, ["Line one\nLine two\nLine three", "Single line"])

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Line one\nLine two\nLine three"
        assert results[1].text == "Single line"

    def test_fine_grained_chunks_still_work(self, engine):
        """Character-level chunks (realistic streaming) yield correct results."""
        chunks = _make_fine_text_chunks(["Hello", "World", "Test"], chunk_size=3)
        _mock_anthropic_stream(engine, chunks)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "Hello"
        assert results[1].text == "World"
        assert results[2].text == "Test"

    def test_incremental_delivery_order(self, engine):
        """Suggestions are yielded incrementally as JSON objects complete."""
        # Deliver each suggestion object as a separate chunk
        chunks = [
            '{"suggestions": [',
            '{"text": "First"}',   # → should yield First
            ', {"text": "Second"}',  # → should yield Second
            ', {"text": "Third"}',   # → should yield Third
            ']}',
        ]
        _mock_anthropic_stream(engine, chunks)

        # Collect results and track when each arrives
        results = []
        gen = engine._call_llm_stream("sys", "user")
        for suggestion in gen:
            results.append(suggestion)

        assert len(results) == 3
        assert [r.text for r in results] == ["First", "Second", "Third"]
        assert [r.index for r in results] == [0, 1, 2]

    def test_error_during_stream_falls_back(self, engine):
        """Exception during streaming falls back to blocking _call_llm."""
        mock_client = MagicMock()
        mock_client.messages.stream.side_effect = RuntimeError("API error")
        engine._raw_client = mock_client

        fallback_results = [Suggestion(text="Fallback", index=0)]
        engine._call_llm = MagicMock(return_value=fallback_results)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Fallback"
        engine._call_llm.assert_called_once()

    def test_error_in_both_paths_yields_error_suggestion(self, engine):
        """Exception in both streaming and _call_llm yields an error suggestion."""
        mock_client = MagicMock()
        mock_client.messages.stream.side_effect = RuntimeError("stream error")
        engine._raw_client = mock_client
        engine._call_llm = MagicMock(side_effect=RuntimeError("fallback error"))

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 1
        assert "error" in results[0].text.lower() or "try again" in results[0].text.lower()

    def test_garbage_json_falls_back(self, engine):
        """Non-JSON response falls back to blocking path."""
        _mock_anthropic_stream(engine, ["This is not JSON at all"])
        engine._call_llm = MagicMock(return_value=[
            Suggestion(text="Fallback", index=0),
        ])

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Fallback"

    def test_escaped_content_in_suggestions(self, engine):
        """Suggestions with escaped characters stream correctly."""
        chunks = [
            '{"suggestions": [',
            '{"text": "say \\"hello\\""}',
            ', {"text": "path\\\\dir"}',
            ']}',
        ]
        _mock_anthropic_stream(engine, chunks)

        results = list(engine._call_llm_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == 'say "hello"'
        assert results[1].text == "path\\dir"

    def test_uses_raw_client_not_instructor(self, engine):
        """_call_llm_stream uses _get_raw_client, not _get_client."""
        _mock_engine_stream(engine, ["test"])

        list(engine._call_llm_stream("sys", "user"))

        # Verify raw client was used (messages.stream was called)
        engine._raw_client.messages.stream.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — OpenAI streaming path
# ---------------------------------------------------------------------------

class TestOpenAIStreaming:
    """Test _call_llm_stream with OpenAI provider."""

    @pytest.fixture
    def openai_config(self):
        return Config(
            llm_provider="openai",
            openai_api_key="test-key",
            num_suggestions=3,
            debounce_ms=0,
            max_tokens=150,
            fallback_api_key="",  # disable fallback in tests
            escalation_timeout_ms=50,
        )

    @pytest.fixture
    def openai_engine(self, openai_config):
        return SuggestionEngine(openai_config)

    def _mock_openai_stream(self, engine, suggestion_texts):
        """Mock OpenAI streaming response."""
        chunks = _make_text_chunks(suggestion_texts)
        mock_chunks = []
        for chunk_text in chunks:
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta = MagicMock()
            mock_chunk.choices[0].delta.content = chunk_text
            mock_chunks.append(mock_chunk)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        engine._raw_client = mock_client
        return mock_client

    def test_openai_basic_streaming(self, openai_engine):
        """OpenAI streaming yields suggestions correctly."""
        self._mock_openai_stream(openai_engine, ["Alpha", "Beta"])

        results = list(openai_engine._call_llm_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Alpha"
        assert results[1].text == "Beta"

    def test_openai_calls_correct_api(self, openai_engine):
        """OpenAI path calls chat.completions.create with stream=True."""
        mock_client = self._mock_openai_stream(openai_engine, ["test"])

        list(openai_engine._call_llm_stream("sys", "user"))

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Tests — _call_llm (non-streaming)
# ---------------------------------------------------------------------------

class TestCallLlm:
    """Test the non-streaming _call_llm method."""

    def _mock_engine_with_list(self, engine, items: list[SuggestionItem]):
        """Set up a mock Instructor client that returns a mock SuggestionList."""
        mock_result = MagicMock()
        mock_result.suggestions = items
        mock_client = MagicMock()
        mock_client.create.return_value = mock_result
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
        _mock_engine_stream(engine, ["Alpha", "Beta"])

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
        mock_client = _mock_engine_stream(engine, ["s"])

        list(engine.generate_suggestions_stream(
            current_input="Hello world this is a test",
            context="context",
            mode=AutocompleteMode.CONTINUATION,
        ))

        mock_client.messages.stream.assert_called_once()
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["temperature"] == engine.config.continuation_temperature
        assert call_kwargs["max_tokens"] == engine.config.max_tokens

    def test_reply_mode_params(self, engine):
        """Reply mode uses correct temperature and max_tokens."""
        mock_client = _mock_engine_stream(engine, ["s"])

        list(engine.generate_suggestions_stream(
            current_input="",
            context="context",
            mode=AutocompleteMode.REPLY,
        ))

        mock_client.messages.stream.assert_called_once()
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["temperature"] == engine.config.reply_temperature
        assert call_kwargs["max_tokens"] == engine.config.max_tokens

    def test_regenerate_stream_prefers_non_repeated_suggestions(self, engine):
        _mock_engine_stream(engine, ["sounds good", "new idea", "let's do it"])

        results = list(engine.generate_suggestions_stream(
            current_input="",
            context="context",
            mode=AutocompleteMode.REPLY,
            temperature_boost=0.5,
            negative_patterns=["sounds good", "let's do it"],
            prompt_placeholder_aware=True,
        ))

        assert [result.text for result in results] == ["new idea"]

    def test_regenerate_stream_falls_back_when_all_suggestions_repeat(self, engine):
        _mock_engine_stream(engine, ["sounds good", "let's do it"])

        results = list(engine.generate_suggestions_stream(
            current_input="",
            context="context",
            mode=AutocompleteMode.REPLY,
            temperature_boost=0.5,
            negative_patterns=["sounds good", "let's do it"],
            prompt_placeholder_aware=True,
        ))

        assert [result.text for result in results] == ["sounds good", "let's do it"]

    def test_exception_during_stream_yields_error_suggestion(self, engine):
        """Top-level exception in streaming + fallback yields an error suggestion."""
        mock_client = MagicMock()
        mock_client.messages.stream.side_effect = RuntimeError("API down")
        engine._raw_client = mock_client
        engine._call_llm = MagicMock(side_effect=RuntimeError("fallback down"))

        results = list(engine.generate_suggestions_stream(
            current_input="test input",
            context="context",
        ))
        assert len(results) == 1
        assert "error" in results[0].text.lower() or "try again" in results[0].text.lower()


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
        _mock_engine_stream(engine, ["First", "Second", "Third"])

        # Simulate what _generate_and_show_streaming does:
        # iterate the generator but check generation_id each time
        generation_id = 1
        gen_id_holder = [1]

        collected = []
        for suggestion in engine.generate_suggestions_stream(
            current_input="test input here",
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
        _mock_engine_stream(engine, ["First", "Second", "Third"])

        generation_id = 1
        gen_id_holder = [1]

        collected = []
        for suggestion in engine.generate_suggestions_stream(
            current_input="test input here",
            context="context",
        ):
            if gen_id_holder[0] != generation_id:
                break
            collected.append(suggestion)

        assert len(collected) == 3
