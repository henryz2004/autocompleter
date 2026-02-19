"""Tests for streaming suggestion generation.

Covers delimiter-based parsing from simulated streamed chunks,
partial suggestions on stream completion, error handling, empty
suggestion filtering, backward compatibility of the non-streaming
path, and generation_id staleness checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from autocompleter.config import Config
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    Suggestion,
    SuggestionEngine,
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
def openai_config():
    return Config(
        llm_provider="openai",
        openai_api_key="test-key",
        num_suggestions=3,
        debounce_ms=0,
        max_tokens=150,
    )


@pytest.fixture
def engine(config):
    return SuggestionEngine(config)


@pytest.fixture
def openai_engine(openai_config):
    return SuggestionEngine(openai_config)


# ---------------------------------------------------------------------------
# Helpers — simulate streaming responses
# ---------------------------------------------------------------------------

class FakeAnthropicStream:
    """Simulates anthropic client.messages.stream() context manager.

    Accepts a list of text chunks that will be yielded by text_stream.
    """

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def text_stream(self):
        for chunk in self._chunks:
            yield chunk


class FakeAnthropicStreamError:
    """Simulates an Anthropic stream that raises an error mid-way."""

    def __init__(self, chunks: list[str], error_after: int):
        self._chunks = chunks
        self._error_after = error_after

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def text_stream(self):
        for i, chunk in enumerate(self._chunks):
            if i >= self._error_after:
                raise ConnectionError("Stream interrupted")
            yield chunk


@dataclass
class FakeOpenAIDelta:
    content: Optional[str] = None


@dataclass
class FakeOpenAIChoice:
    delta: FakeOpenAIDelta


@dataclass
class FakeOpenAIChunk:
    choices: list[FakeOpenAIChoice]


def make_openai_chunks(texts: list[str]) -> list[FakeOpenAIChunk]:
    """Build a list of OpenAI-style streaming chunks."""
    return [
        FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content=t))])
        for t in texts
    ]


# ---------------------------------------------------------------------------
# Tests — Anthropic streaming
# ---------------------------------------------------------------------------

class TestAnthropicStream:
    """Test _call_anthropic_stream with simulated chunks."""

    def test_basic_three_suggestions(self, engine):
        """Three suggestions separated by delimiters stream correctly."""
        chunks = [
            "First suggestion",
            "---SUGGESTION---",
            "Second suggestion",
            "---SUGGESTION---",
            "Third suggestion",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "First suggestion"
        assert results[0].index == 0
        assert results[1].text == "Second suggestion"
        assert results[1].index == 1
        assert results[2].text == "Third suggestion"
        assert results[2].index == 2

    def test_delimiter_split_across_chunks(self, engine):
        """Delimiter arriving across multiple chunks is handled correctly."""
        # "---SUGGESTION---" split as "---SUGG" + "ESTION---"
        chunks = [
            "Hello world",
            "---SUGG",
            "ESTION---",
            "Next suggestion",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Hello world"
        assert results[1].text == "Next suggestion"

    def test_delimiter_in_single_chunk_with_text(self, engine):
        """Delimiter embedded in a single chunk with surrounding text."""
        chunks = [
            "First---SUGGESTION---Second---SUGGESTION---Third",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "First"
        assert results[1].text == "Second"
        assert results[2].text == "Third"

    def test_empty_suggestions_skipped(self, engine):
        """Empty/whitespace-only suggestions are not yielded."""
        chunks = [
            "---SUGGESTION---",  # empty before first delimiter
            "   ",               # whitespace-only
            "---SUGGESTION---",  # whitespace between delimiters
            "Real suggestion",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Real suggestion"

    def test_partial_suggestion_on_stream_end(self, engine):
        """Text remaining after the last delimiter is yielded as a suggestion."""
        chunks = [
            "First",
            "---SUGGESTION---",
            "Second (partial, no trailing delimiter)",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "First"
        assert results[1].text == "Second (partial, no trailing delimiter)"

    def test_stream_error_yields_partial_results(self, engine):
        """If the stream errors mid-way, whatever was buffered is yielded."""
        chunks = [
            "Complete suggestion",
            "---SUGGESTION---",
            "Partial before err",
            "WILL NOT ARRIVE",  # error before this chunk
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStreamError(
            chunks, error_after=3,
        )
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Complete suggestion"
        assert results[1].text == "Partial before err"

    def test_stream_error_at_start_yields_nothing(self, engine):
        """If the stream errors before any text, nothing is yielded."""
        chunks = ["some text"]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStreamError(
            chunks, error_after=0,
        )
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 0

    def test_empty_stream(self, engine):
        """Empty stream yields nothing."""
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream([])
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 0

    def test_single_suggestion_no_delimiter(self, engine):
        """A single suggestion with no delimiter is yielded at stream end."""
        chunks = ["Just one suggestion"]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Just one suggestion"
        assert results[0].index == 0

    def test_whitespace_around_delimiter(self, engine):
        """Whitespace around delimiters is stripped from suggestion text."""
        chunks = [
            "  First with spaces  ",
            "\n---SUGGESTION---\n",
            "  Second with spaces  ",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "First with spaces"
        assert results[1].text == "Second with spaces"

    def test_character_by_character_streaming(self, engine):
        """Delimiter works even when text arrives one character at a time."""
        full_text = "Hi---SUGGESTION---Bye"
        chunks = list(full_text)  # one char per chunk
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Hi"
        assert results[1].text == "Bye"


# ---------------------------------------------------------------------------
# Tests — OpenAI streaming
# ---------------------------------------------------------------------------

class TestOpenAIStream:
    """Test _call_openai_stream with simulated chunks."""

    def test_basic_three_suggestions(self, openai_engine):
        """Three suggestions stream correctly via OpenAI."""
        chunks = make_openai_chunks([
            "First suggestion",
            "---SUGGESTION---",
            "Second suggestion",
            "---SUGGESTION---",
            "Third suggestion",
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        openai_engine._openai_client = mock_client

        results = list(openai_engine._call_openai_stream("sys", "user"))
        assert len(results) == 3
        assert results[0].text == "First suggestion"
        assert results[1].text == "Second suggestion"
        assert results[2].text == "Third suggestion"

    def test_delimiter_split_across_chunks(self, openai_engine):
        """Delimiter split across chunks works for OpenAI."""
        chunks = make_openai_chunks([
            "Hello world",
            "---SUGG",
            "ESTION---",
            "Next suggestion",
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        openai_engine._openai_client = mock_client

        results = list(openai_engine._call_openai_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Hello world"
        assert results[1].text == "Next suggestion"

    def test_empty_suggestions_skipped(self, openai_engine):
        """Empty suggestions are skipped for OpenAI streaming."""
        chunks = make_openai_chunks([
            "---SUGGESTION---",
            "  \n  ",
            "---SUGGESTION---",
            "Real one",
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        openai_engine._openai_client = mock_client

        results = list(openai_engine._call_openai_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Real one"

    def test_stream_error_yields_partial(self, openai_engine):
        """OpenAI stream error mid-way yields what we have so far."""
        def error_generator():
            yield FakeOpenAIChunk(
                choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="Complete"))]
            )
            yield FakeOpenAIChunk(
                choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="---SUGGESTION---"))]
            )
            yield FakeOpenAIChunk(
                choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="Partial"))]
            )
            raise ConnectionError("Stream interrupted")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = error_generator()
        openai_engine._openai_client = mock_client

        results = list(openai_engine._call_openai_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Complete"
        assert results[1].text == "Partial"

    def test_none_content_chunks_ignored(self, openai_engine):
        """Chunks with None content are gracefully skipped."""
        chunks = [
            FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="Hello"))]),
            FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content=None))]),
            FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="---SUGGESTION---"))]),
            FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="World"))]),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        openai_engine._openai_client = mock_client

        results = list(openai_engine._call_openai_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[1].text == "World"

    def test_empty_choices_ignored(self, openai_engine):
        """Chunks with empty choices list are gracefully skipped."""
        chunks = [
            FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content="Hello"))]),
            FakeOpenAIChunk(choices=[]),
            FakeOpenAIChunk(choices=[FakeOpenAIChoice(delta=FakeOpenAIDelta(content=" World"))]),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        openai_engine._openai_client = mock_client

        results = list(openai_engine._call_openai_stream("sys", "user"))
        assert len(results) == 1
        assert results[0].text == "Hello World"


# ---------------------------------------------------------------------------
# Tests — generate_suggestions_stream (high-level)
# ---------------------------------------------------------------------------

class TestGenerateSuggestionsStream:
    """Test the public generate_suggestions_stream method."""

    def test_streams_anthropic(self, engine):
        """generate_suggestions_stream yields from Anthropic provider."""
        chunks = [
            "Alpha",
            "---SUGGESTION---",
            "Beta",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine.generate_suggestions_stream(
            current_input="Hello world test",
            context="some context",
        ))
        assert len(results) == 2
        assert results[0].text == "Alpha"
        assert results[1].text == "Beta"

    def test_streams_openai(self, openai_engine):
        """generate_suggestions_stream yields from OpenAI provider."""
        chunks = make_openai_chunks([
            "Alpha",
            "---SUGGESTION---",
            "Beta",
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        openai_engine._openai_client = mock_client

        results = list(openai_engine.generate_suggestions_stream(
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

    def test_unknown_provider_returns_empty(self, config):
        """Unknown LLM provider yields nothing."""
        config.llm_provider = "unknown"
        engine = SuggestionEngine(config)
        results = list(engine.generate_suggestions_stream(
            current_input="test",
            context="context",
        ))
        assert results == []

    def test_continuation_mode_params(self, engine):
        """Continuation mode uses correct temperature and max_tokens."""
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(["suggestion"])
        engine._anthropic_client = mock_client

        list(engine.generate_suggestions_stream(
            current_input="Hello world this is a test",
            context="context",
            mode=AutocompleteMode.CONTINUATION,
        ))

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["temperature"] == engine.config.continuation_temperature
        assert call_kwargs["max_tokens"] == engine.config.continuation_max_tokens

    def test_reply_mode_params(self, engine):
        """Reply mode uses correct temperature and max_tokens."""
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(["suggestion"])
        engine._anthropic_client = mock_client

        list(engine.generate_suggestions_stream(
            current_input="",
            context="context",
            mode=AutocompleteMode.REPLY,
        ))

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["temperature"] == engine.config.reply_temperature
        assert call_kwargs["max_tokens"] == engine.config.reply_max_tokens

    def test_exception_during_stream_yields_nothing(self, engine):
        """Top-level exception in the provider call yields nothing gracefully."""
        mock_client = MagicMock()
        mock_client.messages.stream.side_effect = RuntimeError("API down")
        engine._anthropic_client = mock_client

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

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_anthropic")
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

    def test_parse_suggestions_unchanged(self):
        """Static _parse_suggestions method still works."""
        raw = "First---SUGGESTION---Second---SUGGESTION---Third"
        results = SuggestionEngine._parse_suggestions(raw)
        assert len(results) == 3
        assert results[0].text == "First"
        assert results[1].text == "Second"
        assert results[2].text == "Third"


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
        chunks = [
            "First",
            "---SUGGESTION---",
            "Second",
            "---SUGGESTION---",
            "Third",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        # Simulate what _generate_and_show_streaming does:
        # iterate the generator but check generation_id each time
        generation_id = 1
        current_generation_id = 1  # mutable via list for closure
        gen_id_holder = [current_generation_id]

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
        chunks = [
            "First",
            "---SUGGESTION---",
            "Second",
            "---SUGGESTION---",
            "Third",
        ]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

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


# ---------------------------------------------------------------------------
# Tests — edge cases with delimiter splitting
# ---------------------------------------------------------------------------

class TestDelimiterEdgeCases:
    """Edge cases for delimiter parsing across chunk boundaries."""

    def test_delimiter_one_char_at_a_time(self, engine):
        """Delimiter arriving one character at a time."""
        text = "A---SUGGESTION---B"
        chunks = list(text)
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "A"
        assert results[1].text == "B"

    def test_multiple_delimiters_in_one_chunk(self, engine):
        """Multiple delimiters in a single chunk."""
        chunks = ["A---SUGGESTION---B---SUGGESTION---C---SUGGESTION---D"]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 4
        texts = [r.text for r in results]
        assert texts == ["A", "B", "C", "D"]

    def test_trailing_delimiter_no_text_after(self, engine):
        """Trailing delimiter with no text after it yields nothing extra."""
        chunks = ["First---SUGGESTION---Second---SUGGESTION---"]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 2
        assert results[0].text == "First"
        assert results[1].text == "Second"

    def test_only_delimiters_no_content(self, engine):
        """Stream with only delimiters and no real content yields nothing."""
        chunks = ["---SUGGESTION------SUGGESTION------SUGGESTION---"]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 0

    def test_partial_delimiter_at_end_of_stream(self, engine):
        """Partial delimiter at end of stream is treated as regular text."""
        chunks = ["Suggestion text---SUGGES"]
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
        engine._anthropic_client = mock_client

        results = list(engine._call_anthropic_stream("sys", "user"))
        assert len(results) == 1
        # The partial delimiter text is part of the suggestion
        assert results[0].text == "Suggestion text---SUGGES"

    def test_delimiter_split_every_possible_way(self, engine):
        """Test splitting the delimiter at every possible position."""
        delimiter = "---SUGGESTION---"
        for split_pos in range(1, len(delimiter)):
            left = delimiter[:split_pos]
            right = delimiter[split_pos:]
            chunks = ["Before", left, right, "After"]

            mock_client = MagicMock()
            mock_client.messages.stream.return_value = FakeAnthropicStream(chunks)
            engine._anthropic_client = mock_client

            results = list(engine._call_anthropic_stream("sys", "user"))
            assert len(results) == 2, (
                f"Failed with delimiter split at position {split_pos}: "
                f"{left!r} + {right!r}"
            )
            assert results[0].text == "Before"
            assert results[1].text == "After"
