"""Tests for the suggestion engine."""

import time
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.config import Config
from autocompleter.suggestion_engine import Suggestion, SuggestionEngine


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


class TestSuggestionEngine:
    def test_parse_suggestions(self):
        raw = "First suggestion---SUGGESTION---Second suggestion---SUGGESTION---Third"
        results = SuggestionEngine._parse_suggestions(raw)
        assert len(results) == 3
        assert results[0].text == "First suggestion"
        assert results[1].text == "Second suggestion"
        assert results[2].text == "Third"

    def test_parse_suggestions_handles_whitespace(self):
        raw = "  First  \n---SUGGESTION---\n  Second  "
        results = SuggestionEngine._parse_suggestions(raw)
        assert len(results) == 2
        assert results[0].text == "First"
        assert results[1].text == "Second"

    def test_parse_suggestions_empty(self):
        results = SuggestionEngine._parse_suggestions("")
        assert len(results) == 0

    def test_parse_suggestions_single(self):
        results = SuggestionEngine._parse_suggestions("Just one suggestion")
        assert len(results) == 1
        assert results[0].text == "Just one suggestion"

    def test_debounce_blocks_rapid_requests(self, engine):
        engine._last_request_time = time.time()
        assert not engine.can_request()

    def test_debounce_allows_after_interval(self, engine):
        engine._last_request_time = time.time() - 1.0
        assert engine.can_request()

    def test_empty_input_returns_empty(self, engine):
        result = engine.generate_suggestions("", "some context")
        assert result == []

    def test_whitespace_input_returns_empty(self, engine):
        result = engine.generate_suggestions("   ", "some context")
        assert result == []

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_anthropic")
    def test_generate_calls_anthropic(self, mock_call, engine):
        mock_call.return_value = [
            Suggestion(text="suggestion 1", index=0),
        ]

        result = engine.generate_suggestions("Hello", "context")
        assert len(result) == 1
        assert result[0].text == "suggestion 1"
        mock_call.assert_called_once()

    @patch("autocompleter.suggestion_engine.SuggestionEngine._call_openai")
    def test_generate_calls_openai(self, mock_call, config):
        config.llm_provider = "openai"
        config.openai_api_key = "test-key"
        engine = SuggestionEngine(config)

        mock_call.return_value = [
            Suggestion(text="openai suggestion", index=0),
        ]

        result = engine.generate_suggestions("Hello", "context")
        assert len(result) == 1
        mock_call.assert_called_once()

    def test_unknown_provider_returns_empty(self, config):
        config.llm_provider = "unknown"
        engine = SuggestionEngine(config)
        result = engine.generate_suggestions("Hello", "context")
        assert result == []
