"""Tests for multi-line and block suggestion support.

Tests cover:
- Text height measurement with single-line, multi-line, and wrapping text
- Suggestion.preview generation (truncation with ellipsis)
- Overlay height calculation with mixed single/multi-line suggestions
- max_suggestion_height clamping
- Partial accept logic (first sentence/line extraction)
- Overlay handling of 0, 1, 2, 3 suggestions of varying heights
"""

from __future__ import annotations

import pytest

from autocompleter.app import Autocompleter
from autocompleter.overlay import (
    OverlayConfig,
    _compute_item_heights,
    _compute_overlay_height,
    _measure_text_height,
)
from autocompleter.suggestion_engine import MAX_PREVIEW_LENGTH, Suggestion


# ---- Helpers ----

# AppKit may not be available in the test environment
try:
    import AppKit

    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False

needs_appkit = pytest.mark.skipif(
    not HAS_APPKIT, reason="AppKit not available in test environment"
)


# ---- _measure_text_height tests ----


class TestMeasureTextHeight:
    @needs_appkit
    def test_single_line(self):
        font = AppKit.NSFont.systemFontOfSize_(13)
        h = _measure_text_height("Hello world", font, 400.0)
        assert h > 0
        # A single line of 13pt text should be roughly 14-20 px
        assert h < 30

    @needs_appkit
    def test_multiline_explicit_newlines(self):
        font = AppKit.NSFont.systemFontOfSize_(13)
        single_h = _measure_text_height("Line one", font, 400.0)
        multi_h = _measure_text_height("Line one\nLine two\nLine three", font, 400.0)
        # Multi-line should be taller (roughly 3x the single line)
        assert multi_h > single_h * 2

    @needs_appkit
    def test_long_wrapping_text(self):
        font = AppKit.NSFont.systemFontOfSize_(13)
        short_h = _measure_text_height("Short", font, 400.0)
        long_text = "This is a very long sentence that should wrap " * 10
        long_h = _measure_text_height(long_text, font, 200.0)  # narrow width forces wrapping
        assert long_h > short_h * 2

    def test_fallback_without_appkit(self):
        """When AppKit is not available, _measure_text_height returns 20.0."""
        # We test the fallback by calling with a None font (the function
        # checks HAS_APPKIT before using the font). If HAS_APPKIT is True,
        # we skip this since we can't easily force the fallback.
        if HAS_APPKIT:
            pytest.skip("AppKit is available; can't test fallback path")
        h = _measure_text_height("anything", None, 400.0)
        assert h == 20.0


# ---- Suggestion.preview tests ----


class TestSuggestionPreview:
    def test_short_single_line(self):
        s = Suggestion(text="Hello world", index=0)
        assert s.preview == "Hello world"
        assert "..." not in s.preview

    def test_multiline_shows_first_line_with_ellipsis(self):
        s = Suggestion(text="First line\nSecond line\nThird line", index=0)
        assert s.preview == "First line..."
        assert s.preview.endswith("...")

    def test_long_single_line_truncated(self):
        long_text = "x" * 120
        s = Suggestion(text=long_text, index=0)
        assert len(s.preview) <= MAX_PREVIEW_LENGTH + 3  # +3 for "..."
        assert s.preview.endswith("...")

    def test_exact_max_length_no_truncation(self):
        text = "a" * MAX_PREVIEW_LENGTH
        s = Suggestion(text=text, index=0)
        # Exactly at limit, no newlines -> no truncation
        assert s.preview == text
        assert "..." not in s.preview

    def test_empty_text(self):
        s = Suggestion(text="", index=0)
        assert s.preview == ""

    def test_multiline_with_long_first_line(self):
        first_line = "x" * 100
        s = Suggestion(text=first_line + "\nSecond line", index=0)
        assert s.preview.endswith("...")
        # Should truncate the first line to MAX_PREVIEW_LENGTH
        assert len(s.preview) <= MAX_PREVIEW_LENGTH + 3

    def test_single_sentence_no_newline(self):
        s = Suggestion(text="Just one sentence.", index=0)
        assert s.preview == "Just one sentence."


# ---- Overlay height calculation tests ----


class TestOverlayHeightCalculation:
    def _make_config(self, **kwargs):
        defaults = dict(
            width=400,
            max_height=400,
            font_size=13,
            padding=8.0,
            item_height=32.0,
            max_suggestion_height=150.0,
        )
        defaults.update(kwargs)
        return OverlayConfig(**defaults)

    def test_empty_suggestions(self):
        cfg = self._make_config()
        h = _compute_overlay_height([], cfg)
        # Just padding + hint bar
        assert h == cfg.padding * 2 + cfg.hint_bar_height

    def test_single_short_suggestion(self):
        cfg = self._make_config()
        suggestions = [Suggestion(text="Hello", index=0)]
        h = _compute_overlay_height(suggestions, cfg)
        # padding * 2 + at least item_height
        assert h >= cfg.padding * 2 + cfg.item_height

    def test_multiple_short_suggestions(self):
        cfg = self._make_config()
        suggestions = [
            Suggestion(text="One", index=0),
            Suggestion(text="Two", index=1),
            Suggestion(text="Three", index=2),
        ]
        h = _compute_overlay_height(suggestions, cfg)
        # Should be at least padding*2 + 3 * item_height
        min_expected = cfg.padding * 2 + 3 * cfg.item_height
        assert h >= min_expected

    def test_clamped_to_max_height(self):
        cfg = self._make_config(max_height=50)
        suggestions = [
            Suggestion(text="One", index=0),
            Suggestion(text="Two", index=1),
            Suggestion(text="Three", index=2),
        ]
        h = _compute_overlay_height(suggestions, cfg)
        assert h <= cfg.max_height

    def test_zero_suggestions_height(self):
        cfg = self._make_config()
        heights = _compute_item_heights([], cfg)
        assert heights == []

    def test_one_suggestion_height(self):
        cfg = self._make_config()
        heights = _compute_item_heights(
            [Suggestion(text="Short", index=0)], cfg
        )
        assert len(heights) == 1
        assert heights[0] >= cfg.item_height

    def test_two_suggestions_height(self):
        cfg = self._make_config()
        heights = _compute_item_heights(
            [
                Suggestion(text="Short", index=0),
                Suggestion(text="Also short", index=1),
            ],
            cfg,
        )
        assert len(heights) == 2
        for h in heights:
            assert h >= cfg.item_height

    def test_three_suggestions_varying_heights(self):
        cfg = self._make_config()
        suggestions = [
            Suggestion(text="Short", index=0),
            Suggestion(text="Medium length text that is a bit longer", index=1),
            Suggestion(
                text="Very long text\nwith multiple\nlines that should\nbe taller",
                index=2,
            ),
        ]
        heights = _compute_item_heights(suggestions, cfg)
        assert len(heights) == 3
        # All should be at least item_height
        for h in heights:
            assert h >= cfg.item_height


# ---- max_suggestion_height clamping tests ----


class TestMaxSuggestionHeightClamping:
    def _make_config(self, **kwargs):
        defaults = dict(
            width=400,
            max_height=600,
            font_size=13,
            padding=8.0,
            item_height=32.0,
            max_suggestion_height=60.0,  # very low clamp for testing
        )
        defaults.update(kwargs)
        return OverlayConfig(**defaults)

    @needs_appkit
    def test_tall_suggestion_clamped(self):
        cfg = self._make_config(max_suggestion_height=60.0)
        tall_text = "\n".join([f"Line {i}" for i in range(20)])
        suggestions = [Suggestion(text=tall_text, index=0)]
        heights = _compute_item_heights(suggestions, cfg)
        assert len(heights) == 1
        # Should be clamped to max_suggestion_height
        assert heights[0] <= cfg.max_suggestion_height

    @needs_appkit
    def test_expanded_item_not_clamped_to_suggestion_max(self):
        cfg = self._make_config(max_suggestion_height=60.0, max_height=600)
        tall_text = "\n".join([f"Line {i}" for i in range(20)])
        suggestions = [Suggestion(text=tall_text, index=0)]
        heights = _compute_item_heights(suggestions, cfg, expanded_index=0)
        assert len(heights) == 1
        # Expanded item is clamped to max_height, not max_suggestion_height
        assert heights[0] <= cfg.max_height
        # But should be taller than the clamped version
        clamped_heights = _compute_item_heights(suggestions, cfg, expanded_index=-1)
        assert heights[0] >= clamped_heights[0]

    def test_short_suggestion_unaffected_by_clamp(self):
        cfg = self._make_config(max_suggestion_height=150.0)
        suggestions = [Suggestion(text="Short", index=0)]
        heights = _compute_item_heights(suggestions, cfg)
        assert len(heights) == 1
        # Short suggestion shouldn't hit the clamp
        assert heights[0] == cfg.item_height

    def test_without_appkit_returns_item_height(self):
        """Without AppKit, item heights fall back to config.item_height."""
        if HAS_APPKIT:
            pytest.skip("AppKit is available; can't test fallback path")
        cfg = self._make_config()
        suggestions = [
            Suggestion(text="a\nb\nc\nd\ne", index=0),
            Suggestion(text="short", index=1),
        ]
        heights = _compute_item_heights(suggestions, cfg)
        assert all(h == cfg.item_height for h in heights)


# ---- Partial accept tests ----


class TestPartialAccept:
    def test_first_sentence_with_period_space(self):
        text = "First sentence. Second sentence. Third sentence."
        result = Autocompleter._extract_first_segment(text)
        assert result == "First sentence."

    def test_first_line_with_newline(self):
        text = "First line\nSecond line\nThird line"
        result = Autocompleter._extract_first_segment(text)
        assert result == "First line"

    def test_single_sentence_returns_all(self):
        text = "Just one sentence without a period space"
        result = Autocompleter._extract_first_segment(text)
        assert result == text

    def test_single_sentence_ending_with_period(self):
        text = "One sentence."
        result = Autocompleter._extract_first_segment(text)
        # No ". " delimiter (period at end, no space after)
        assert result == "One sentence."

    def test_empty_text(self):
        result = Autocompleter._extract_first_segment("")
        assert result == ""

    def test_newline_takes_precedence_over_period(self):
        """If text has both newlines and '. ', newline split happens first."""
        text = "First. Still first\nSecond line"
        result = Autocompleter._extract_first_segment(text)
        assert result == "First. Still first"

    def test_only_newlines_no_periods(self):
        text = "Line one\nLine two"
        result = Autocompleter._extract_first_segment(text)
        assert result == "Line one"

    def test_period_at_end_of_single_sentence(self):
        text = "Hello world."
        result = Autocompleter._extract_first_segment(text)
        assert result == "Hello world."

    def test_multiple_periods_extracts_first_sentence(self):
        text = "First. Second. Third."
        result = Autocompleter._extract_first_segment(text)
        assert result == "First."

    def test_whitespace_handling(self):
        text = "  First line  \n  Second line  "
        result = Autocompleter._extract_first_segment(text)
        # first_line is stripped from split, then strip() applied
        assert result == "First line"


# ---- Overlay with varying suggestion counts ----


class TestOverlayVaryingSuggestionCounts:
    def _make_config(self, **kwargs):
        defaults = dict(
            width=400,
            max_height=500,
            font_size=13,
            padding=8.0,
            item_height=32.0,
            max_suggestion_height=150.0,
        )
        defaults.update(kwargs)
        return OverlayConfig(**defaults)

    def test_zero_suggestions(self):
        cfg = self._make_config()
        h = _compute_overlay_height([], cfg)
        assert h == cfg.padding * 2 + cfg.hint_bar_height

    def test_one_short_suggestion(self):
        cfg = self._make_config()
        suggestions = [Suggestion(text="Hi", index=0)]
        h = _compute_overlay_height(suggestions, cfg)
        expected_min = cfg.padding * 2 + cfg.item_height
        assert h >= expected_min

    def test_two_mixed_suggestions(self):
        cfg = self._make_config()
        suggestions = [
            Suggestion(text="Short", index=0),
            Suggestion(text="Multi\nline\nsuggestion", index=1),
        ]
        heights = _compute_item_heights(suggestions, cfg)
        h = _compute_overlay_height(suggestions, cfg)
        assert len(heights) == 2
        assert h >= cfg.padding * 2 + sum(heights)  # or equal, since not clamped to max

    def test_three_varied_suggestions(self):
        cfg = self._make_config()
        suggestions = [
            Suggestion(text="Short", index=0),
            Suggestion(text="A medium length sentence.", index=1),
            Suggestion(
                text="A much longer suggestion\nwith multiple lines\nand more content\nthat spans several lines.",
                index=2,
            ),
        ]
        heights = _compute_item_heights(suggestions, cfg)
        total_h = _compute_overlay_height(suggestions, cfg)
        assert len(heights) == 3
        assert total_h <= cfg.max_height
        # Total should be at least the sum of individual heights + padding + hint bar
        raw_sum = cfg.padding * 2 + sum(heights) + cfg.hint_bar_height
        assert total_h == min(raw_sum, cfg.max_height)
