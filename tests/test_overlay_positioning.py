"""Tests for overlay.py — _clamp_to_screen and move_selection logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autocompleter.overlay import OverlayConfig, SuggestionOverlay
from autocompleter.suggestion_engine import Suggestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screen(x: float, y: float, w: float, h: float) -> MagicMock:
    """Create a mock NSScreen with the given frame."""
    screen = MagicMock()
    frame = MagicMock()
    frame.origin.x = x
    frame.origin.y = y
    frame.size.width = w
    frame.size.height = h
    screen.frame.return_value = frame
    return screen


def _make_overlay(screens=None) -> SuggestionOverlay:
    """Create a SuggestionOverlay with mocked AppKit."""
    overlay = SuggestionOverlay.__new__(SuggestionOverlay)
    overlay._config = OverlayConfig()
    overlay._window = None
    overlay._view = None
    overlay._suggestions = []
    overlay._selected_index = 0
    overlay._on_accept = None
    overlay._visible = False
    overlay._last_x = 0.0
    overlay._last_ns_y = 0.0
    overlay._last_caret_ns_y = 0.0
    overlay._last_caret_height = 20.0
    return overlay


# ---------------------------------------------------------------------------
# _clamp_to_screen tests
# ---------------------------------------------------------------------------

class TestClampToScreen:
    def test_within_bounds_unchanged(self):
        """Position fully within screen bounds stays the same."""
        screen = _make_screen(0, 0, 1920, 1080)
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", True), \
             patch("autocompleter.overlay.AppKit") as mock_appkit:
            mock_appkit.NSScreen.screens.return_value = [screen]
            x, y = overlay._clamp_to_screen(100, 500, 400, 200)
        assert x == 100
        assert y == 500

    def test_shift_from_right_edge(self):
        """Overlay extending past right edge is shifted left."""
        screen = _make_screen(0, 0, 1920, 1080)
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", True), \
             patch("autocompleter.overlay.AppKit") as mock_appkit:
            mock_appkit.NSScreen.screens.return_value = [screen]
            x, y = overlay._clamp_to_screen(1800, 500, 400, 200)
        assert x == 1920 - 400  # shifted to fit

    def test_shift_from_left_edge(self):
        """Overlay extending past left edge is shifted right."""
        screen = _make_screen(0, 0, 1920, 1080)
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", True), \
             patch("autocompleter.overlay.AppKit") as mock_appkit:
            mock_appkit.NSScreen.screens.return_value = [screen]
            x, y = overlay._clamp_to_screen(-50, 500, 400, 200)
        assert x == 0

    def test_flip_above_when_below_bottom(self):
        """Overlay below screen bottom should flip above the caret."""
        screen = _make_screen(0, 0, 1920, 1080)
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", True), \
             patch("autocompleter.overlay.AppKit") as mock_appkit:
            mock_appkit.NSScreen.screens.return_value = [screen]
            # ns_y=-50 means overlay bottom is below the screen
            x, y = overlay._clamp_to_screen(
                100, -50, 400, 200, caret_ns_y=300, caret_height=20,
            )
        # Should flip above caret: y = caret_ns_y + caret_height + gap
        assert y == 300 + 20 + 4.0

    def test_clamp_to_top(self):
        """Overlay extending above the top edge is clamped down."""
        screen = _make_screen(0, 0, 1920, 1080)
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", True), \
             patch("autocompleter.overlay.AppKit") as mock_appkit:
            mock_appkit.NSScreen.screens.return_value = [screen]
            x, y = overlay._clamp_to_screen(100, 950, 400, 200)
        assert y == 1080 - 200  # clamped so top doesn't exceed screen

    def test_multi_monitor_secondary_screen(self):
        """Overlay on secondary monitor uses that screen's bounds."""
        primary = _make_screen(0, 0, 1920, 1080)
        secondary = _make_screen(1920, 0, 2560, 1440)
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", True), \
             patch("autocompleter.overlay.AppKit") as mock_appkit:
            mock_appkit.NSScreen.screens.return_value = [primary, secondary]
            # Position on secondary, extending past right
            x, y = overlay._clamp_to_screen(4200, 500, 400, 200)
        assert x == 1920 + 2560 - 400  # clamped to secondary right edge

    def test_no_appkit_returns_unchanged(self):
        """Without AppKit, returns position unchanged."""
        overlay = _make_overlay()
        with patch("autocompleter.overlay.HAS_APPKIT", False):
            x, y = overlay._clamp_to_screen(100, 200, 400, 300)
        assert x == 100
        assert y == 200


# ---------------------------------------------------------------------------
# move_selection tests
# ---------------------------------------------------------------------------

class TestMoveSelection:
    def test_wraps_forward(self):
        """Moving forward past the last suggestion wraps to 0."""
        overlay = _make_overlay()
        overlay._suggestions = [
            Suggestion(text="A", index=0),
            Suggestion(text="B", index=1),
            Suggestion(text="C", index=2),
        ]
        overlay._selected_index = 2
        overlay.move_selection(1)
        assert overlay._selected_index == 0

    def test_wraps_backward(self):
        """Moving backward past 0 wraps to the last suggestion."""
        overlay = _make_overlay()
        overlay._suggestions = [
            Suggestion(text="A", index=0),
            Suggestion(text="B", index=1),
            Suggestion(text="C", index=2),
        ]
        overlay._selected_index = 0
        overlay.move_selection(-1)
        assert overlay._selected_index == 2

    def test_single_suggestion_stays(self):
        """With one suggestion, move_selection should stay at 0."""
        overlay = _make_overlay()
        overlay._suggestions = [Suggestion(text="Only", index=0)]
        overlay._selected_index = 0
        overlay.move_selection(1)
        assert overlay._selected_index == 0

    def test_empty_suggestions_noop(self):
        """With no suggestions, move_selection does nothing."""
        overlay = _make_overlay()
        overlay._suggestions = []
        overlay._selected_index = 0
        overlay.move_selection(1)  # should not crash
        assert overlay._selected_index == 0

    def test_normal_forward(self):
        """Normal forward movement increments index."""
        overlay = _make_overlay()
        overlay._suggestions = [
            Suggestion(text="A", index=0),
            Suggestion(text="B", index=1),
        ]
        overlay._selected_index = 0
        overlay.move_selection(1)
        assert overlay._selected_index == 1
