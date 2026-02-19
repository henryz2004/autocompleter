"""Tests for TextInjector cursor-aware injection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# We need to mock macOS-only modules before importing TextInjector.
# The text_injector module imports AppKit, ApplicationServices, and Quartz
# at the top level, guarded by a try/except.  We patch the module-level
# flag and AX helpers so we can test logic without real AX elements.


@pytest.fixture(autouse=True)
def _patch_has_injection():
    """Ensure HAS_INJECTION is True for all tests in this module."""
    with patch("autocompleter.text_injector.HAS_INJECTION", True):
        yield


@pytest.fixture()
def injector():
    """Create a TextInjector with a mocked system-wide element."""
    from autocompleter.text_injector import TextInjector

    inj = TextInjector.__new__(TextInjector)
    inj._system_wide = MagicMock(name="system_wide")
    return inj


# ---------------------------------------------------------------------------
# Helpers to build mock AX state
# ---------------------------------------------------------------------------

def _make_ax_mocks(
    current_value: str = "",
    is_settable: bool = True,
    set_succeeds: bool = True,
):
    """Return (mock_get, mock_set, mock_settable, mock_perform, captured)
    where *captured* is a dict that records what was set."""

    captured: dict = {}
    focused = MagicMock(name="focused_element")

    def fake_get(element, attr):
        if attr == "AXFocusedUIElement":
            return focused
        if attr == "AXValue":
            return current_value
        return None

    def fake_set(element, attr, value):
        captured[attr] = value
        return set_succeeds

    def fake_settable(element, attr):
        return is_settable

    return fake_get, fake_set, fake_settable, focused, captured


# ===================================================================
# _inject_via_ax — insertion_point=None (append, backward compat)
# ===================================================================

class TestInjectViaAxAppend:
    """When insertion_point is None, text is appended to the end."""

    def test_append_to_empty(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("hello")
        assert result is True
        assert captured["AXValue"] == "hello"

    def test_append_to_existing(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="foo")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("bar")
        assert result is True
        assert captured["AXValue"] == "foobar"


# ===================================================================
# _inject_via_ax — insertion_point=0 (insert at beginning)
# ===================================================================

class TestInjectViaAxInsertAtBeginning:

    def test_insert_at_zero(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="world")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("hello ", insertion_point=0)
        assert result is True
        assert captured["AXValue"] == "hello world"


# ===================================================================
# _inject_via_ax — insertion_point in middle
# ===================================================================

class TestInjectViaAxInsertMiddle:

    def test_insert_at_middle(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="helo")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("l", insertion_point=3)
        assert result is True
        assert captured["AXValue"] == "hello"

    def test_insert_preserves_text_after_cursor(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(
            current_value="The  fox jumps"
        )
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("quick brown", insertion_point=4)
        assert result is True
        assert captured["AXValue"] == "The quick brown fox jumps"


# ===================================================================
# _inject_via_ax — insertion_point at end (same as append)
# ===================================================================

class TestInjectViaAxInsertAtEnd:

    def test_insert_at_end_matches_append(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="abc")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("def", insertion_point=3)
        assert result is True
        assert captured["AXValue"] == "abcdef"


# ===================================================================
# _inject_via_ax — insertion_point beyond string length (clamp)
# ===================================================================

class TestInjectViaAxClamp:

    def test_clamp_beyond_length(self, injector):
        """insertion_point past the end should be clamped to len(value)."""
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="ab")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("cd", insertion_point=999)
        assert result is True
        # Clamped to end → same as append
        assert captured["AXValue"] == "abcd"

    def test_clamp_negative(self, injector):
        """Negative insertion_point should be clamped to 0."""
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="world")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("hello ", insertion_point=-5)
        assert result is True
        assert captured["AXValue"] == "hello world"


# ===================================================================
# _inject_via_ax — replace=True ignores insertion_point
# ===================================================================

class TestInjectViaAxReplace:

    def test_replace_ignores_insertion_point(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="old text")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax(
                "new text", insertion_point=3, replace=True,
            )
        assert result is True
        assert captured["AXValue"] == "new text"

    def test_replace_without_insertion_point(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="old text")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("new text", replace=True)
        assert result is True
        assert captured["AXValue"] == "new text"


# ===================================================================
# _inject_via_ax — cursor position (AXSelectedTextRange) after inject
# ===================================================================

class TestInjectViaAxCursorUpdate:

    def test_cursor_set_after_middle_insert(self, injector):
        """After injecting at position 3, cursor should be at 3+len(text)."""
        get, set_, settable, focused, captured = _make_ax_mocks(
            current_value="abcdef"
        )

        mock_as = MagicMock()
        range_sentinel = MagicMock(name="range_value")
        mock_as.AXValueCreate.return_value = range_sentinel
        mock_as.kAXValueTypeCFRange = "CFRange"

        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", mock_as), \
             patch.dict("sys.modules", {"ApplicationServices": mock_as}):
            result = injector._inject_via_ax("XY", insertion_point=3)

        assert result is True
        assert captured["AXValue"] == "abcXYdef"
        # The cursor should be placed at position 3 + 2 = 5
        mock_as.AXValueCreate.assert_called_once_with("CFRange", (5, 0))
        assert captured["AXSelectedTextRange"] == range_sentinel

    def test_cursor_set_after_append(self, injector):
        """When insertion_point is None (append), cursor goes to end."""
        get, set_, settable, focused, captured = _make_ax_mocks(
            current_value="abc"
        )

        mock_as = MagicMock()
        range_sentinel = MagicMock(name="range_value")
        mock_as.AXValueCreate.return_value = range_sentinel
        mock_as.kAXValueTypeCFRange = "CFRange"

        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", mock_as), \
             patch.dict("sys.modules", {"ApplicationServices": mock_as}):
            result = injector._inject_via_ax("de")

        assert result is True
        assert captured["AXValue"] == "abcde"
        # Cursor at end: len("abcde") = 5
        mock_as.AXValueCreate.assert_called_once_with("CFRange", (5, 0))


# ===================================================================
# inject() passes insertion_point through to _inject_via_ax
# ===================================================================

class TestInjectPassthrough:

    def test_inject_passes_insertion_point(self, injector):
        """inject() should forward insertion_point to _inject_via_ax."""
        with patch.object(injector, "_inject_via_ax", return_value=True) as mock_ax:
            result = injector.inject("text", insertion_point=7)
        assert result is True
        mock_ax.assert_called_once_with("text", insertion_point=7)

    def test_inject_passes_none_by_default(self, injector):
        """When insertion_point is omitted, None is forwarded."""
        with patch.object(injector, "_inject_via_ax", return_value=True) as mock_ax:
            result = injector.inject("text")
        assert result is True
        mock_ax.assert_called_once_with("text", insertion_point=None)

    def test_inject_skips_ax_when_replace(self, injector):
        """When replace=True, _inject_via_ax is skipped entirely."""
        with patch.object(injector, "_inject_via_ax") as mock_ax, \
             patch.object(injector, "_inject_via_clipboard", return_value=True):
            result = injector.inject("text", replace=True, insertion_point=5)
        assert result is True
        mock_ax.assert_not_called()


# ===================================================================
# Clipboard and keystroke methods are NOT affected by insertion_point
# ===================================================================

class TestClipboardAndKeystrokesUnaffected:
    """Clipboard paste and keystroke injection do not accept insertion_point.

    They inject at the current OS cursor position natively, so the
    insertion_point parameter is intentionally not passed through.
    """

    def test_clipboard_has_no_insertion_point_param(self, injector):
        """_inject_via_clipboard signature has no insertion_point param."""
        import inspect
        sig = inspect.signature(injector._inject_via_clipboard)
        params = list(sig.parameters.keys())
        assert "insertion_point" not in params

    def test_keystrokes_has_no_insertion_point_param(self, injector):
        """_inject_via_keystrokes signature has no insertion_point param."""
        import inspect
        sig = inspect.signature(injector._inject_via_keystrokes)
        params = list(sig.parameters.keys())
        assert "insertion_point" not in params

    def test_fallback_to_clipboard_when_ax_fails(self, injector):
        """When AX fails, inject() falls back to clipboard without insertion_point."""
        with patch.object(injector, "_inject_via_ax", return_value=False), \
             patch.object(injector, "_inject_via_clipboard", return_value=True) as mock_clip:
            result = injector.inject("text", insertion_point=5)
        assert result is True
        # Clipboard called with just text, no insertion_point
        mock_clip.assert_called_once_with("text")

    def test_fallback_to_keystrokes_when_all_fail(self, injector):
        """When AX and clipboard fail, inject() falls back to keystrokes."""
        with patch.object(injector, "_inject_via_ax", return_value=False), \
             patch.object(injector, "_inject_via_clipboard", return_value=False), \
             patch.object(injector, "_inject_via_keystrokes", return_value=True) as mock_keys:
            result = injector.inject("text", insertion_point=5)
        assert result is True
        mock_keys.assert_called_once_with("text")


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_inject_empty_text_returns_false(self, injector):
        result = injector.inject("")
        assert result is False

    def test_ax_not_settable_returns_false(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(is_settable=False)
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("text", insertion_point=0)
        assert result is False

    def test_ax_set_fails_returns_false(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(set_succeeds=False)
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("text", insertion_point=2)
        assert result is False

    def test_no_focused_element_returns_false(self, injector):
        def fake_get(element, attr):
            if attr == "AXFocusedUIElement":
                return None
            return None

        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=fake_get), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("text", insertion_point=0)
        assert result is False

    def test_insert_at_zero_on_empty_string(self, injector):
        get, set_, settable, focused, captured = _make_ax_mocks(current_value="")
        with patch("autocompleter.text_injector.ax_get_attribute", side_effect=get), \
             patch("autocompleter.text_injector.ax_set_attribute", side_effect=set_), \
             patch("autocompleter.text_injector.ax_is_attribute_settable", side_effect=settable), \
             patch("autocompleter.text_injector.ApplicationServices", MagicMock()):
            result = injector._inject_via_ax("hello", insertion_point=0)
        assert result is True
        assert captured["AXValue"] == "hello"
