"""Tests for hotkey.py — parse_hotkey and HotkeyListener registration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autocompleter.hotkey import (
    HotkeyListener,
    KEY_CODES,
    MODIFIER_FLAGS,
    parse_hotkey,
)


# ---------------------------------------------------------------------------
# parse_hotkey tests
# ---------------------------------------------------------------------------

class TestParseHotkey:
    @pytest.mark.parametrize("hotkey_str, expected_keycode, expected_has_ctrl", [
        ("ctrl+space", KEY_CODES["space"], True),
        ("ctrl+tab", KEY_CODES["tab"], True),
        ("ctrl+return", KEY_CODES["return"], True),
    ])
    def test_ctrl_combos(self, hotkey_str, expected_keycode, expected_has_ctrl):
        keycode, flags = parse_hotkey(hotkey_str)
        assert keycode == expected_keycode
        if expected_has_ctrl:
            assert flags & MODIFIER_FLAGS["ctrl"]

    def test_single_character_key(self):
        keycode, flags = parse_hotkey("cmd+j")
        assert keycode == 38  # j's keycode
        assert flags & MODIFIER_FLAGS["cmd"]

    def test_unknown_component_returns_zero(self):
        """Unknown multi-char components produce keycode 0."""
        keycode, flags = parse_hotkey("ctrl+unknownkey")
        assert keycode == 0
        assert flags & MODIFIER_FLAGS["ctrl"]

    def test_modifier_order_irrelevant(self):
        """ctrl+shift+space == shift+ctrl+space."""
        kc1, f1 = parse_hotkey("ctrl+shift+space")
        kc2, f2 = parse_hotkey("shift+ctrl+space")
        assert kc1 == kc2
        assert f1 == f2

    def test_multiple_modifiers(self):
        """Multiple modifiers should all be present in flags."""
        keycode, flags = parse_hotkey("ctrl+alt+shift+space")
        assert keycode == KEY_CODES["space"]
        assert flags & MODIFIER_FLAGS["ctrl"]
        assert flags & MODIFIER_FLAGS["alt"]
        assert flags & MODIFIER_FLAGS["shift"]

    def test_plain_key_no_modifier(self):
        """A key with no modifier yields flags=0."""
        keycode, flags = parse_hotkey("space")
        assert keycode == KEY_CODES["space"]
        assert flags == 0

    def test_whitespace_around_parts(self):
        """Whitespace around + should be stripped."""
        keycode, flags = parse_hotkey("ctrl + space")
        assert keycode == KEY_CODES["space"]
        assert flags & MODIFIER_FLAGS["ctrl"]


# ---------------------------------------------------------------------------
# HotkeyListener tests
# ---------------------------------------------------------------------------

class TestHotkeyListenerRegistration:
    def test_register_stores_callback(self):
        listener = HotkeyListener()
        cb = MagicMock()
        listener.register("ctrl+space", cb)
        key = (KEY_CODES["space"], MODIFIER_FLAGS["ctrl"])
        assert key in listener._callbacks
        assert listener._callbacks[key] is cb

    def test_start_without_quartz_is_noop(self):
        """When Quartz is not available, start() should not crash."""
        listener = HotkeyListener()
        with patch("autocompleter.hotkey.HAS_QUARTZ", False):
            listener.start()
        assert not listener._running
