"""Tests for help_overlay pure-Python helpers (hotkey formatting & defaults).

The AppKit-backed view is skipped; we only exercise the logic that shapes the
help content and glyph-formats the hotkeys.
"""

from __future__ import annotations

from autocompleter.help_overlay import (
    HelpEntry,
    _default_entries,
    _format_hotkey,
)


class TestFormatHotkey:
    def test_ctrl_space(self):
        assert _format_hotkey("ctrl+space") == "⌃Space"

    def test_ctrl_shift_b(self):
        assert _format_hotkey("ctrl+shift+b") == "⌃⇧B"

    def test_ctrl_slash(self):
        assert _format_hotkey("ctrl+/") == "⌃/"

    def test_arrow_keys(self):
        assert _format_hotkey("up") == "↑"
        assert _format_hotkey("down") == "↓"

    def test_tab_and_return(self):
        assert _format_hotkey("shift+tab") == "⇧Tab"
        assert _format_hotkey("return") == "Return"

    def test_cmd_modifier(self):
        assert _format_hotkey("cmd+k") == "⌘K"

    def test_alt_and_option_equivalent(self):
        assert _format_hotkey("alt+tab") == _format_hotkey("option+tab") == "⌥Tab"


class TestDefaultEntries:
    def test_has_three_sections(self):
        sections = _default_entries(
            trigger_hotkey="ctrl+space",
            regenerate_hotkey="ctrl+r",
            help_hotkey="ctrl+/",
            report_hotkey="ctrl+shift+b",
        )
        names = [s[0] for s in sections]
        assert len(sections) == 3
        assert "Generate" in names
        assert "Navigate & accept" in names
        assert "Help & feedback" in names

    def test_entries_are_help_entry_instances(self):
        sections = _default_entries(
            trigger_hotkey="ctrl+space",
            regenerate_hotkey="ctrl+r",
            help_hotkey="ctrl+/",
            report_hotkey="ctrl+shift+b",
        )
        for _, entries in sections:
            for entry in entries:
                assert isinstance(entry, HelpEntry)
                assert entry.keys
                assert entry.description

    def test_config_hotkeys_appear_in_entries(self):
        """Changing the configured hotkeys should be reflected in the help panel."""
        sections = _default_entries(
            trigger_hotkey="cmd+j",
            regenerate_hotkey="cmd+shift+r",
            help_hotkey="cmd+?",
            report_hotkey="cmd+shift+x",
        )
        all_keys = [e.keys for _, entries in sections for e in entries]
        assert "⌘J" in all_keys
        assert "⌘⇧R" in all_keys
        assert "⌘⇧X" in all_keys

    def test_report_hotkey_description_is_content_free(self):
        """The help entry text makes the content-free promise visible to users."""
        sections = _default_entries(
            trigger_hotkey="ctrl+space",
            regenerate_hotkey="ctrl+r",
            help_hotkey="ctrl+/",
            report_hotkey="ctrl+shift+b",
        )
        help_section = next(s for s in sections if s[0] == "Help & feedback")
        report_entry = next(e for e in help_section[1] if "bug" in e.description.lower() or "report" in e.description.lower())
        assert "no content" in report_entry.description.lower()
