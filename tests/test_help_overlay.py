"""Tests for help overlay pure-Python helpers."""

from __future__ import annotations

from autocompleter.help_overlay import HelpEntry, _default_entries, _format_hotkey


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


class TestDefaultEntries:
    def test_has_expected_sections(self):
        sections = _default_entries(
            trigger_hotkey="ctrl+space",
            regenerate_hotkey="ctrl+r",
            help_hotkey="ctrl+/",
            report_hotkey="ctrl+shift+b",
        )
        names = [section[0] for section in sections]
        assert names == ["Generate", "Navigate & accept", "Help & feedback"]

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

    def test_content_free_report_description(self):
        sections = _default_entries(
            trigger_hotkey="ctrl+space",
            regenerate_hotkey="ctrl+r",
            help_hotkey="ctrl+/",
            report_hotkey="ctrl+shift+b",
        )
        help_section = next(section for section in sections if section[0] == "Help & feedback")
        report_entry = next(entry for entry in help_section[1] if "report" in entry.description.lower())
        assert "no content" in report_entry.description.lower()
