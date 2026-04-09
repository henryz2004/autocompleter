"""Tests for configuration loading."""

from __future__ import annotations

from autocompleter.config import load_config


class TestFollowupAfterAcceptConfig:
    def test_followup_after_accept_defaults_on(self, monkeypatch):
        monkeypatch.delenv("AUTOCOMPLETER_FOLLOWUP_AFTER_ACCEPT", raising=False)
        cfg = load_config()
        assert cfg.followup_after_accept_enabled is True

    def test_followup_after_accept_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETER_FOLLOWUP_AFTER_ACCEPT", "false")
        cfg = load_config()
        assert cfg.followup_after_accept_enabled is False
