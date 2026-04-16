"""Tests for autocompleter.feedback."""

from __future__ import annotations

import json
from pathlib import Path

from autocompleter.feedback import (
    FeedbackContext,
    FeedbackReporter,
    build_payload,
    extract_url_domain,
)


class TestExtractURLDomain:
    def test_http_url_returns_host_only(self):
        assert extract_url_domain("https://github.com/owner/repo/issues/123") == "github.com"

    def test_http_with_query(self):
        assert extract_url_domain("https://mail.google.com/mail/u/0/#inbox?subject=Secret") == "mail.google.com"

    def test_lowercases_host(self):
        assert extract_url_domain("https://Example.COM/path") == "example.com"

    def test_empty_returns_none(self):
        assert extract_url_domain("") is None
        assert extract_url_domain(None) is None

    def test_non_http_scheme_returns_scheme_only(self):
        assert extract_url_domain("file:///Users/alice/secret.txt") == "file"


class TestBuildPayload:
    def test_minimal_context_still_builds(self):
        payload = build_payload(FeedbackContext())
        assert payload["schema"] == "autocompleter.feedback.v1"
        assert payload["report_id"]
        assert isinstance(payload["timestamp"], int)
        assert payload["version"]
        assert payload["app"] == {
            "name": None,
            "role": None,
            "placeholder_detected": None,
            "url_domain": None,
        }

    def test_preserves_safe_fields(self):
        payload = build_payload(
            FeedbackContext(
                app_name="Slack",
                app_bundle_role="AXTextArea",
                placeholder_detected=True,
                url_domain="slack.com",
                mode="reply",
                trigger_type="manual",
                extractor_name="SlackExtractor",
                conversation_turns_detected=4,
                conversation_speakers=2,
                shell_detected=False,
                tui_detected=False,
                visible_source="cache",
                visible_text_elements_count=32,
                subtree_context_chars=410,
                used_subtree_context=True,
                used_semantic_context=False,
                used_memory_context=False,
                llm_provider="openai",
                llm_model="qwen-3",
                fallback_provider="openai",
                fallback_model="qwen-small",
                fallback_used=False,
                latency_ms=1234.5,
                first_suggestion_ms=456.7,
                suggestion_count=3,
            )
        )
        assert payload["app"]["name"] == "Slack"
        assert payload["context_pipeline"]["mode"] == "reply"
        assert payload["llm"]["latency_ms"] == 1234.5

    def test_feedback_context_has_no_content_fields(self):
        forbidden = {
            "before_cursor",
            "after_cursor",
            "visible_text",
            "window_title",
            "suggestion_text",
            "conversation_turns",
        }
        fields = set(FeedbackContext.__dataclass_fields__.keys())
        assert not (fields & forbidden)


class TestFeedbackReporter:
    def test_writes_local_file(self, tmp_path: Path):
        reporter = FeedbackReporter(feedback_dir=tmp_path / "feedback")
        payload = reporter.submit(
            FeedbackContext(app_name="Notes", mode="continuation"),
            installation_id="install-123",
        )
        files = list((tmp_path / "feedback").glob("*.json"))
        assert len(files) == 1
        on_disk = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["app"]["name"] == "Notes"
        assert on_disk["installation_id"] == "install-123"
