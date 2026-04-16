"""Tests for autocompleter.feedback.

Focus:
- The feedback payload never contains user content (cursor text, visible text,
  window title, suggestion text, full URL path).
- URL domain extraction strips path/query.
- Local write works with or without a webhook.
- Webhook failures never raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocompleter.feedback import (
    FeedbackContext,
    FeedbackReporter,
    build_payload,
    extract_url_domain,
)


# ---------------------------------------------------------------------------
# extract_url_domain
# ---------------------------------------------------------------------------

class TestExtractURLDomain:
    def test_http_url_returns_host_only(self):
        assert extract_url_domain("https://github.com/owner/repo/issues/123") == "github.com"

    def test_http_with_query(self):
        assert (
            extract_url_domain("https://mail.google.com/mail/u/0/#inbox?subject=Secret")
            == "mail.google.com"
        )

    def test_lowercases_host(self):
        assert extract_url_domain("https://Example.COM/path") == "example.com"

    def test_empty_returns_none(self):
        assert extract_url_domain("") is None
        assert extract_url_domain(None) is None

    def test_non_http_scheme_returns_scheme_only(self):
        """Don't leak filesystem paths via file:// URLs."""
        assert extract_url_domain("file:///Users/alice/secret.txt") == "file"

    def test_custom_scheme_returns_scheme_only(self):
        assert extract_url_domain("obsidian://vault/note") == "obsidian"

    def test_no_scheme_returns_none(self):
        assert extract_url_domain("just-some-text") is None


# ---------------------------------------------------------------------------
# build_payload — schema and scrubbing
# ---------------------------------------------------------------------------

class TestBuildPayload:
    def test_minimal_context_still_builds(self):
        payload = build_payload(FeedbackContext())
        # Schema + required identifiers are always present.
        assert payload["schema"] == "autocompleter.feedback.v1"
        assert isinstance(payload["report_id"], str) and payload["report_id"]
        assert isinstance(payload["timestamp"], int)
        assert payload["version"]
        # All app fields are None — no content leaked from empty context.
        assert payload["app"] == {
            "name": None,
            "role": None,
            "placeholder_detected": None,
            "url_domain": None,
        }

    def test_uses_provided_installation_id(self):
        payload = build_payload(FeedbackContext(), installation_id="abcdef")
        assert payload["installation_id"] == "abcdef"

    def test_preserves_safe_fields(self):
        ctx = FeedbackContext(
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
            used_semantic_context=True,
            used_memory_context=False,
            llm_provider="openai",
            llm_model="qwen-3",
            fallback_provider="openai",
            fallback_model="groq-qwen",
            fallback_used=False,
            latency_ms=1234.5,
            first_suggestion_ms=456.7,
            suggestion_count=3,
        )
        payload = build_payload(ctx)
        assert payload["app"]["name"] == "Slack"
        assert payload["app"]["role"] == "AXTextArea"
        assert payload["app"]["url_domain"] == "slack.com"
        assert payload["context_pipeline"]["mode"] == "reply"
        assert payload["context_pipeline"]["extractor"] == "SlackExtractor"
        assert payload["context_pipeline"]["conversation_turns_detected"] == 4
        assert payload["context_pipeline"]["visible_source"] == "cache"
        assert payload["llm"]["provider"] == "openai"
        assert payload["llm"]["latency_ms"] == 1234.5

    def test_payload_does_not_include_disallowed_keys(self):
        """The payload schema must not contain fields that carry user content.

        Defense-in-depth: if someone adds e.g. a "cursor_text" field later,
        this test fails loudly. We only flag keys that strongly suggest
        content (not counts or flags, which are allowed).
        """
        ctx = FeedbackContext()
        payload = build_payload(ctx)
        forbidden_key_substrings = {
            "cursor_text",
            "before_cursor",
            "after_cursor",
            "suggestion_text",
            "suggestions_text",
            "window_title",
            "full_url",
            "conversation_turn_text",
            "draft_text",
        }

        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k
                    yield from _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    yield from _walk(item)

        keys = {str(k).lower() for k in _walk(payload)}
        for needle in forbidden_key_substrings:
            matches = {k for k in keys if needle in k}
            assert not matches, (
                f"Disallowed content-like key(s) in payload: {matches}"
            )

    def test_feedback_context_has_no_content_fields(self):
        """FeedbackContext dataclass must not grow a user-content field."""
        forbidden = {
            "before_cursor",
            "after_cursor",
            "cursor_text",
            "visible_text",
            "visible_text_elements",
            "window_title",
            "full_url",
            "url",
            "suggestion_text",
            "suggestions",
            "conversation_turn_text",
            "conversation_turns",
        }
        fields = set(FeedbackContext.__dataclass_fields__.keys())
        leaked = fields & forbidden
        assert not leaked, f"FeedbackContext leaked content fields: {leaked}"


# ---------------------------------------------------------------------------
# FeedbackReporter — local write + webhook
# ---------------------------------------------------------------------------

class TestFeedbackReporter:
    def test_writes_local_file(self, tmp_path: Path):
        reporter = FeedbackReporter(feedback_dir=tmp_path / "feedback")
        ctx = FeedbackContext(app_name="Notes", mode="continuation")
        payload = reporter.submit(ctx)
        assert payload["app"]["name"] == "Notes"
        files = list((tmp_path / "feedback").glob("*.json"))
        assert len(files) == 1
        on_disk = json.loads(files[0].read_text())
        assert on_disk["app"]["name"] == "Notes"
        assert on_disk["installation_id"] == reporter.installation_id

    def test_installation_id_is_stable_across_instances(self, tmp_path: Path):
        r1 = FeedbackReporter(feedback_dir=tmp_path / "feedback")
        r2 = FeedbackReporter(feedback_dir=tmp_path / "feedback")
        assert r1.installation_id
        assert r1.installation_id == r2.installation_id

    def test_submit_without_webhook_does_not_raise(self, tmp_path: Path):
        reporter = FeedbackReporter(feedback_dir=tmp_path / "feedback", webhook_url=None)
        reporter.submit(FeedbackContext(app_name="Terminal"))  # should not raise

    def test_webhook_failure_does_not_raise(self, tmp_path: Path, monkeypatch):
        """A broken webhook must never surface errors to the caller."""
        reporter = FeedbackReporter(
            feedback_dir=tmp_path / "feedback",
            webhook_url="http://127.0.0.1:1/does-not-exist",
            http_timeout_s=0.05,
        )
        # The webhook POST runs on a daemon thread; submit() itself must
        # return cleanly even if the POST would fail.
        payload = reporter.submit(FeedbackContext(app_name="App"))
        assert payload["app"]["name"] == "App"
