"""Tests for fixture-driven quality review helpers."""

from pathlib import Path

from autocompleter.quality_review import (
    COMBINED_VARIANT,
    QualityVariant,
    apply_quality_variant_to_context,
    load_valid_invocation_artifacts,
)
from autocompleter.suggestion_engine import AutocompleteMode, build_messages


class TestLoadInvocationArtifacts:
    def test_skips_malformed_json(self, tmp_path: Path):
        good = tmp_path / "good.json"
        bad = tmp_path / "bad.json"
        good.write_text('{"artifactType":"manual_invocation_v1","app":"Codex"}')
        bad.write_text('{"artifactType":')

        valid, skipped = load_valid_invocation_artifacts([good, bad])

        assert len(valid) == 1
        assert valid[0]["app"] == "Codex"
        assert skipped == [("bad.json", "JSONDecodeError")]


class TestContextVariants:
    def test_combined_variant_removes_pollution_blocks(self):
        context = (
            "App: Codex | Window: Codex\n\n"
            "[Recent activity from other apps]\n- Terminal: launch command\n\n"
            "Nearby content:\n"
            "<context>\n"
            "  <Group><StaticText>Threads</StaticText></Group>\n"
            "  <Group><StaticText>Meaningful nearby message content</StaticText></Group>\n"
            "  <input><TextArea focused=\"true\">placeholder-ish</TextArea></input>\n"
            "</context>\n\n"
            "Background context (lower priority):\nold noisy fragment\n\n"
            "Text before cursor:\nhi there "
        )

        result = apply_quality_variant_to_context(context, COMBINED_VARIANT)

        assert "[Recent activity from other apps]" not in result
        assert "Background context (lower priority):" not in result
        assert "Meaningful nearby message content" in result
        assert 'focused="true"' not in result
        assert "Text before cursor:" in result

    def test_combined_variant_prefers_cursor_only_for_strong_prefix(self):
        context = (
            "App: Codex | Window: Codex\n\n"
            "Visible context:\n"
            "Meaningful nearby message content from the thread\n\n"
            "Nearby content:\n"
            "<context>\n"
            "  <Group><StaticText>Meaningful nearby message content</StaticText></Group>\n"
            "</context>\n\n"
            "Text before cursor:\njust invoked it, can you check? also, do you think "
        )

        result = apply_quality_variant_to_context(context, COMBINED_VARIANT)

        assert "Nearby content:" not in result
        assert "Meaningful nearby message content" not in result
        assert "Visible context:" not in result
        assert "Text before cursor:" in result


class TestPromptAwareness:
    def test_placeholder_aware_build_messages_adds_rules(self):
        system, _user = build_messages(
            mode=AutocompleteMode.CONTINUATION,
            context="Text before cursor:\nhello ",
            prompt_placeholder_aware=True,
        )
        assert "placeholder text" in system
        assert "strongest signal" in system
        assert "unfinished lead-in" in system
        assert "generic fallback phrase" in system

    def test_default_build_messages_omits_extra_rules(self):
        system, _user = build_messages(
            mode=AutocompleteMode.CONTINUATION,
            context="Text before cursor:\nhello ",
            prompt_placeholder_aware=False,
        )
        assert "placeholder text" not in system

    def test_completion_user_prompt_prioritizes_cursor_text(self):
        _system, user = build_messages(
            mode=AutocompleteMode.CONTINUATION,
            context="Text before cursor:\nhello ",
            prompt_placeholder_aware=True,
        )
        assert "Prioritize 'Text before cursor'" in user
