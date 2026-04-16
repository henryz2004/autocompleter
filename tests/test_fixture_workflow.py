"""Tests for normalized fixture loading, replay, and context benchmarks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocompleter.fixture_tools import (
    build_focused_element,
    extract_conversation_turns,
    load_normalized_fixture,
)
from autocompleter.suggestion_engine import AutocompleteMode
from replay_fixture import assemble_context, extract_subtree
from analyze_fixture import build_analysis


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AX_TREES_DIR = FIXTURES_DIR / "ax_trees"
REPLAY_DIR = FIXTURES_DIR / "replay_sessions"
CONTEXT_EXPECTED_DIR = FIXTURES_DIR / "context_expected"


def _discover_context_expectations() -> list[str]:
    if not CONTEXT_EXPECTED_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONTEXT_EXPECTED_DIR.glob("*.json"))


def test_normalized_replay_fixture_reconstructs_focused_value() -> None:
    fixture = load_normalized_fixture(
        REPLAY_DIR / "20260316-143623-gen8-claude.json"
    )

    assert fixture.artifact_type.startswith("manual_invocation")
    assert fixture.focused is not None
    assert fixture.focused.value.startswith("thinking of making a simple waitlist")
    assert fixture.focused.cursor_position == len(fixture.focused.before_cursor)

    focused = build_focused_element(fixture)
    assert focused is not None
    assert focused.before_cursor.startswith("thinking of making a simple waitlist")
    assert focused.insertion_point == 86


def test_replay_assemble_context_uses_current_context_store_signature() -> None:
    fixture = load_normalized_fixture(AX_TREES_DIR / "claude-current-chat.json")
    turns, extractor_name = extract_conversation_turns(
        fixture.tree,
        app_name=fixture.app,
        window_title=fixture.window_title,
    )
    subtree = extract_subtree(fixture.tree)

    context, system_prompt, user_prompt, tree_overview = assemble_context(
        fixture,
        focused=None,
        conversation_turns=turns or [],
        subtree_context=subtree,
        mode=AutocompleteMode.REPLY,
        num_suggestions=3,
    )

    assert extractor_name == "ClaudeDesktopExtractor"
    assert "Conversation:" in context
    assert "government bonds vs corporate bonds risk and return?" in context
    assert "Generate exactly 3 distinct suggestions" in user_prompt
    assert "respond with a JSON object" in system_prompt
    assert tree_overview is None or isinstance(tree_overview, str)


def test_build_analysis_reports_extractor_and_containers() -> None:
    analysis = build_analysis(AX_TREES_DIR / "chatgpt-current-chat.json")

    assert analysis["extractor"]["name"] == "ChatGPTExtractor"
    assert analysis["extractor"]["extractedTurnCount"] >= 1
    assert analysis["stats"]["totalNodes"] > 0
    assert analysis["topContainers"]
    assert analysis["sampleTextNodes"]


@pytest.mark.parametrize(
    "fixture_name",
    _discover_context_expectations(),
    ids=_discover_context_expectations(),
)
def test_fixture_context_benchmarks(fixture_name: str) -> None:
    fixture = load_normalized_fixture(AX_TREES_DIR / f"{fixture_name}.json")
    expected = json.loads(
        (CONTEXT_EXPECTED_DIR / f"{fixture_name}.json").read_text(encoding="utf-8")
    )
    mode = AutocompleteMode[expected["mode"].upper()]
    turns, extractor_name = extract_conversation_turns(
        fixture.tree,
        app_name=fixture.app,
        window_title=fixture.window_title,
    )
    subtree = extract_subtree(fixture.tree)
    focused = build_focused_element(fixture)

    context, _system_prompt, _user_prompt, tree_overview = assemble_context(
        fixture,
        focused=focused,
        conversation_turns=turns or fixture.conversation_turns,
        subtree_context=subtree,
        mode=mode,
        num_suggestions=3,
    )

    if expected.get("extractor"):
        assert extractor_name == expected["extractor"]
    for needle in expected.get("context_contains", []):
        assert needle in context
    for needle in expected.get("context_not_contains", []):
        assert needle not in context
    for needle in expected.get("overview_contains", []):
        assert tree_overview is not None
        assert needle in tree_overview
    for needle in expected.get("subtree_contains", []):
        assert subtree is not None
        assert needle in subtree
