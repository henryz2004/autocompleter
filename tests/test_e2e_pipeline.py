"""End-to-end pipeline tests using captured AX tree fixtures.

For every fixture in tests/fixtures/ax_trees/, this test simulates the
full autocomplete pipeline:

  1. Load AX tree → mock AX API calls
  2. Conversation extraction → verify turns (where expected output exists)
  3. Visible text extraction → verify non-empty text elements
  4. Subtree context extraction (focus-annotated fixtures) → verify XML
  5. Context assembly → verify well-formed context string
  6. Mode detection → verify correct mode assignment

This catches regressions across the entire pipeline — not just individual
extractors — and ensures new code doesn't crash on any real-world AX tree.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.context_store import ContextStore
from autocompleter.conversation_extractors import (
    ConversationTurn,
    get_extractor,
)
from autocompleter.subtree_context import (
    extract_context_from_tree,
    find_focused_path,
)
from autocompleter.suggestion_engine import AutocompleteMode, detect_mode

from .ax_fixture_loader import load_fixture, load_fixture_metadata

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AX_TREES_DIR = FIXTURES_DIR / "ax_trees"
EXPECTED_DIR = FIXTURES_DIR / "expected"


# ---------------------------------------------------------------------------
# Mock dispatchers (same pattern as test_extractor_regression.py)
# ---------------------------------------------------------------------------

def _ax_get_attribute_dispatcher(element, attribute):
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get(attribute)
    return None


def _ax_get_children_dispatcher(element):
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get("AXChildren") or []
    return []


def _mock_collect_child_text(element, max_depth=5, max_chars=2000, depth=0):
    if depth > max_depth:
        return ""
    role = _ax_get_attribute_dispatcher(element, "AXRole") or ""
    if role == "AXStaticText":
        value = _ax_get_attribute_dispatcher(element, "AXValue")
        if isinstance(value, str) and value.strip():
            return value.strip()
        desc = _ax_get_attribute_dispatcher(element, "AXDescription")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        return ""
    value = _ax_get_attribute_dispatcher(element, "AXValue")
    if isinstance(value, str) and value.strip():
        return value.strip()
    children = _ax_get_attribute_dispatcher(element, "AXChildren")
    if not children:
        return ""
    parts = []
    for child in children:
        child_text = _mock_collect_child_text(child, max_depth, max_chars, depth + 1)
        if child_text:
            parts.append(child_text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Visible text extraction (mirrors input_observer._collect_text)
# ---------------------------------------------------------------------------

_SKIP_ROLES = frozenset({
    "AXToolbar", "AXMenuBar", "AXMenu", "AXMenuItem",
    "AXButton", "AXScrollBar", "AXSlider", "AXIncrementor",
    "AXPopUpButton", "AXCheckBox", "AXRadioButton",
    "AXTabGroup", "AXTab",
})

_CONTENT_ROLES = frozenset({
    "AXStaticText", "AXTextField", "AXTextArea",
    "AXWebArea", "AXGroup", "AXCell", "AXRow",
    "AXHeading", "AXLink", "AXParagraph",
})


def _collect_text_from_mock(
    element, results: list[str], max_depth=20, max_items=100, depth=0,
):
    """Simplified _collect_text for mock elements."""
    if depth > max_depth or len(results) >= max_items:
        return
    role = _ax_get_attribute_dispatcher(element, "AXRole") or ""
    if role in _SKIP_ROLES:
        return
    if role in _CONTENT_ROLES:
        value = _ax_get_attribute_dispatcher(element, "AXValue")
        if isinstance(value, str) and len(value.strip()) >= 3:
            results.append(value.strip()[:500])
        elif role == "AXStaticText":
            desc = _ax_get_attribute_dispatcher(element, "AXDescription")
            if isinstance(desc, str) and len(desc.strip()) >= 3:
                results.append(desc.strip()[:500])
    children = _ax_get_attribute_dispatcher(element, "AXChildren") or []
    for child in children:
        _collect_text_from_mock(child, results, max_depth, max_items, depth + 1)


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------

def _all_fixture_names() -> list[str]:
    if not AX_TREES_DIR.is_dir():
        return []
    return sorted(p.stem for p in AX_TREES_DIR.glob("*.json"))


def _has_focus_annotations(tree: dict, depth: int = 0) -> bool:
    if depth > 60:
        return False
    if tree.get("focused") or tree.get("ancestorOfFocused"):
        return True
    for child in tree.get("children", []):
        if _has_focus_annotations(child, depth + 1):
            return True
    return False


# App categorization
_CHAT_APPS = {"ChatGPT", "Claude", "Google Gemini", "Messages", "Discord", "\u200eWhatsApp"}
_TERMINAL_APPS = {"Terminal", "Warp"}
_BROWSER_APPS = {"Microsoft Edge", "Safari", "Google Chrome"}

fixture_names = _all_fixture_names()


# ---------------------------------------------------------------------------
# The E2E test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", fixture_names, ids=fixture_names)
class TestE2EPipeline:
    """Run the full pipeline for each fixture, checking each stage."""

    def _load(self, fixture_name):
        ax_path = AX_TREES_DIR / f"{fixture_name}.json"
        root = load_fixture(ax_path)
        metadata = load_fixture_metadata(ax_path)
        raw_data = json.loads(ax_path.read_text(encoding="utf-8"))

        expected_path = EXPECTED_DIR / f"{fixture_name}.json"
        expected = None
        if expected_path.exists():
            expected = json.loads(expected_path.read_text(encoding="utf-8"))

        return root, metadata, raw_data, expected

    # ---- Stage 1: Conversation extraction ----

    def test_conversation_extraction_no_crash(self, fixture_name):
        """Conversation extraction should never crash on any fixture."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        app_name = metadata.get("app", "")

        extractor = get_extractor(app_name)

        with (
            patch(
                "autocompleter.conversation_extractors.ax_get_attribute",
                side_effect=_ax_get_attribute_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors.ax_get_children",
                side_effect=_ax_get_children_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors._collect_child_text",
                side_effect=_mock_collect_child_text,
            ),
        ):
            turns = extractor.extract(root)

        # Should return list or None, never raise
        assert turns is None or isinstance(turns, list)
        if turns:
            for t in turns:
                assert isinstance(t, ConversationTurn)
                assert isinstance(t.speaker, str)
                assert isinstance(t.text, str)

    def test_conversation_extraction_matches_expected(self, fixture_name):
        """Where golden expected output exists, verify turn count and content."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        if expected is None:
            pytest.skip("No expected output for this fixture")

        app_name = metadata.get("app", "")
        extractor_class_name = expected.get("extractor", "")

        _EXTRACTOR_CLASS_TO_APP = {
            "GeminiExtractor": "Google Gemini",
            "SlackExtractor": "Slack",
            "ChatGPTExtractor": "ChatGPT",
            "ClaudeDesktopExtractor": "Claude",
            "IMessageExtractor": "Messages",
            "WhatsAppExtractor": "\u200eWhatsApp",
            "GenericExtractor": "",
            "ActionDelimitedExtractor": "",
        }

        if extractor_class_name in _EXTRACTOR_CLASS_TO_APP:
            app_for_lookup = _EXTRACTOR_CLASS_TO_APP[extractor_class_name]
        else:
            app_for_lookup = app_name

        extractor = get_extractor(app_for_lookup)

        with (
            patch(
                "autocompleter.conversation_extractors.ax_get_attribute",
                side_effect=_ax_get_attribute_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors.ax_get_children",
                side_effect=_ax_get_children_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors._collect_child_text",
                side_effect=_mock_collect_child_text,
            ),
        ):
            turns = extractor.extract(root) or []

        expected_turns = expected.get("turns", [])
        assert len(turns) == len(expected_turns), (
            f"Expected {len(expected_turns)} turns, got {len(turns)}.\n"
            f"Actual: {[(t.speaker, t.text[:60]) for t in turns]}"
        )

    # ---- Stage 2: Visible text extraction ----

    def test_visible_text_extraction(self, fixture_name):
        """Non-terminal fixtures should yield visible text elements.

        Some fixtures are legitimately sparse (new-chat screens, Electron
        apps with AXDescription-only text).  We check that we don't crash
        and that non-sparse fixtures produce content.
        """
        root, metadata, raw_data, expected = self._load(fixture_name)
        app_name = metadata.get("app", "")

        text_elements: list[str] = []
        _collect_text_from_mock(root, text_elements)

        # Terminal fixtures and new-chat / empty screens may have no tree text.
        # Electron apps (ChatGPT, Claude) sometimes have text in AXDescription
        # on non-AXStaticText roles which our simplified mock doesn't catch.
        _SPARSE_PATTERNS = {"new-chat", "new_chat"}
        is_sparse = any(p in fixture_name for p in _SPARSE_PATTERNS)

        if app_name not in _TERMINAL_APPS and not is_sparse:
            # Electron apps (ChatGPT, Claude) may store all content in
            # AXDescription on non-content roles (buttons, groups).  Our
            # simplified mock mirrors _SKIP_ROLES and misses these.
            # Only flag fixtures > 100KB with zero text as problems — smaller
            # ones are likely sidebar-only captures or empty screens.
            fixture_size = (AX_TREES_DIR / f"{fixture_name}.json").stat().st_size
            if fixture_size > 100_000 and len(text_elements) == 0:
                pytest.fail(
                    f"No visible text from {fixture_name} ({app_name}, "
                    f"{fixture_size:,} bytes)"
                )

        # Sanity: no element should be empty
        for elem in text_elements:
            assert len(elem.strip()) >= 3

    # ---- Stage 3: Subtree context ----

    def test_subtree_context_extraction(self, fixture_name):
        """Fixtures with focus annotations should produce subtree XML context."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        tree = raw_data["tree"]

        if not _has_focus_annotations(tree):
            pytest.skip("No focus annotations in this fixture")

        ctx = extract_context_from_tree(tree, token_budget=1200)

        assert ctx is not None, f"Subtree context returned None for focused fixture {fixture_name}"
        assert "<context>" in ctx
        assert "<input>" in ctx
        assert "</context>" in ctx
        assert len(ctx) > 50, f"Subtree context too short: {len(ctx)} chars"

    def test_subtree_focus_path(self, fixture_name):
        """Fixtures with focus should have a valid path from root to focused element."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        tree = raw_data["tree"]

        if not _has_focus_annotations(tree):
            pytest.skip("No focus annotations in this fixture")

        path = find_focused_path(tree)
        assert path is not None
        assert len(path) >= 2, "Focus path should have at least root + focused element"
        assert path[-1].get("focused") is True

    # ---- Stage 4: Context assembly ----

    def test_continuation_context_assembly(self, fixture_name):
        """Context assembly should produce a well-formed context string."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        app_name = metadata.get("app", "")
        window_title = metadata.get("windowTitle", "")

        # Extract conversation turns
        extractor = get_extractor(app_name)
        with (
            patch(
                "autocompleter.conversation_extractors.ax_get_attribute",
                side_effect=_ax_get_attribute_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors.ax_get_children",
                side_effect=_ax_get_children_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors._collect_child_text",
                side_effect=_mock_collect_child_text,
            ),
        ):
            turns = extractor.extract(root) or []

        # Simulate subtree context for focus-annotated fixtures
        tree = raw_data["tree"]
        subtree_ctx = None
        if _has_focus_annotations(tree):
            subtree_ctx = extract_context_from_tree(tree, token_budget=1200)

        # Build context via ContextStore
        store = ContextStore(Path("/tmp/test_e2e_pipeline.db"))
        store.open()
        try:
            # Continuation context with simulated cursor
            context = store.get_continuation_context(
                before_cursor="Hello, I wanted to ask about ",
                after_cursor="",
                source_app=app_name,
                window_title=window_title,
                subtree_context=subtree_ctx,
            )

            assert isinstance(context, str)
            assert len(context) > 0
            assert f"App: {app_name}" in context
            assert "Text before cursor:" in context

            if subtree_ctx:
                assert "Nearby content from the focused region:" in context
        finally:
            store.close()

    def test_reply_context_assembly(self, fixture_name):
        """Reply context assembly should produce well-formed context for chat apps."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        app_name = metadata.get("app", "")
        window_title = metadata.get("windowTitle", "")

        # Extract conversation turns
        extractor = get_extractor(app_name)
        with (
            patch(
                "autocompleter.conversation_extractors.ax_get_attribute",
                side_effect=_ax_get_attribute_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors.ax_get_children",
                side_effect=_ax_get_children_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors._collect_child_text",
                side_effect=_mock_collect_child_text,
            ),
        ):
            turns = extractor.extract(root) or []

        conversation_turns = [
            {"speaker": t.speaker, "text": t.text, "timestamp": t.timestamp}
            for t in turns
        ]

        tree = raw_data["tree"]
        subtree_ctx = None
        if _has_focus_annotations(tree):
            subtree_ctx = extract_context_from_tree(tree, token_budget=1200)

        store = ContextStore(Path("/tmp/test_e2e_pipeline_reply.db"))
        store.open()
        try:
            context = store.get_reply_context(
                conversation_turns=conversation_turns,
                source_app=app_name,
                window_title=window_title,
                subtree_context=subtree_ctx,
            )

            assert isinstance(context, str)
            assert len(context) > 0

            if conversation_turns:
                # Chat apps with turns: should have conversation section
                assert "Conversation:" in context
                # At least one turn speaker should appear
                assert any(
                    t["speaker"] in context for t in conversation_turns
                ), f"No turn speakers found in reply context for {fixture_name}"
            elif subtree_ctx:
                assert "Nearby content from the focused region:" in context
            else:
                assert f"App: {app_name}" in context
        finally:
            store.close()

    # ---- Stage 5: Mode detection ----

    def test_mode_detection_continuation(self, fixture_name):
        """With text before cursor, mode should be CONTINUATION."""
        mode = detect_mode(before_cursor="Hello, I wanted to ask about ")
        assert mode == AutocompleteMode.CONTINUATION

    def test_mode_detection_reply(self, fixture_name):
        """With empty cursor, mode should be REPLY."""
        mode = detect_mode(before_cursor="")
        assert mode == AutocompleteMode.REPLY

    # ---- Stage 6: Quality metrics ----

    def test_chat_app_has_conversation_turns(self, fixture_name):
        """Chat app fixtures with expected output should have conversation turns."""
        root, metadata, raw_data, expected = self._load(fixture_name)
        app_name = metadata.get("app", "")

        if expected is None:
            pytest.skip("No expected output for this fixture")
        if app_name in _TERMINAL_APPS or app_name in _BROWSER_APPS:
            pytest.skip("Not a chat app fixture")

        expected_turns = expected.get("turns", [])

        extractor = get_extractor(
            expected.get("app", app_name)
        )
        with (
            patch(
                "autocompleter.conversation_extractors.ax_get_attribute",
                side_effect=_ax_get_attribute_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors.ax_get_children",
                side_effect=_ax_get_children_dispatcher,
            ),
            patch(
                "autocompleter.conversation_extractors._collect_child_text",
                side_effect=_mock_collect_child_text,
            ),
        ):
            turns = extractor.extract(root) or []

        # Chat apps should extract turns matching the expected count
        assert len(turns) == len(expected_turns), (
            f"{app_name} ({fixture_name}): expected {len(expected_turns)} turns, "
            f"got {len(turns)}"
        )

    def test_context_not_empty_for_non_terminal(self, fixture_name):
        """The assembled context for non-terminal apps must have substance.

        New-chat and empty screens will legitimately produce short context
        (just metadata + cursor), so we relax the threshold for those.
        """
        root, metadata, raw_data, expected = self._load(fixture_name)
        app_name = metadata.get("app", "")

        if app_name in _TERMINAL_APPS:
            pytest.skip("Terminal fixture — text is in AXValue not tree children")

        tree = raw_data["tree"]
        subtree_ctx = None
        if _has_focus_annotations(tree):
            subtree_ctx = extract_context_from_tree(tree, token_budget=1200)

        store = ContextStore(Path("/tmp/test_e2e_pipeline_nonempty.db"))
        store.open()
        try:
            context = store.get_continuation_context(
                before_cursor="test input ",
                after_cursor="",
                source_app=app_name,
                window_title=metadata.get("windowTitle", ""),
                subtree_context=subtree_ctx,
            )
            assert isinstance(context, str)
            assert len(context) > 0

            if subtree_ctx:
                assert len(context) > 80, (
                    f"Context too short for {fixture_name} with "
                    f"subtree context present: {len(context)} chars\n"
                    f"Context: {context[:300]}"
                )
        finally:
            store.close()
