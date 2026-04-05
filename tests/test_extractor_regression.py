"""Regression tests for conversation extractors using captured AX tree fixtures.

Auto-discovers fixture pairs in ``tests/fixtures/``:
  - ``ax_trees/<name>.json``  — captured AX tree
  - ``expected/<name>.json``  — golden expected extraction result

If no fixtures exist, zero tests are discovered (no failure).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autocompleter.conversation_extractors import get_extractor

from .ax_fixture_loader import load_fixture, load_fixture_metadata

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AX_TREES_DIR = FIXTURES_DIR / "ax_trees"
EXPECTED_DIR = FIXTURES_DIR / "expected"

# Map extractor class names to app names for get_extractor()
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


def _discover_fixture_pairs() -> list[str]:
    """Return fixture names that have both ax_tree and expected files."""
    if not AX_TREES_DIR.is_dir() or not EXPECTED_DIR.is_dir():
        return []
    ax_names = {p.stem for p in AX_TREES_DIR.glob("*.json")}
    expected_names = {p.stem for p in EXPECTED_DIR.glob("*.json")}
    return sorted(ax_names & expected_names)


def _ax_get_attribute_dispatcher(element, attribute):
    """Dispatcher for patched ax_get_attribute using mock _ax_attrs."""
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get(attribute)
    return None


def _ax_get_children_dispatcher(element):
    """Dispatcher for patched ax_get_children using mock _ax_attrs."""
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get("AXChildren") or []
    return []


def _mock_collect_child_text(element, max_depth=5, max_chars=2000, depth=0):
    """Mock of _collect_child_text that mirrors real behavior.

    Checks AXValue first, then AXDescription for AXStaticText nodes
    (matching the real implementation's Electron/ChatGPT handling).
    """
    if depth > max_depth:
        return ""

    role = _ax_get_attribute_dispatcher(element, "AXRole") or ""

    # For AXStaticText: check value, then description (like real impl lines 82-92)
    if role == "AXStaticText":
        value = _ax_get_attribute_dispatcher(element, "AXValue")
        if isinstance(value, str) and value.strip():
            return value.strip()
        desc = _ax_get_attribute_dispatcher(element, "AXDescription")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        return ""

    # For non-AXStaticText with a value, use it
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


fixture_names = _discover_fixture_pairs()


@pytest.mark.parametrize("fixture_name", fixture_names, ids=fixture_names)
def test_extractor_regression(fixture_name: str):
    """Replay a captured AX tree through the appropriate extractor and
    verify the extracted turns match the golden expected output."""
    ax_path = AX_TREES_DIR / f"{fixture_name}.json"
    expected_path = EXPECTED_DIR / f"{fixture_name}.json"

    root = load_fixture(ax_path)
    metadata = load_fixture_metadata(ax_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    # Determine which extractor to use
    extractor_class_name = expected.get("extractor", "")
    app_name = expected.get("app") or metadata.get("app", "")

    if extractor_class_name and extractor_class_name in _EXTRACTOR_CLASS_TO_APP:
        app_for_lookup = _EXTRACTOR_CLASS_TO_APP[extractor_class_name]
    else:
        app_for_lookup = app_name

    extractor = get_extractor(app_for_lookup)

    # Verify we got the right extractor class if specified
    if extractor_class_name:
        actual_class = type(extractor).__name__
        assert actual_class == extractor_class_name, (
            f"Expected {extractor_class_name} for app={app_for_lookup!r}, "
            f"got {actual_class}"
        )

    # Run extraction with patched AX access
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

    # Assert turn count
    assert len(turns) == len(expected_turns), (
        f"Expected {len(expected_turns)} turns, got {len(turns)}.\n"
        f"Actual turns: {[(t.speaker, t.text[:60]) for t in turns]}"
    )

    # Assert each turn
    for i, (actual, exp) in enumerate(zip(turns, expected_turns)):
        # Speaker match
        if "speaker" in exp:
            assert actual.speaker == exp["speaker"], (
                f"Turn {i}: expected speaker={exp['speaker']!r}, "
                f"got {actual.speaker!r}"
            )

        # Text match: substring by default, exact via "text_exact"
        if "text_exact" in exp:
            assert actual.text == exp["text_exact"], (
                f"Turn {i}: exact text mismatch.\n"
                f"  Expected: {exp['text_exact']!r}\n"
                f"  Actual:   {actual.text!r}"
            )
        elif "text" in exp:
            assert exp["text"] in actual.text, (
                f"Turn {i}: expected substring {exp['text']!r} "
                f"not found in {actual.text!r}"
            )

        # Timestamp match (optional — only checked if present in expected)
        if "timestamp" in exp:
            assert actual.timestamp == exp["timestamp"], (
                f"Turn {i}: expected timestamp={exp['timestamp']!r}, "
                f"got {actual.timestamp!r}"
            )
