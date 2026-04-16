"""Tests for the subtree context extractor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocompleter.subtree_context import (
    build_context_bundle_from_tree,
    CHROME_ROLES,
    CONTENT_ROLES,
    count_text_nodes,
    extract_context_from_tree,
    extract_focus_path_overview,
    find_focused_path,
    subtree_to_xml,
    text_char_count,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ax_trees"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_focus_annotations(tree: dict) -> bool:
    """Check if a tree has focused/ancestorOfFocused annotations."""
    if tree.get("focused") or tree.get("ancestorOfFocused"):
        return True
    for child in tree.get("children", []):
        if _has_focus_annotations(child):
            return True
    return False


# ---------------------------------------------------------------------------
# find_focused_path
# ---------------------------------------------------------------------------

class TestFindFocusedPath:
    def test_simple_path(self):
        tree = {
            "role": "AXWindow", "children": [
                {"role": "AXGroup", "ancestorOfFocused": True, "children": [
                    {"role": "AXTextArea", "focused": True, "value": "hello", "children": []},
                ]},
            ],
        }
        path = find_focused_path(tree)
        assert path is not None
        assert len(path) == 3
        assert path[0]["role"] == "AXWindow"
        assert path[1]["role"] == "AXGroup"
        assert path[2]["role"] == "AXTextArea"
        assert path[2].get("focused") is True

    def test_no_focused_returns_none(self):
        tree = {"role": "AXWindow", "children": [{"role": "AXGroup", "children": []}]}
        assert find_focused_path(tree) is None

    def test_deep_path(self):
        # Build a 10-deep chain
        node = {"role": "AXTextArea", "focused": True, "value": "x", "children": []}
        for i in range(9):
            node = {"role": "AXGroup", "ancestorOfFocused": True, "children": [node]}
        root = {"role": "AXWindow", "ancestorOfFocused": True, "children": [node]}
        path = find_focused_path(root)
        assert path is not None
        assert len(path) == 11
        assert path[-1].get("focused") is True


# ---------------------------------------------------------------------------
# count_text_nodes / text_char_count
# ---------------------------------------------------------------------------

class TestCountHelpers:
    def test_count_text_nodes_skips_chrome(self):
        tree = {
            "role": "AXGroup", "children": [
                {"role": "AXStaticText", "value": "hello", "children": []},
                {"role": "AXButton", "value": "click", "children": []},
                {"role": "AXStaticText", "value": "world", "children": []},
            ]
        }
        assert count_text_nodes(tree) == 2

    def test_text_char_count(self):
        tree = {
            "role": "AXGroup", "children": [
                {"role": "AXStaticText", "value": "hello", "children": []},
                {"role": "AXStaticText", "value": "world!", "children": []},
            ]
        }
        assert text_char_count(tree) == 11  # "hello" + "world!"

    def test_empty_tree(self):
        tree = {"role": "AXGroup", "children": []}
        assert count_text_nodes(tree) == 0
        assert text_char_count(tree) == 0


# ---------------------------------------------------------------------------
# subtree_to_xml
# ---------------------------------------------------------------------------

class TestSubtreeToXml:
    def test_static_text(self):
        node = {"role": "AXStaticText", "value": "hello world", "title": "", "children": []}
        xml = subtree_to_xml(node)
        assert "<StaticText>hello world</StaticText>" in xml

    def test_chrome_skipped(self):
        node = {"role": "AXButton", "value": "Click me", "children": []}
        assert subtree_to_xml(node) == ""

    def test_collapses_single_child_wrapper(self):
        tree = {
            "role": "AXGroup", "title": "", "children": [
                {"role": "AXStaticText", "value": "inner", "title": "", "children": []},
            ],
        }
        xml = subtree_to_xml(tree)
        # The AXGroup should be collapsed, just the text
        assert "Group" not in xml
        assert "<StaticText>inner</StaticText>" in xml

    def test_preserves_titled_container(self):
        tree = {
            "role": "AXGroup", "title": "Messages", "children": [
                {"role": "AXStaticText", "value": "hello", "title": "", "children": []},
            ],
        }
        xml = subtree_to_xml(tree)
        assert 'title="Messages"' in xml

    def test_focused_attribute(self):
        node = {
            "role": "AXTextArea", "value": "draft", "title": "",
            "focused": True, "children": [],
        }
        xml = subtree_to_xml(node)
        assert 'focused="true"' in xml

    def test_focused_empty_input_is_preserved(self):
        node = {
            "role": "AXTextArea",
            "value": "",
            "title": "",
            "focused": True,
            "placeholderDetected": True,
            "cursorPosition": 0,
            "selectionLength": 0,
            "valueLength": 0,
            "children": [],
        }
        xml = subtree_to_xml(node)
        assert "<TextArea" in xml
        assert 'placeholder_detected="true"' in xml
        assert 'cursor="0"' in xml
        assert 'value_length="0"' in xml

    def test_escapes_special_chars(self):
        node = {"role": "AXStaticText", "value": '<script>"alert"</script>', "title": "", "children": []}
        xml = subtree_to_xml(node)
        assert "&lt;" in xml
        assert "&gt;" in xml
        assert "&quot;" in xml

    def test_empty_subtree_returns_empty(self):
        tree = {
            "role": "AXGroup", "title": "", "children": [
                {"role": "AXGroup", "title": "", "children": []},
            ],
        }
        assert subtree_to_xml(tree) == ""

    def test_max_depth_respected(self):
        # 5-deep chain with content at the bottom
        node = {"role": "AXStaticText", "value": "deep", "title": "", "children": []}
        for _ in range(5):
            node = {"role": "AXGroup", "title": "", "children": [node]}
        # max_depth=3 should not reach the text
        xml = subtree_to_xml(node, max_depth=3)
        assert xml == ""


# ---------------------------------------------------------------------------
# extract_context_from_tree — synthetic trees
# ---------------------------------------------------------------------------

class TestExtractContextSynthetic:
    def test_basic_extraction(self):
        tree = {
            "role": "AXWindow", "title": "", "ancestorOfFocused": True,
            "children": [
                {"role": "AXStaticText", "value": "Previous message", "title": "", "children": []},
                {
                    "role": "AXGroup", "title": "", "ancestorOfFocused": True,
                    "children": [
                        {"role": "AXStaticText", "value": "Nearby content", "title": "", "children": []},
                        {"role": "AXTextArea", "value": "my draft", "title": "", "focused": True, "children": []},
                    ],
                },
            ],
        }
        xml = extract_context_from_tree(tree)
        assert xml is not None
        assert "<input>" in xml
        assert "my draft" in xml
        assert "Nearby content" in xml

    def test_no_focused_returns_none(self):
        tree = {"role": "AXWindow", "children": []}
        assert extract_context_from_tree(tree) is None

    def test_chrome_excluded(self):
        tree = {
            "role": "AXWindow", "title": "", "ancestorOfFocused": True,
            "children": [
                {"role": "AXButton", "value": "Submit", "title": "Submit", "children": []},
                {"role": "AXToolbar", "title": "toolbar", "children": [
                    {"role": "AXStaticText", "value": "toolbar text", "title": "", "children": []},
                ]},
                {"role": "AXTextArea", "value": "draft", "title": "", "focused": True, "children": []},
            ],
        }
        xml = extract_context_from_tree(tree)
        assert xml is not None
        assert "Submit" not in xml
        assert "toolbar" not in xml

    def test_token_budget_limits_output(self):
        # Build a tree with a lot of content siblings
        siblings = []
        for i in range(20):
            siblings.append({
                "role": "AXStaticText",
                "value": f"Message {i}: " + "x" * 200,
                "title": "",
                "children": [],
            })
        siblings.append({
            "role": "AXTextArea", "value": "draft", "title": "",
            "focused": True, "children": [],
        })
        tree = {
            "role": "AXWindow", "title": "", "ancestorOfFocused": True,
            "children": siblings,
        }
        xml = extract_context_from_tree(tree, token_budget=100)
        assert xml is not None
        # Should not contain all 20 messages
        assert xml.count("Message") < 20

    def test_focus_path_overview_includes_focused_metadata(self):
        tree = {
            "role": "AXWindow",
            "title": "Codex",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXGroup",
                    "description": "Automation folders",
                    "children": [
                        {"role": "AXStaticText", "value": "Threads", "children": []},
                    ],
                },
                {
                    "role": "AXGroup",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXTextArea",
                            "value": "",
                            "focused": True,
                            "placeholderDetected": True,
                            "cursorPosition": 0,
                            "selectionLength": 0,
                            "valueLength": 0,
                            "children": [],
                        },
                    ],
                },
            ],
        }
        overview = extract_focus_path_overview(tree, token_budget=120)
        assert overview is not None
        assert "<focusPath>" in overview
        assert 'placeholder_detected="true"' in overview
        assert "Threads" in overview

    def test_context_bundle_contains_both_views(self):
        tree = {
            "role": "AXWindow",
            "title": "Codex",
            "ancestorOfFocused": True,
            "children": [
                {"role": "AXStaticText", "value": "Recent thread", "children": []},
                {
                    "role": "AXGroup",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXTextArea",
                            "value": "",
                            "focused": True,
                            "placeholderDetected": True,
                            "cursorPosition": 0,
                            "selectionLength": 0,
                            "valueLength": 0,
                            "children": [],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=120, overview_token_budget=80)
        assert bundle is not None
        assert bundle.top_down_context is not None
        assert bundle.bottom_up_context is not None
        assert "Recent thread" in bundle.bottom_up_context
        assert "<focusPath>" in bundle.top_down_context

    def test_transcript_branch_beats_sidebar_chrome(self):
        tree = {
            "role": "AXWindow",
            "title": "Codex",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXGroup",
                    "description": "Automation folders",
                    "children": [
                        {"role": "AXStaticText", "value": "Threads", "children": []},
                        {"role": "AXStaticText", "value": "Clean up app pipeline", "children": []},
                    ],
                },
                {
                    "role": "AXWebArea",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXGroup",
                            "children": [
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {
                                            "role": "AXStaticText",
                                            "value": "The stall was caused by the worker thread never starting the generation call.",
                                            "children": [],
                                        },
                                        {"role": "AXStaticText", "value": "3:03 PM", "children": []},
                                        {"role": "AXButton", "description": "Copy message", "children": []},
                                        {"role": "AXButton", "description": "Fork from this message", "children": []},
                                    ],
                                },
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {
                                            "role": "AXStaticText",
                                            "value": "The root cause was still present in normal trigger paths, so I fixed the remaining launch sites too.",
                                            "children": [],
                                        },
                                        {"role": "AXStaticText", "value": "3:04 PM", "children": []},
                                        {"role": "AXButton", "description": "Copy message", "children": []},
                                    ],
                                },
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "",
                                    "focused": True,
                                    "placeholderDetected": True,
                                    "cursorPosition": 0,
                                    "selectionLength": 0,
                                    "valueLength": 0,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=200, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "The stall was caused" in bundle.bottom_up_context
        assert "The root cause was still present" in bundle.bottom_up_context
        assert "Threads" not in bundle.bottom_up_context
        assert "Clean up app pipeline" not in bundle.bottom_up_context
        assert bundle.selection_debug is not None
        assert bundle.selection_debug["strategy"] == "transcript_branch"
        assert any(item["selected"] for item in bundle.selection_debug["messageObjects"])

    def test_long_message_keeps_multiple_recent_message_objects(self):
        tree = {
            "role": "AXWindow",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXWebArea",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXGroup",
                            "children": [
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {
                                            "role": "AXStaticText",
                                            "value": "Earlier summary: " + ("really long context " * 80),
                                            "children": [],
                                        },
                                        {"role": "AXStaticText", "value": "2:59 PM", "children": []},
                                    ],
                                },
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {
                                            "role": "AXStaticText",
                                            "value": "Recent fix: I updated the evaluate_invocations path to use the richer dump schema.",
                                            "children": [],
                                        },
                                        {"role": "AXStaticText", "value": "3:00 PM", "children": []},
                                    ],
                                },
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {
                                            "role": "AXStaticText",
                                            "value": "Latest note: the next capture should show the selected transcript rows cleanly.",
                                            "children": [],
                                        },
                                        {"role": "AXStaticText", "value": "3:01 PM", "children": []},
                                    ],
                                },
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "why is it still doing that ",
                                    "focused": True,
                                    "cursorPosition": 27,
                                    "selectionLength": 0,
                                    "valueLength": 27,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=180, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "Latest note" in bundle.bottom_up_context
        assert "Recent fix" in bundle.bottom_up_context
        selected = [
            item for item in (bundle.selection_debug or {}).get("messageObjects", [])
            if item.get("selected")
        ]
        assert len(selected) >= 2

    def test_generic_chat_transcript_uses_ordered_recent_rows(self):
        tree = {
            "role": "AXWindow",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXGroup",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXList",
                            "subrole": "AXContentList",
                            "children": [
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {"role": "AXStaticText", "value": "oldest message in the thread", "children": []},
                                        {"role": "AXStaticText", "value": "9:41 AM", "children": []},
                                    ],
                                },
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {"role": "AXStaticText", "value": "middle message with useful context", "children": []},
                                        {"role": "AXStaticText", "value": "9:42 AM", "children": []},
                                    ],
                                },
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {"role": "AXStaticText", "value": "latest reply closest to the draft input", "children": []},
                                        {"role": "AXStaticText", "value": "9:43 AM", "children": []},
                                    ],
                                },
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "draft reply",
                                    "focused": True,
                                    "cursorPosition": 11,
                                    "selectionLength": 0,
                                    "valueLength": 11,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=160, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "latest reply closest to the draft input" in bundle.bottom_up_context
        assert "middle message with useful context" in bundle.bottom_up_context
        assert bundle.selection_debug is not None
        assert bundle.selection_debug["strategy"] == "transcript_branch"

    def test_flat_transcript_stream_groups_messages_around_timestamps(self):
        tree = {
            "role": "AXWindow",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXGroup",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXGroup",
                            "children": [
                                {"role": "AXGroup", "children": [
                                    {"role": "AXStaticText", "value": "we should include multiple message-like objects no?", "children": []},
                                    {"role": "AXStaticText", "value": "7:26 PM", "children": []},
                                    {"role": "AXButton", "description": "Copy message", "children": []},
                                    {"role": "AXButton", "description": "Fork from this message", "children": []},
                                    {"role": "AXGroup", "children": [
                                        {"role": "AXStaticText", "value": "Yes, I think so.", "children": []},
                                    ]},
                                    {"role": "AXGroup", "children": [
                                        {"role": "AXStaticText", "value": "The cleaner rule is:", "children": []},
                                    ]},
                                    {"role": "AXList", "children": [
                                        {"role": "AXStaticText", "value": "use one branch only", "children": []},
                                        {"role": "AXStaticText", "value": "but include multiple message-like objects when available", "children": []},
                                    ]},
                                    {"role": "AXStaticText", "value": "7:28 PM", "children": []},
                                    {"role": "AXButton", "description": "Copy", "children": []},
                                ]},
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "",
                                    "focused": True,
                                    "placeholderDetected": True,
                                    "cursorPosition": 0,
                                    "selectionLength": 0,
                                    "valueLength": 0,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=180, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "we should include multiple message-like objects no?" in bundle.bottom_up_context
        assert "The cleaner rule is:" in bundle.bottom_up_context
        assert bundle.bottom_up_context.count("7:26 PM") <= 1
        selected = [
            item for item in (bundle.selection_debug or {}).get("messageObjects", [])
            if item.get("selected")
        ]
        assert len(selected) >= 2

    def test_transcript_branch_emits_clean_message_blocks(self):
        tree = {
            "role": "AXWindow",
            "title": "Codex",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXGroup",
                    "description": "Automation folders",
                    "children": [
                        {"role": "AXStaticText", "value": "Threads", "children": []},
                    ],
                },
                {
                    "role": "AXWebArea",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXGroup",
                            "children": [
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {
                                            "role": "AXStaticText",
                                            "value": "Yes. Since",
                                            "children": [],
                                        },
                                        {
                                            "role": "AXStaticText",
                                            "value": "venv/",
                                            "children": [],
                                        },
                                        {
                                            "role": "AXStaticText",
                                            "value": "is gone, here’s the clean repo-local rebuild path.",
                                            "children": [],
                                        },
                                        {"role": "AXStaticText", "value": "4:05 PM", "children": []},
                                        {"role": "AXButton", "description": "Copy message", "children": []},
                                        {"role": "AXStaticText", "value": "Local", "children": []},
                                        {"role": "AXStaticText", "value": "main", "children": []},
                                        {"role": "AXStaticText", "value": "Full access", "children": []},
                                    ],
                                },
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "",
                                    "focused": True,
                                    "placeholderDetected": True,
                                    "cursorPosition": 0,
                                    "selectionLength": 0,
                                    "valueLength": 0,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=200, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "<Message>" in bundle.bottom_up_context
        assert "Yes. Since venv/ is gone, here’s the clean repo-local rebuild path." in bundle.bottom_up_context
        assert "Copy message" not in bundle.bottom_up_context
        assert "4:05 PM" not in bundle.bottom_up_context
        assert "Full access" not in bundle.bottom_up_context
        assert "<Group" not in bundle.bottom_up_context
        assert bundle.top_down_context is not None
        assert "Threads" not in bundle.top_down_context

    def test_transcript_branch_keeps_message_boundaries_when_merging_fragments(self):
        tree = {
            "role": "AXWindow",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXWebArea",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXGroup",
                            "children": [
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {"role": "AXStaticText", "value": "First reply starts", "children": []},
                                        {"role": "AXStaticText", "value": "here.", "children": []},
                                    ],
                                },
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {"role": "AXStaticText", "value": "Second reply starts", "children": []},
                                        {"role": "AXStaticText", "value": "there.", "children": []},
                                    ],
                                },
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "",
                                    "focused": True,
                                    "placeholderDetected": True,
                                    "cursorPosition": 0,
                                    "selectionLength": 0,
                                    "valueLength": 0,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=180, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "First reply starts here." in bundle.bottom_up_context
        assert "Second reply starts there." in bundle.bottom_up_context
        assert "here. Second reply starts" not in bundle.bottom_up_context
        assert bundle.bottom_up_context.count("<Message>") >= 2

    def test_transcript_branch_middle_trims_long_messages(self):
        long_message = (
            "Message start with important setup. "
            + ("middle filler " * 90)
            + "Message ending with the key ask."
        )
        tree = {
            "role": "AXWindow",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXWebArea",
                    "ancestorOfFocused": True,
                    "children": [
                        {
                            "role": "AXGroup",
                            "children": [
                                {
                                    "role": "AXGroup",
                                    "children": [
                                        {"role": "AXStaticText", "value": long_message, "children": []},
                                    ],
                                },
                            ],
                        },
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "",
                                    "focused": True,
                                    "placeholderDetected": True,
                                    "cursorPosition": 0,
                                    "selectionLength": 0,
                                    "valueLength": 0,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=180, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert "Message start with important setup." in bundle.bottom_up_context
        assert "Message ending with the key ask." in bundle.bottom_up_context
        assert "[trimmed middle" in bundle.bottom_up_context

    def test_transcript_branch_caps_recent_messages_to_last_six(self):
        transcript_children = []
        for i in range(8):
            transcript_children.append(
                {
                    "role": "AXGroup",
                    "children": [
                        {"role": "AXStaticText", "value": f"message {i}", "children": []},
                        {"role": "AXStaticText", "value": f"9:4{i} AM", "children": []},
                    ],
                }
            )
        tree = {
            "role": "AXWindow",
            "ancestorOfFocused": True,
            "children": [
                {
                    "role": "AXWebArea",
                    "ancestorOfFocused": True,
                    "children": [
                        {"role": "AXGroup", "children": transcript_children},
                        {
                            "role": "AXGroup",
                            "ancestorOfFocused": True,
                            "children": [
                                {
                                    "role": "AXTextArea",
                                    "value": "",
                                    "focused": True,
                                    "placeholderDetected": True,
                                    "cursorPosition": 0,
                                    "selectionLength": 0,
                                    "valueLength": 0,
                                    "children": [],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        bundle = build_context_bundle_from_tree(tree, token_budget=1000, overview_token_budget=80)
        assert bundle is not None
        assert bundle.bottom_up_context is not None
        assert bundle.bottom_up_context.count("<Message>") <= 6
        assert "message 7" in bundle.bottom_up_context
        assert "message 2" in bundle.bottom_up_context
        assert "message 0" not in bundle.bottom_up_context
        assert "message 1" not in bundle.bottom_up_context


# ---------------------------------------------------------------------------
# extract_context_from_tree — real fixtures
# ---------------------------------------------------------------------------

class TestExtractContextFixtures:
    """Test against real AX tree captures that have focus annotations."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_fixtures(self):
        if not FIXTURES.exists():
            pytest.skip("No fixtures directory")

    def _fixtures_with_focus(self):
        """Yield (name, data) for fixtures that have focus annotations."""
        for path in sorted(FIXTURES.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            tree = data.get("tree", {})
            if _has_focus_annotations(tree):
                yield path.stem, data

    def test_all_focused_fixtures_produce_output(self):
        """Every fixture with focus annotations should produce non-empty XML."""
        found = False
        for name, data in self._fixtures_with_focus():
            found = True
            xml = extract_context_from_tree(data["tree"])
            assert xml is not None, f"{name}: expected XML output"
            assert "<context>" in xml, f"{name}: missing <context> root"
            assert "<input>" in xml, f"{name}: missing <input> section"
        if not found:
            pytest.skip("No fixtures with focus annotations found")

    def test_focused_input_contains_value(self):
        """The <input> section should contain the focused element's text."""
        for name, data in self._fixtures_with_focus():
            focused_value = (data.get("focusedElement") or {}).get("value", "")
            if not focused_value or not focused_value.strip():
                continue
            xml = extract_context_from_tree(data["tree"])
            assert xml is not None
            # The input section should contain (part of) the focused value
            input_start = xml.find("<input>")
            input_end = xml.find("</input>")
            input_section = xml[input_start:input_end]
            # Check first 20 chars of the value appear
            check_text = focused_value.strip()[:20]
            assert check_text in input_section or check_text in xml, (
                f"{name}: focused value {check_text!r} not in input section"
            )

    def test_no_chrome_in_output(self):
        """Chrome roles should never appear in the output XML."""
        chrome_tags = {role.removeprefix("AX") for role in CHROME_ROLES}
        for name, data in self._fixtures_with_focus():
            xml = extract_context_from_tree(data["tree"])
            if xml is None:
                continue
            for tag in chrome_tags:
                assert f"<{tag}" not in xml, (
                    f"{name}: chrome tag <{tag}> found in output"
                )

    def test_gemini_captures_conversation(self):
        """Gemini fixtures with conversation should include speaker headings."""
        for suffix in ("8", "9"):
            name = f"google-gemini-{suffix}"
            data = _load(name)
            if not _has_focus_annotations(data["tree"]):
                pytest.skip(f"{name} has no focus annotations")
            xml = extract_context_from_tree(data["tree"])
            assert xml is not None
            assert "You said" in xml, f"{name}: missing 'You said' heading"
            assert "Gemini said" in xml, f"{name}: missing 'Gemini said' heading"

    def test_gmail_captures_email_body(self):
        """Gmail (Edge) fixtures should include email content."""
        for suffix in ("5", "6"):
            name = f"microsoft-edge-{suffix}"
            data = _load(name)
            if not _has_focus_annotations(data["tree"]):
                pytest.skip(f"{name} has no focus annotations")
            xml = extract_context_from_tree(data["tree"])
            assert xml is not None
            # Should contain email sender and some body text
            assert "Tal Gur" in xml, f"{name}: missing sender name"
            assert "Elevate Society" in xml, f"{name}: missing email body content"

    def test_codex_fixture_uses_cleaned_transcript_messages(self):
        data = _load("codex-transcript")
        bundle = build_context_bundle_from_tree(
            data["tree"],
            token_budget=200,
            overview_token_budget=80,
        )
        assert bundle is not None
        assert bundle.selection_debug is not None
        assert bundle.selection_debug["strategy"] == "transcript_branch"
        assert bundle.bottom_up_context is not None
        assert "<Message>" in bundle.bottom_up_context
        assert "Yes. Since venv/ is gone, here’s the clean repo-local rebuild path." in bundle.bottom_up_context
        assert "yeah and we deleted the venv so let's reconfigure the environment." in bundle.bottom_up_context
        assert "how do i leave a venv again" in bundle.bottom_up_context
        assert "Copy message" not in bundle.bottom_up_context
        assert "Full access" not in bundle.bottom_up_context
        assert "Local" not in bundle.bottom_up_context
        assert "Threads" not in bundle.bottom_up_context
        assert bundle.top_down_context is not None
        assert "<focusPath>" in bundle.top_down_context

    def test_claude_fixture_prefers_active_transcript_over_sidebar_recents(self):
        for name, expected_text in (
            ("claude-9", "usb-a sources just dumbly output 5v without any negotiation"),
            ("claude-10", "&quot;remote&quot; is the general concept"),
        ):
            data = _load(name)
            bundle = build_context_bundle_from_tree(
                data["tree"],
                token_budget=500,
                overview_token_budget=80,
            )
            assert bundle is not None
            assert bundle.selection_debug is not None
            assert bundle.selection_debug["strategy"] == "transcript_branch"
            assert bundle.bottom_up_context is not None
            assert expected_text in bundle.bottom_up_context
            assert "Killing a tmux session from within" not in bundle.bottom_up_context
            assert "App bundler cache corruption issues" not in bundle.bottom_up_context
            assert "Updated to 1.2278.0 Relaunch to apply" not in bundle.bottom_up_context

    def test_unfocused_fixtures_return_none(self):
        """Fixtures without focus annotations should return None gracefully."""
        for path in sorted(FIXTURES.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            tree = data.get("tree", {})
            if _has_focus_annotations(tree):
                continue
            result = extract_context_from_tree(tree)
            assert result is None, f"{path.stem}: expected None for unfocused tree"

    def test_output_within_budget(self):
        """Output should roughly respect the token budget."""
        budget = 300
        for name, data in self._fixtures_with_focus():
            xml = extract_context_from_tree(data["tree"], token_budget=budget)
            if xml is None:
                continue
            # Allow 50% overshoot (budget is approximate)
            max_chars = budget * 4 * 1.5
            assert len(xml) < max_chars, (
                f"{name}: output {len(xml)} chars exceeds budget "
                f"({budget} tokens ≈ {budget * 4} chars, max {max_chars:.0f})"
            )
