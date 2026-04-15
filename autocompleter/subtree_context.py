"""Subtree-based context extraction from AX trees.

Instead of per-app conversation extractors, this module walks *up* from the
focused element to collect nearby content.  The output is minimal XML that
an LLM can parse directly.

Algorithm
---------
1. Follow the ``ancestorOfFocused`` / ``focused`` breadcrumb trail from the
   root down to the keyboard-active element.
2. Starting from the focused element's parent, walk upward through ancestors.
3. At each ancestor, collect sibling subtrees that contain meaningful text
   (skipping UI chrome: buttons, scrollbars, toolbars, etc.).
4. Prioritise large, content-rich subtrees; skip navigation-like clusters
   (many small text nodes with low average character count).
5. Serialize collected subtrees as compact XML, collapsing empty wrapper
   nodes.
6. Stop when the token budget is exhausted or a natural boundary
   (``AXWebArea``) is reached.

Runtime usage
-------------
At runtime, the ``focused`` / ``ancestorOfFocused`` flags are set by
``ax_utils.serialize_ax_tree()`` when passed a ``focused_element``.  For
test fixtures captured with the updated ``dump_ax_tree_json.py``, these
flags are baked into the JSON.

For live AX elements (not serialized dicts), use ``extract_context_live()``
which calls ``serialize_ax_tree()`` and then runs the walker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

#: Roles that represent UI chrome — never contain user-relevant content.
CHROME_ROLES: frozenset[str] = frozenset({
    "AXButton",
    "AXCheckBox",
    "AXBusyIndicator",
    "AXImage",
    "AXMenu",
    "AXMenuBar",
    "AXMenuItem",
    "AXPopUpButton",
    "AXProgressIndicator",
    "AXScrollBar",
    "AXSlider",
    "AXTabGroup",
    "AXToolbar",
    "AXValueIndicator",
})

#: Roles that carry text content.
CONTENT_ROLES: frozenset[str] = frozenset({
    "AXHeading",
    "AXLink",
    "AXParagraph",
    "AXStaticText",
    "AXTextArea",
    "AXTextField",
})

#: Structural roles that might wrap content.
CONTAINER_ROLES: frozenset[str] = frozenset({
    "AXCell",
    "AXGroup",
    "AXList",
    "AXOutline",
    "AXRow",
    "AXScrollArea",
    "AXTable",
    "AXWebArea",
})

#: Natural tree boundaries — stop climbing past these when content is found.
BOUNDARY_ROLES: frozenset[str] = frozenset({"AXWebArea"})

# Minimum average characters per text node for a subtree to be considered
# "content" rather than navigation.
_MIN_AVG_CHARS_CONTENT = 15

# Minimum number of text nodes before the avg-chars heuristic kicks in.
_MIN_TEXT_NODES_FOR_NAV_CHECK = 5

_MIN_MESSAGE_TEXT_CHARS = 20
_LONG_MESSAGE_TEXT_CHARS = 80
_MAX_MESSAGE_OBJECT_CHARS = 700
_MAX_BRANCH_DEBUG_CANDIDATES = 8

_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b")
_ACTION_KEYWORDS = (
    "copy",
    "copy message",
    "fork",
    "reply",
    "react",
    "edit",
    "share",
    "regenerate",
)
_NAV_KEYWORDS = (
    "thread",
    "threads",
    "new chat",
    "search",
    "plugin",
    "plugins",
    "automation",
    "automations",
    "project",
    "projects",
    "folder",
    "folders",
    "filter sidebar",
    "filter chats",
)


@dataclass
class TreeContextBundle:
    """Focus-aware tree context used for prompt assembly and debugging."""

    tree: dict
    bottom_up_context: Optional[str]
    top_down_context: Optional[str]
    selection_debug: Optional[dict] = None


@dataclass(frozen=True)
class _SubtreeSummary:
    """Cheap semantic summary used by transcript/message heuristics."""

    text_chars: int = 0
    text_nodes: int = 0
    medium_text_nodes: int = 0
    long_text_nodes: int = 0
    timestamp_nodes: int = 0
    action_nodes: int = 0
    nav_nodes: int = 0
    control_nodes: int = 0
    container_nodes: int = 0
    content_list_nodes: int = 0


@dataclass
class _BranchCandidate:
    node: dict
    ancestor_distance: int
    score: float
    qualifies_as_transcript: bool
    summary: _SubtreeSummary
    preview: str


@dataclass
class _MessageObject:
    node: dict
    summary: _SubtreeSummary
    preview: str


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def find_focused_path(node: dict) -> Optional[list[dict]]:
    """Return the list of nodes from *root* to the ``focused`` node.

    Returns ``None`` if no focused node exists in the tree.
    """
    path: list[dict] = []

    def _walk(n: dict) -> bool:
        path.append(n)
        if n.get("focused"):
            return True
        for child in n.get("children", []):
            if child.get("ancestorOfFocused") or child.get("focused"):
                if _walk(child):
                    return True
        path.pop()
        return False

    return list(path) if _walk(node) else None


def count_text_nodes(node: dict, depth: int = 0, max_depth: int = 10) -> int:
    """Count text-bearing content nodes in a subtree, skipping chrome."""
    if depth > max_depth:
        return 0
    role = node.get("role", "")
    if role in CHROME_ROLES:
        return 0
    count = 0
    if role in CONTENT_ROLES:
        val = node.get("value")
        desc = node.get("description", "")
        if (isinstance(val, str) and val.strip()) or (
            isinstance(desc, str) and desc.strip()
        ):
            count += 1
    for child in node.get("children", []):
        count += count_text_nodes(child, depth + 1, max_depth)
    return count


def text_char_count(node: dict, depth: int = 0, max_depth: int = 10) -> int:
    """Total text characters in a subtree (content roles only)."""
    if depth > max_depth:
        return 0
    role = node.get("role", "")
    if role in CHROME_ROLES:
        return 0
    total = 0
    val = node.get("value")
    if isinstance(val, str) and val.strip():
        total += len(val.strip())
    for child in node.get("children", []):
        total += text_char_count(child, depth + 1, max_depth)
    return total


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _node_strings(node: dict) -> list[str]:
    strings: list[str] = []
    for key in ("value", "title", "description"):
        text = _normalize_text(node.get(key))
        if text and text not in strings:
            strings.append(text)
    return strings


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def _is_nav_like_node(node: dict) -> bool:
    role = node.get("role", "")
    subrole = _normalize_text(node.get("subrole", ""))
    role_desc = _normalize_text(node.get("roleDescription", ""))
    if "AXLandmarkNavigation" in subrole or "AXLandmarkComplementary" in subrole:
        return True
    combined = " ".join([role, subrole, role_desc, *_node_strings(node)]).lower()
    return _contains_keyword(combined, _NAV_KEYWORDS)


def _summarize_subtree(
    node: dict,
    memo: dict[int, _SubtreeSummary],
    depth: int = 0,
    max_depth: int = 14,
) -> _SubtreeSummary:
    """Compute a cached semantic summary for a subtree."""
    cache_key = id(node)
    cached = memo.get(cache_key)
    if cached is not None:
        return cached
    if depth > max_depth:
        summary = _SubtreeSummary()
        memo[cache_key] = summary
        return summary

    role = node.get("role", "")
    summary = _SubtreeSummary(
        control_nodes=1 if role in CHROME_ROLES else 0,
        container_nodes=1 if role in CONTAINER_ROLES else 0,
        content_list_nodes=1 if role in {"AXList", "AXOutline"} or "AXContentList" in _normalize_text(node.get("subrole", "")) else 0,
        nav_nodes=1 if _is_nav_like_node(node) else 0,
    )

    for text in _node_strings(node):
        if _TIMESTAMP_RE.search(text):
            summary = _SubtreeSummary(
                **{**summary.__dict__, "timestamp_nodes": summary.timestamp_nodes + 1}
            )
        if _contains_keyword(text, _ACTION_KEYWORDS):
            summary = _SubtreeSummary(
                **{**summary.__dict__, "action_nodes": summary.action_nodes + 1}
            )
        if role in CONTENT_ROLES:
            text_chars = summary.text_chars + len(text)
            text_nodes = summary.text_nodes + 1
            medium_nodes = summary.medium_text_nodes + (1 if len(text) >= _MIN_MESSAGE_TEXT_CHARS else 0)
            long_nodes = summary.long_text_nodes + (1 if len(text) >= _LONG_MESSAGE_TEXT_CHARS else 0)
            summary = _SubtreeSummary(
                text_chars=text_chars,
                text_nodes=text_nodes,
                medium_text_nodes=medium_nodes,
                long_text_nodes=long_nodes,
                timestamp_nodes=summary.timestamp_nodes,
                action_nodes=summary.action_nodes,
                nav_nodes=summary.nav_nodes,
                control_nodes=summary.control_nodes,
                container_nodes=summary.container_nodes,
                content_list_nodes=summary.content_list_nodes,
            )

    for child in node.get("children", []):
        child_summary = _summarize_subtree(child, memo, depth + 1, max_depth)
        summary = _SubtreeSummary(
            text_chars=summary.text_chars + child_summary.text_chars,
            text_nodes=summary.text_nodes + child_summary.text_nodes,
            medium_text_nodes=summary.medium_text_nodes + child_summary.medium_text_nodes,
            long_text_nodes=summary.long_text_nodes + child_summary.long_text_nodes,
            timestamp_nodes=summary.timestamp_nodes + child_summary.timestamp_nodes,
            action_nodes=summary.action_nodes + child_summary.action_nodes,
            nav_nodes=summary.nav_nodes + child_summary.nav_nodes,
            control_nodes=summary.control_nodes + child_summary.control_nodes,
            container_nodes=summary.container_nodes + child_summary.container_nodes,
            content_list_nodes=summary.content_list_nodes + child_summary.content_list_nodes,
        )

    memo[cache_key] = summary
    return summary


def _summary_to_debug(summary: _SubtreeSummary) -> dict[str, int]:
    return {
        "textChars": summary.text_chars,
        "textNodes": summary.text_nodes,
        "mediumTextNodes": summary.medium_text_nodes,
        "longTextNodes": summary.long_text_nodes,
        "timestampNodes": summary.timestamp_nodes,
        "actionNodes": summary.action_nodes,
        "navNodes": summary.nav_nodes,
        "controlNodes": summary.control_nodes,
        "containerNodes": summary.container_nodes,
        "contentListNodes": summary.content_list_nodes,
    }


def _preview_text(node: dict, max_items: int = 3, max_chars: int = 140) -> str:
    parts: list[str] = []

    def _walk(current: dict) -> None:
        if len(parts) >= max_items:
            return
        for text in _node_strings(current):
            if len(parts) >= max_items:
                break
            if len(text) >= 4:
                parts.append(text)
        for child in current.get("children", []):
            if len(parts) >= max_items:
                break
            _walk(child)

    _walk(node)
    preview = " | ".join(parts)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    return preview


def _looks_like_message_object(
    node: dict,
    summary: _SubtreeSummary,
    child_message_count: int = 0,
) -> bool:
    role = node.get("role", "")
    if role in CHROME_ROLES:
        return False
    if _is_nav_like_node(node):
        return False
    if summary.text_chars < _MIN_MESSAGE_TEXT_CHARS:
        return False
    if summary.long_text_nodes >= 1:
        return True
    if summary.medium_text_nodes >= 2:
        return True
    if summary.medium_text_nodes >= 1 and (summary.timestamp_nodes > 0 or summary.action_nodes > 0):
        return True
    if child_message_count >= 1 and summary.text_chars >= _LONG_MESSAGE_TEXT_CHARS:
        return True
    return False


def _is_aggregate_message_container(
    node: dict,
    child_message_count: int,
) -> bool:
    if child_message_count < 2:
        return False
    role = node.get("role", "")
    if role in {"AXWebArea", "AXScrollArea", "AXList", "AXOutline"}:
        return True
    return False


def _count_immediate_message_children(
    node: dict,
    memo: dict[int, _SubtreeSummary],
) -> int:
    count = 0
    for child in node.get("children", []):
        child_summary = _summarize_subtree(child, memo)
        if _looks_like_message_object(child, child_summary):
            count += 1
    return count


def _score_transcript_branch(
    node: dict,
    summary: _SubtreeSummary,
    memo: dict[int, _SubtreeSummary],
    ancestor_distance: int,
) -> tuple[float, bool]:
    immediate_message_children = _count_immediate_message_children(node, memo)
    control_density = summary.control_nodes / max(summary.text_nodes, 1)
    score = (
        summary.long_text_nodes * 7.0
        + summary.medium_text_nodes * 2.5
        + min(summary.text_chars / 120.0, 12.0)
        + summary.timestamp_nodes * 4.0
        + summary.action_nodes * 2.5
        + immediate_message_children * 6.0
        + summary.content_list_nodes * 3.0
        - summary.nav_nodes * 7.0
        - max(control_density - 2.0, 0.0) * 3.0
        - max(ancestor_distance - 1, 0) * 0.5
    )
    strong_signal = (
        immediate_message_children >= 2
        or summary.long_text_nodes >= 2
        or (
            summary.long_text_nodes >= 1
            and (summary.timestamp_nodes > 0 or summary.action_nodes > 0)
        )
    )
    qualifies = strong_signal and score >= 12.0
    return score, qualifies


def _collect_branch_candidates(
    path: list[dict],
    memo: dict[int, _SubtreeSummary],
) -> list[_BranchCandidate]:
    candidates: list[_BranchCandidate] = []
    for i in range(len(path) - 2, -1, -1):
        ancestor = path[i]
        ancestor_distance = len(path) - 1 - i
        for child in ancestor.get("children", []):
            if child.get("ancestorOfFocused") or child.get("focused"):
                continue
            summary = _summarize_subtree(child, memo)
            if summary.text_nodes == 0:
                continue
            score, qualifies = _score_transcript_branch(
                child, summary, memo, ancestor_distance,
            )
            candidates.append(
                _BranchCandidate(
                    node=child,
                    ancestor_distance=ancestor_distance,
                    score=score,
                    qualifies_as_transcript=qualifies,
                    summary=summary,
                    preview=_preview_text(child),
                )
            )
    candidates.sort(
        key=lambda item: (
            item.qualifies_as_transcript,
            item.score,
            -item.ancestor_distance,
            item.summary.text_chars,
        ),
        reverse=True,
    )
    return candidates


def _collect_message_objects(
    node: dict,
    memo: dict[int, _SubtreeSummary],
    *,
    is_root: bool = False,
) -> list[_MessageObject]:
    """Collect ordered message-like containers from a transcript branch."""
    child_objects: list[_MessageObject] = []
    for child in node.get("children", []):
        child_objects.extend(_collect_message_objects(child, memo))

    summary = _summarize_subtree(node, memo)
    if is_root and child_objects:
        return child_objects
    if not _looks_like_message_object(node, summary, child_message_count=len(child_objects)):
        return child_objects
    if _is_aggregate_message_container(node, len(child_objects)):
        return child_objects
    return [_MessageObject(node=node, summary=summary, preview=_preview_text(node))]


def _serialize_message_object(node: dict, remaining_chars: int) -> str:
    """Serialize one message object, clipping depth if it is too large."""
    target = max(180, min(_MAX_MESSAGE_OBJECT_CHARS, remaining_chars))
    best_within_budget = ""
    shortest_non_empty = ""
    for depth in (6, 4, 3, 2):
        xml = subtree_to_xml(node, max_depth=depth, indent=2)
        if not xml.strip():
            continue
        if not shortest_non_empty or len(xml) < len(shortest_non_empty):
            shortest_non_empty = xml
        if len(xml) <= target:
            return xml
        if len(xml) <= remaining_chars and (
            not best_within_budget or len(xml) < len(best_within_budget)
        ):
            best_within_budget = xml
    if best_within_budget:
        return best_within_budget
    compact_xml = _compact_message_object_xml(node, remaining_chars)
    if compact_xml:
        return compact_xml
    return shortest_non_empty if len(shortest_non_empty) <= int(remaining_chars * 1.15) else ""


def _compact_message_object_xml(node: dict, max_chars: int) -> str:
    """Build a valid compact XML snapshot from the best descendant text nodes."""
    if max_chars <= 0:
        return ""
    tag = (node.get("role") or "AXGroup").removeprefix("AX") or "Group"
    opening = f"    <{tag}>"
    closing = f"    </{tag}>"
    snippets: list[str] = []
    total = len(opening) + len(closing) + 2

    def _walk(current: dict) -> None:
        nonlocal total
        if total >= max_chars or len(snippets) >= 8:
            return
        role = current.get("role", "")
        if role in CHROME_ROLES:
            return
        snippet = ""
        if role in CONTENT_ROLES:
            text = _normalize_text(current.get("value")) or _normalize_text(current.get("title")) or _normalize_text(current.get("description"))
            if text:
                tag_name = role.removeprefix("AX")
                snippet = f"      <{tag_name}>{_esc(text[:220])}</{tag_name}>"
        if snippet and total + len(snippet) + 1 <= max_chars:
            snippets.append(snippet)
            total += len(snippet) + 1
        for child in current.get("children", []):
            if total >= max_chars or len(snippets) >= 8:
                break
            _walk(child)

    _walk(node)
    if not snippets:
        return ""
    return "\n".join([opening, *snippets, closing])


def _extract_transcript_context(
    transcript_branch: _BranchCandidate,
    focused_xml: str,
    char_budget: int,
    memo: dict[int, _SubtreeSummary],
) -> tuple[str, dict]:
    """Extract ordered recent message objects from the selected transcript branch."""
    message_objects = _collect_message_objects(
        transcript_branch.node,
        memo,
        is_root=True,
    )
    if not message_objects:
        message_objects = [
            _MessageObject(
                node=transcript_branch.node,
                summary=transcript_branch.summary,
                preview=transcript_branch.preview,
            )
        ]

    selected_xmls_rev: list[str] = []
    selected_indexes: list[int] = []
    total_chars = len(focused_xml)
    for index in range(len(message_objects) - 1, -1, -1):
        obj = message_objects[index]
        remaining = max(char_budget - total_chars, 0)
        if remaining <= 0:
            break
        xml = _serialize_message_object(obj.node, remaining)
        if not xml.strip():
            continue
        if total_chars + len(xml) > char_budget and selected_xmls_rev:
            continue
        selected_xmls_rev.append(xml)
        selected_indexes.append(index)
        total_chars += len(xml)

    selected_xmls = list(reversed(selected_xmls_rev))
    selected_index_set = set(selected_indexes)
    debug = {
        "strategy": "transcript_branch",
        "selectedBranch": {
            "ancestorDistance": transcript_branch.ancestor_distance,
            "score": round(transcript_branch.score, 2),
            "preview": transcript_branch.preview,
            "summary": _summary_to_debug(transcript_branch.summary),
        },
        "branchCandidates": [],
        "messageObjects": [
            {
                "index": index,
                "selected": index in selected_index_set,
                "preview": obj.preview,
                "summary": _summary_to_debug(obj.summary),
            }
            for index, obj in enumerate(message_objects)
        ],
    }

    lines = ["<context>"]
    lines.extend(selected_xmls)
    lines.append("  <input>")
    lines.append(focused_xml)
    lines.append("  </input>")
    lines.append("</context>")
    return "\n".join(lines), debug


def _extract_fallback_context(
    path: list[dict],
    focused_xml: str,
    char_budget: int,
) -> tuple[str, dict]:
    """Preserve the previous sibling-accumulation behavior as a fallback."""
    total_chars = len(focused_xml)
    context_parts: list[str] = []

    for i in range(len(path) - 2, -1, -1):
        ancestor = path[i]
        ancestor_role = ancestor.get("role", "")

        candidates: list[tuple[int, dict]] = []
        for child in ancestor.get("children", []):
            if child.get("ancestorOfFocused") or child.get("focused"):
                continue

            chars = text_char_count(child)
            nodes = count_text_nodes(child)
            if nodes == 0:
                continue

            avg = chars / max(nodes, 1)
            if nodes > _MIN_TEXT_NODES_FOR_NAV_CHECK and avg < _MIN_AVG_CHARS_CONTENT:
                continue

            candidates.append((chars, child))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _chars, child in candidates:
            xml = subtree_to_xml(child, max_depth=8, indent=2)
            if not xml.strip():
                continue

            xml_len = len(xml)
            if total_chars + xml_len > char_budget:
                xml = subtree_to_xml(child, max_depth=4, indent=2)
                xml_len = len(xml)
                if total_chars + xml_len > char_budget:
                    continue

            context_parts.append(xml)
            total_chars += xml_len

        if total_chars > char_budget:
            break

        if ancestor_role in BOUNDARY_ROLES and context_parts:
            break

    context_parts.reverse()
    lines = ["<context>"]
    lines.extend(context_parts)
    lines.append("  <input>")
    lines.append(focused_xml)
    lines.append("  </input>")
    lines.append("</context>")
    return "\n".join(lines), {"strategy": "fallback_siblings"}


# ---------------------------------------------------------------------------
# XML serialization
# ---------------------------------------------------------------------------

def subtree_to_xml(
    node: dict,
    max_depth: int = 8,
    depth: int = 0,
    indent: int = 0,
) -> str:
    """Convert a subtree dict to compact XML, collapsing empty wrappers.

    Chrome roles are skipped entirely.  Container nodes with no own content
    and a single child are collapsed (the child is emitted in their place).
    Content-role nodes emit their text as the element body.
    """
    role = node.get("role", "")
    val = node.get("value")
    title = node.get("title", "")
    children = node.get("children", [])

    if role in CHROME_ROLES:
        return ""

    has_value = isinstance(val, str) and val.strip()
    has_title = isinstance(title, str) and title.strip()
    has_own_content = has_value or has_title
    is_focused = node.get("focused", False)

    # Collapse single-child containers with no own content.
    if role in CONTAINER_ROLES and not has_own_content and len(children) == 1 and not is_focused:
        return subtree_to_xml(children[0], max_depth, depth + 1, indent)

    if depth >= max_depth:
        return ""

    # Recurse into children.
    child_xmls: list[str] = []
    for child in children:
        cx = subtree_to_xml(child, max_depth, depth + 1, indent + 1)
        if cx.strip():
            child_xmls.append(cx)

    if not has_own_content and not child_xmls and not is_focused:
        return ""

    prefix = "  " * indent
    tag = role.removeprefix("AX")  # shorter tags: StaticText, Group, …

    # Content roles → inline text element.
    if role in CONTENT_ROLES:
        text = val.strip()[:300] if has_value else ""
        attrs: list[str] = []
        if has_title and title.strip() != text:
            attrs.append(f'title="{_esc(title.strip()[:100])}"')
        # AXDescription often carries speaker/timestamp metadata in chat apps
        # (e.g. "Your iMessage, Hello, 3:15 PM" or "Daniel, Hi there, 3:04 PM").
        desc = node.get("description", "")
        if isinstance(desc, str) and desc.strip() and desc.strip() != text:
            attrs.append(f'desc="{_esc(desc.strip()[:200])}"')
        attrs.extend(_focus_attrs(node))
        attr_str = (" " + " ".join(attrs)) if attrs else ""
        return f"{prefix}<{tag}{attr_str}>{_esc(text)}</{tag}>"

    # Container roles.
    attrs = []
    if has_title:
        attrs.append(f'title="{_esc(title.strip()[:100])}"')
    desc = node.get("description", "")
    if isinstance(desc, str) and desc.strip():
        attrs.append(f'desc="{_esc(desc.strip()[:200])}"')
    attrs.extend(_focus_attrs(node))
    attr_str = (" " + " ".join(attrs)) if attrs else ""

    if not child_xmls:
        if is_focused:
            return f"{prefix}<{tag}{attr_str}></{tag}>"
        return ""

    inner = "\n".join(child_xmls)
    return f"{prefix}<{tag}{attr_str}>\n{inner}\n{prefix}</{tag}>"


def _esc(text: str) -> str:
    """Minimal XML-safe escaping for attribute values and text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _focus_attrs(node: dict) -> list[str]:
    """Serialize focus/caret metadata into compact XML attrs."""
    attrs: list[str] = []
    if node.get("focused"):
        attrs.append('focused="true"')
    if node.get("placeholderDetected"):
        attrs.append('placeholder_detected="true"')
    cursor = node.get("cursorPosition")
    if isinstance(cursor, (int, float)):
        attrs.append(f'cursor="{int(cursor)}"')
    selection = node.get("selectionLength")
    if isinstance(selection, (int, float)):
        attrs.append(f'selection="{int(selection)}"')
    value_length = node.get("valueLength", node.get("numberOfCharacters"))
    if isinstance(value_length, (int, float)):
        attrs.append(f'value_length="{int(value_length)}"')
    return attrs


def extract_focus_path_overview(
    tree: dict,
    token_budget: int = 120,
    max_siblings_per_level: int = 2,
) -> Optional[str]:
    """Build a compact top-down overview centered on the focus path."""
    path = find_focused_path(tree)
    if not path:
        return None

    char_budget = token_budget * 4
    lines = ["<focusPath>"]
    total_chars = len(lines[0]) + 1

    for depth, node in enumerate(path):
        tag = (node.get("role") or "AXUnknown").removeprefix("AX")
        attrs: list[str] = [f'depth="{depth}"']
        title = node.get("title", "")
        desc = node.get("description", "")
        if isinstance(title, str) and title.strip():
            attrs.append(f'title="{_esc(title.strip()[:100])}"')
        if isinstance(desc, str) and desc.strip():
            attrs.append(f'desc="{_esc(desc.strip()[:160])}"')
        attrs.extend(_focus_attrs(node))
        attr_str = " " + " ".join(attrs)

        value = node.get("value")
        if depth == len(path) - 1 and isinstance(value, str):
            node_line = f"  <{tag}{attr_str}>{_esc(value.strip()[:160])}</{tag}>"
        else:
            node_line = f"  <{tag}{attr_str}/>"

        if total_chars + len(node_line) > char_budget:
            break
        lines.append(node_line)
        total_chars += len(node_line) + 1

        if depth == len(path) - 1:
            continue

        next_node = path[depth + 1]
        sibling_xmls: list[str] = []
        for child in node.get("children", []):
            if child is next_node or child.get("ancestorOfFocused") or child.get("focused"):
                continue
            chars = text_char_count(child, max_depth=6)
            nodes = count_text_nodes(child, max_depth=6)
            if chars <= 0 or nodes <= 0:
                continue
            avg = chars / max(nodes, 1)
            if nodes > _MIN_TEXT_NODES_FOR_NAV_CHECK and avg < _MIN_AVG_CHARS_CONTENT:
                continue
            sibling_xml = subtree_to_xml(child, max_depth=2, indent=2)
            if sibling_xml.strip():
                sibling_xmls.append((chars, sibling_xml))

        for _, sibling_xml in sorted(sibling_xmls, key=lambda item: item[0], reverse=True)[:max_siblings_per_level]:
            if total_chars + len(sibling_xml) > char_budget:
                break
            lines.append(sibling_xml)
            total_chars += len(sibling_xml) + 1

    lines.append("</focusPath>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main context walker
# ---------------------------------------------------------------------------

def extract_context_from_tree(
    tree: dict,
    token_budget: int = 500,
) -> Optional[str]:
    """Walk from the focused element upward, collecting nearby content as XML.

    Parameters
    ----------
    tree:
        A serialized AX tree dict (as produced by ``serialize_ax_tree`` or
        loaded from a JSON fixture).  Must contain ``focused`` /
        ``ancestorOfFocused`` annotations.
    token_budget:
        Approximate maximum output size in tokens (1 token ≈ 4 chars).

    Returns
    -------
    str or None
        Minimal XML context string, or ``None`` if no focused element was
        found.
    """
    path = find_focused_path(tree)
    if not path:
        return None

    context, _ = _extract_context_from_tree_with_debug(path, token_budget=token_budget)
    return context


def _extract_context_from_tree_with_debug(
    path: list[dict],
    token_budget: int = 500,
) -> tuple[Optional[str], Optional[dict]]:
    """Internal helper that returns both the context XML and selection debug."""
    if not path:
        return None, None

    focused = path[-1]
    char_budget = token_budget * 4
    focused_xml = subtree_to_xml(focused, max_depth=2, indent=2)
    memo: dict[int, _SubtreeSummary] = {}

    branch_candidates = _collect_branch_candidates(path, memo)
    transcript_candidates = [
        candidate for candidate in branch_candidates
        if candidate.qualifies_as_transcript
    ]

    if transcript_candidates:
        selected = transcript_candidates[0]
        context, debug = _extract_transcript_context(
            selected,
            focused_xml=focused_xml,
            char_budget=char_budget,
            memo=memo,
        )
        debug["branchCandidates"] = [
            {
                "score": round(candidate.score, 2),
                "qualifiesAsTranscript": candidate.qualifies_as_transcript,
                "ancestorDistance": candidate.ancestor_distance,
                "preview": candidate.preview,
                "summary": _summary_to_debug(candidate.summary),
            }
            for candidate in branch_candidates[:_MAX_BRANCH_DEBUG_CANDIDATES]
        ]
        return context, debug

    context, debug = _extract_fallback_context(
        path,
        focused_xml=focused_xml,
        char_budget=char_budget,
    )
    debug["branchCandidates"] = [
        {
            "score": round(candidate.score, 2),
            "qualifiesAsTranscript": candidate.qualifies_as_transcript,
            "ancestorDistance": candidate.ancestor_distance,
            "preview": candidate.preview,
            "summary": _summary_to_debug(candidate.summary),
        }
        for candidate in branch_candidates[:_MAX_BRANCH_DEBUG_CANDIDATES]
    ]
    return context, debug


def extract_context_live(
    window_element,
    focused_element,
    max_depth: int = 40,
    token_budget: int = 500,
) -> Optional[str]:
    """Convenience wrapper for live AX elements.

    Serializes the window's AX tree with focus annotations, then runs the
    subtree walker.  Use this at runtime in ``app.py``.
    """
    from autocompleter.ax_utils import serialize_ax_tree

    tree = serialize_ax_tree(
        window_element,
        max_depth=max_depth,
        focused_element=focused_element,
    )
    if tree is None:
        return None
    return extract_context_from_tree(tree, token_budget=token_budget)


def build_context_bundle_from_tree(
    tree: dict,
    token_budget: int = 500,
    overview_token_budget: int = 120,
) -> TreeContextBundle | None:
    """Build the prompt-ready context bundle from a serialized tree."""
    if not tree:
        return None
    path = find_focused_path(tree)
    bottom_up_context, selection_debug = _extract_context_from_tree_with_debug(
        path or [],
        token_budget=token_budget,
    )
    return TreeContextBundle(
        tree=tree,
        bottom_up_context=bottom_up_context,
        top_down_context=extract_focus_path_overview(
            tree, token_budget=overview_token_budget,
        ),
        selection_debug=selection_debug,
    )


def build_context_bundle_live(
    window_element,
    focused_element,
    focused_value: str | None = None,
    placeholder_detected: bool = False,
    insertion_point: int | None = None,
    selection_length: int = 0,
    raw_value: str = "",
    raw_placeholder_value: str = "",
    raw_number_of_characters: int | None = None,
    max_depth: int = 40,
    token_budget: int = 500,
    overview_token_budget: int = 120,
) -> TreeContextBundle | None:
    """Serialize a focus-aware tree and derive prompt/debug context blocks."""
    from autocompleter.ax_utils import serialize_ax_tree

    focused_metadata = {
        "placeholderDetected": placeholder_detected,
        "cursorPosition": insertion_point,
        "selectionLength": selection_length,
        "rawValue": raw_value,
        "rawValueLength": len(raw_value or ""),
        "rawPlaceholderValue": raw_placeholder_value,
        "rawNumberOfCharacters": raw_number_of_characters,
    }
    tree = serialize_ax_tree(
        window_element,
        max_depth=max_depth,
        focused_element=focused_element,
        focused_value_override=focused_value,
        focused_metadata=focused_metadata,
    )
    if tree is None:
        return None
    return build_context_bundle_from_tree(
        tree,
        token_budget=token_budget,
        overview_token_budget=overview_token_budget,
    )
