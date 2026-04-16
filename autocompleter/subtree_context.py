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
_MAX_TRANSCRIPT_MESSAGE_OBJECTS = 6
_MAX_BRANCH_DEBUG_CANDIDATES = 8
_TINY_OVERVIEW_TOKEN_BUDGET = 60
_TINY_OVERVIEW_MAX_SIBLINGS = 0

_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b")
_SPEAKER_HEADING_RE = re.compile(r"\bsaid\b", re.IGNORECASE)
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
_STATUS_NOISE_TEXT = frozenset({
    "copy",
    "copy message",
    "undo",
    "commit",
    "local",
    "main",
    "full access",
    "high",
})


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


def _preferred_node_text(node: dict) -> str:
    for key in ("value", "title", "description"):
        text = _normalize_text(node.get(key))
        if text:
            return text
    return ""


def _is_timestamp_text(text: str) -> bool:
    return bool(_TIMESTAMP_RE.fullmatch(text))


def _looks_like_action_label(text: str) -> bool:
    lowered = text.lower().strip()
    if lowered in _STATUS_NOISE_TEXT:
        return True
    return lowered in _ACTION_KEYWORDS


def _is_action_text(text: str) -> bool:
    return _looks_like_action_label(text)


def _is_status_text(text: str) -> bool:
    return text.lower() in _STATUS_NOISE_TEXT


def _is_transcript_noise_text(text: str) -> bool:
    if not text:
        return True
    if _is_timestamp_text(text):
        return True
    if _is_action_text(text):
        return True
    if _is_status_text(text) and len(text) <= 24:
        return True
    return False


def _join_text_fragments(fragments: list[str]) -> str:
    if not fragments:
        return ""

    merged = fragments[0]
    for fragment in fragments[1:]:
        if not fragment:
            continue
        if fragment == merged or fragment.lower() == merged.lower():
            continue
        if fragment.startswith((",", ".", ":", ";", "!", "?", ")", "]", "}")):
            merged += fragment
            continue
        if merged.endswith(("(", "[", "{", "“", '"', "'")):
            merged += fragment
            continue
        merged += " " + fragment
    return merged.strip()


def _trim_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker_overhead = len("[trimmed middle  chars]")
    if max_chars <= marker_overhead + 20:
        return text[:max_chars]

    available = max_chars - marker_overhead
    head = max(int(available * 0.6), 24)
    tail = max(available - head, 18)
    if head + tail >= len(text):
        head = max_chars // 2
        tail = max_chars - head
    trimmed_chars = max(len(text) - head - tail, 0)
    marker = f"[trimmed middle {trimmed_chars} chars]"
    available = max_chars - len(marker) - 2
    head = max(int(available * 0.6), 24)
    tail = max(available - head, 18)
    if head + tail >= len(text):
        head = max_chars // 2
        tail = max_chars - head
    return f"{text[:head].rstrip()} {marker} {text[-tail:].lstrip()}".strip()


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def _is_nav_like_node(node: dict) -> bool:
    role = node.get("role", "")
    subrole = _normalize_text(node.get("subrole", ""))
    role_desc = _normalize_text(node.get("roleDescription", ""))
    if role in CONTENT_ROLES:
        preferred = _preferred_node_text(node)
        if len(preferred) >= _MIN_MESSAGE_TEXT_CHARS:
            return False
    if "AXLandmarkNavigation" in subrole or "AXLandmarkComplementary" in subrole:
        return True
    short_node_strings = [text for text in _node_strings(node) if len(text) <= 40]
    combined = " ".join([role, subrole, role_desc, *short_node_strings]).lower()
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
        if _looks_like_action_label(text):
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
    if summary.text_chars < 8:
        return False
    if summary.long_text_nodes >= 1:
        return True
    if summary.medium_text_nodes >= 2:
        return True
    if summary.medium_text_nodes >= 1 and (summary.timestamp_nodes > 0 or summary.action_nodes > 0):
        return True
    if summary.text_chars >= _MIN_MESSAGE_TEXT_CHARS and summary.text_nodes >= 2 and summary.nav_nodes == 0:
        return True
    if summary.text_chars >= 8 and (summary.timestamp_nodes > 0 or summary.action_nodes > 0):
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
    if role == "AXGroup" and not _node_strings(node):
        return True
    return False


def _has_message_heading_cue(node: dict, max_depth: int = 3) -> bool:
    def _walk(current: dict, depth: int) -> bool:
        if depth > max_depth:
            return False
        role = current.get("role", "")
        if role == "AXHeading":
            for text in _node_strings(current):
                if _SPEAKER_HEADING_RE.search(text):
                    return True
        for child in current.get("children", []):
            if _walk(child, depth + 1):
                return True
        return False

    return _walk(node, 0)


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


def _make_synthetic_message_object(
    children: list[dict],
    memo: dict[int, _SubtreeSummary],
) -> _MessageObject | None:
    synthetic = {
        "role": "AXGroup",
        "title": "",
        "description": "",
        "value": "",
        "children": children,
    }
    summary = _summarize_subtree(synthetic, memo)
    if summary.text_chars < 8:
        return None
    return _MessageObject(
        node=synthetic,
        summary=summary,
        preview=_preview_text(synthetic),
    )


def _segment_flat_message_stream(
    node: dict,
    memo: dict[int, _SubtreeSummary],
) -> list[_MessageObject]:
    """Segment flattened transcript children into ordered message objects.

    Some apps expose a transcript as a flat sequence of direct children:
    prose blocks, then timestamp/action controls, then the next prose block.
    When that pattern is present, convert it into synthetic grouped messages
    before falling back to container-level message detection.
    """
    children = node.get("children", [])
    if len(children) < 4:
        return []

    strong_text_children = 0
    boundary_children = 0
    for child in children:
        summary = _summarize_subtree(child, memo)
        if summary.long_text_nodes >= 1 or summary.medium_text_nodes >= 1:
            strong_text_children += 1
        if summary.timestamp_nodes > 0 or summary.action_nodes > 0:
            boundary_children += 1
    if strong_text_children < 2 or boundary_children < 1:
        return []

    segments: list[list[dict]] = []
    current: list[dict] = []
    current_has_text = False
    saw_boundary_after_text = False

    for child in children:
        summary = _summarize_subtree(child, memo)
        has_text = summary.long_text_nodes >= 1 or summary.medium_text_nodes >= 1
        has_boundary = summary.timestamp_nodes > 0 or summary.action_nodes > 0

        if has_text and current_has_text and saw_boundary_after_text:
            segments.append(current)
            current = [child]
            current_has_text = True
            saw_boundary_after_text = False
            continue

        if has_text or has_boundary or current:
            current.append(child)

        if has_text:
            current_has_text = True
        if current_has_text and has_boundary:
            saw_boundary_after_text = True

    if current:
        segments.append(current)

    message_objects: list[_MessageObject] = []
    for segment in segments:
        obj = _make_synthetic_message_object(segment, memo)
        if obj is not None:
            message_objects.append(obj)

    return message_objects


def _collect_message_fragments(
    node: dict,
    *,
    depth: int = 0,
    max_depth: int = 12,
) -> list[str]:
    if depth > max_depth:
        return []

    role = node.get("role", "")
    if role in CHROME_ROLES or _is_nav_like_node(node):
        return []

    fragments: list[str] = []
    if role in CONTENT_ROLES:
        text = _preferred_node_text(node)
        if text and not _is_transcript_noise_text(text):
            fragments.append(text)

    for child in node.get("children", []):
        child_fragments = _collect_message_fragments(
            child,
            depth=depth + 1,
            max_depth=max_depth,
        )
        for fragment in child_fragments:
            if fragments and (
                fragment == fragments[-1]
                or fragment.lower() == fragments[-1].lower()
            ):
                continue
            fragments.append(fragment)
    return fragments


def _serialize_clean_message_object(
    node: dict,
    remaining_chars: int,
) -> str:
    if remaining_chars <= 0:
        return ""

    wrapper_overhead = len("    <Message></Message>")
    text_budget = max(
        80,
        min(_MAX_MESSAGE_OBJECT_CHARS, remaining_chars) - wrapper_overhead,
    )
    merged_text = _join_text_fragments(_collect_message_fragments(node))
    if not merged_text:
        return ""
    merged_text = _trim_middle(merged_text, text_budget)
    return f"    <Message>{_esc(merged_text)}</Message>"


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
        or (summary.text_nodes >= 4 and summary.timestamp_nodes >= 2)
        or (summary.text_nodes >= 3 and summary.action_nodes >= 1)
        or summary.long_text_nodes >= 2
        or (summary.text_chars >= _LONG_MESSAGE_TEXT_CHARS and summary.text_nodes >= 2)
        or (
            summary.long_text_nodes >= 1
            and (summary.timestamp_nodes > 0 or summary.action_nodes > 0)
        )
        or (
            summary.medium_text_nodes >= 3
            and (summary.timestamp_nodes > 0 or summary.action_nodes > 0)
        )
        or (summary.text_chars >= _LONG_MESSAGE_TEXT_CHARS and summary.nav_nodes == 0)
    )
    qualifies = strong_signal and score >= 11.0
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
    segmented_objects = _segment_flat_message_stream(node, memo)
    if segmented_objects:
        return segmented_objects
    if child_objects and len(node.get("children", [])) == 1 and not _node_strings(node):
        return child_objects
    if is_root and child_objects:
        return child_objects
    if not _looks_like_message_object(node, summary, child_message_count=len(child_objects)):
        return child_objects
    if child_objects and _has_message_heading_cue(node):
        return [_MessageObject(node=node, summary=summary, preview=_preview_text(node))]
    if _is_aggregate_message_container(node, len(child_objects)):
        return child_objects
    return [_MessageObject(node=node, summary=summary, preview=_preview_text(node))]


def _serialize_message_object(node: dict, remaining_chars: int) -> str:
    """Serialize one message object as a compact cleaned message block."""
    return _serialize_clean_message_object(node, remaining_chars)


def _compact_message_object_xml(node: dict, max_chars: int) -> str:
    """Build a valid compact XML snapshot from the best descendant text nodes."""
    if max_chars <= 0:
        return ""
    tag = (node.get("role") or "AXGroup").removeprefix("AX") or "Group"
    opening = f"    <{tag}>"
    closing = f"    </{tag}>"
    total = len(opening) + len(closing) + 2
    snippet_candidates: list[tuple[int, int, str]] = []

    def _walk(current: dict, depth: int = 0) -> None:
        if len(snippet_candidates) >= 24:
            return
        role = current.get("role", "")
        if role in CHROME_ROLES:
            return
        if role in CONTENT_ROLES:
            text = _normalize_text(current.get("value")) or _normalize_text(current.get("title")) or _normalize_text(current.get("description"))
            if text:
                tag_name = role.removeprefix("AX")
                snippet = f"      <{tag_name}>{_esc(text[:220])}</{tag_name}>"
                is_timestamp = bool(_TIMESTAMP_RE.fullmatch(text))
                score = len(text)
                if is_timestamp:
                    score -= 120
                if len(text) >= _MIN_MESSAGE_TEXT_CHARS:
                    score += 80
                if len(text) >= _LONG_MESSAGE_TEXT_CHARS:
                    score += 80
                snippet_candidates.append((score, depth, snippet))
        for child in current.get("children", []):
            if len(snippet_candidates) >= 24:
                break
            _walk(child, depth + 1)

    _walk(node)
    if not snippet_candidates:
        return ""
    snippet_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    snippets: list[str] = []
    saw_substantive_text = False
    for score, _depth, snippet in snippet_candidates:
        is_timestamp = ">" in snippet and bool(_TIMESTAMP_RE.fullmatch(snippet.split(">", 1)[1].rsplit("<", 1)[0]))
        if is_timestamp and not saw_substantive_text:
            continue
        if total + len(snippet) + 1 > max_chars:
            continue
        snippets.append(snippet)
        total += len(snippet) + 1
        if not is_timestamp:
            saw_substantive_text = True
        if len(snippets) >= 8:
            break
    if not snippets:
        # If all we had was timestamp-level metadata, return nothing rather than
        # polluting the prompt with a bare clock value.
        return ""
    return "\n".join([opening, *snippets, closing])


def _serialized_xml_has_meaningful_prose(xml: str) -> bool:
    """Return True when serialized XML includes at least one prose-bearing line."""
    non_timestamp_lines: list[str] = []
    for raw_line in xml.splitlines():
        text = re.sub(r"<[^>]+>", " ", raw_line)
        text = _normalize_text(text)
        if not text:
            continue
        if _TIMESTAMP_RE.fullmatch(text):
            continue
        non_timestamp_lines.append(text)
        if len(text) >= _MIN_MESSAGE_TEXT_CHARS:
            return True
        if len(text) >= 4:
            return True
    if len(non_timestamp_lines) >= 2:
        combined = " ".join(non_timestamp_lines)
        if len(combined) >= 12:
            return True
    return False


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

    recent_offset = max(len(message_objects) - _MAX_TRANSCRIPT_MESSAGE_OBJECTS, 0)
    recent_message_objects = message_objects[recent_offset:]
    selected_xmls_rev: list[str] = []
    selected_indexes: list[int] = []
    total_chars = len(focused_xml)
    for local_index in range(len(recent_message_objects) - 1, -1, -1):
        obj = recent_message_objects[local_index]
        index = recent_offset + local_index
        remaining = max(char_budget - total_chars, 0)
        if remaining <= 0:
            break
        xml = _serialize_message_object(obj.node, remaining)
        if not xml.strip():
            continue
        if not _serialized_xml_has_meaningful_prose(xml):
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
    desc = node.get("description", "")
    has_desc = isinstance(desc, str) and desc.strip()
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
        raw_text = ""
        if has_value:
            raw_text = val.strip()
        elif has_title:
            raw_text = title.strip()
        elif has_desc:
            raw_text = desc.strip()
        text = raw_text[:300]
        attrs: list[str] = []
        if has_title and title.strip() != text:
            attrs.append(f'title="{_esc(title.strip()[:100])}"')
        # AXDescription often carries speaker/timestamp metadata in chat apps
        # (e.g. "Your iMessage, Hello, 3:15 PM" or "Daniel, Hi there, 3:04 PM").
        if has_desc and desc.strip() != text:
            attrs.append(f'desc="{_esc(desc.strip()[:200])}"')
        attrs.extend(_focus_attrs(node))
        attr_str = (" " + " ".join(attrs)) if attrs else ""
        return f"{prefix}<{tag}{attr_str}>{_esc(text)}</{tag}>"

    # Container roles.
    attrs = []
    if has_title:
        attrs.append(f'title="{_esc(title.strip()[:100])}"')
    if has_desc:
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
    token_budget: int = 1000,
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
    token_budget: int = 1000,
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
    token_budget: int = 1000,
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
    token_budget: int = 1000,
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
    is_transcript_context = (
        selection_debug is not None
        and selection_debug.get("strategy") == "transcript_branch"
    )
    effective_overview_budget = overview_token_budget
    max_siblings_per_level = 2
    if is_transcript_context:
        effective_overview_budget = min(
            overview_token_budget,
            _TINY_OVERVIEW_TOKEN_BUDGET,
        )
        max_siblings_per_level = _TINY_OVERVIEW_MAX_SIBLINGS
    return TreeContextBundle(
        tree=tree,
        bottom_up_context=bottom_up_context,
        top_down_context=extract_focus_path_overview(
            tree,
            token_budget=effective_overview_budget,
            max_siblings_per_level=max_siblings_per_level,
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
    token_budget: int = 1000,
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
