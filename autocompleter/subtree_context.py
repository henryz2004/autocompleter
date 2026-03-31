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
    if role in CONTAINER_ROLES and not has_own_content and len(children) == 1:
        return subtree_to_xml(children[0], max_depth, depth + 1, indent)

    if depth >= max_depth:
        return ""

    # Recurse into children.
    child_xmls: list[str] = []
    for child in children:
        cx = subtree_to_xml(child, max_depth, depth + 1, indent + 1)
        if cx.strip():
            child_xmls.append(cx)

    if not has_own_content and not child_xmls:
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
        if is_focused:
            attrs.append('focused="true"')
        attr_str = (" " + " ".join(attrs)) if attrs else ""
        return f"{prefix}<{tag}{attr_str}>{_esc(text)}</{tag}>"

    # Container roles.
    attrs = []
    if has_title:
        attrs.append(f'title="{_esc(title.strip()[:100])}"')
    desc = node.get("description", "")
    if isinstance(desc, str) and desc.strip():
        attrs.append(f'desc="{_esc(desc.strip()[:200])}"')
    attr_str = (" " + " ".join(attrs)) if attrs else ""

    if not child_xmls:
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

    focused = path[-1]
    char_budget = token_budget * 4

    focused_xml = subtree_to_xml(focused, max_depth=2, indent=2)
    total_chars = len(focused_xml)

    context_parts: list[str] = []

    # Walk upward through ancestors, nearest first.
    for i in range(len(path) - 2, -1, -1):
        ancestor = path[i]
        ancestor_role = ancestor.get("role", "")

        # Score each sibling by text richness.
        candidates: list[tuple[int, dict]] = []
        for child in ancestor.get("children", []):
            if child.get("ancestorOfFocused") or child.get("focused"):
                continue

            chars = text_char_count(child)
            nodes = count_text_nodes(child)
            if nodes == 0:
                continue

            # Heuristic: skip navigation-like clusters (many tiny labels).
            avg = chars / max(nodes, 1)
            if nodes > _MIN_TEXT_NODES_FOR_NAV_CHECK and avg < _MIN_AVG_CHARS_CONTENT:
                continue

            candidates.append((chars, child))

        # Largest content-rich subtrees first.
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _chars, child in candidates:
            xml = subtree_to_xml(child, max_depth=8, indent=2)
            if not xml.strip():
                continue

            xml_len = len(xml)
            if total_chars + xml_len > char_budget:
                # Retry with shallower depth.
                xml = subtree_to_xml(child, max_depth=4, indent=2)
                xml_len = len(xml)
                if total_chars + xml_len > char_budget:
                    continue

            context_parts.append(xml)
            total_chars += xml_len

        if total_chars > char_budget:
            break

        # Stop at boundary when we already have content.
        if ancestor_role in BOUNDARY_ROLES and context_parts:
            break

    # Outermost context first (reverse the nearest-first collection order).
    context_parts.reverse()

    lines = ["<context>"]
    lines.extend(context_parts)
    lines.append("  <input>")
    lines.append(focused_xml)
    lines.append("  </input>")
    lines.append("</context>")
    return "\n".join(lines)


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
