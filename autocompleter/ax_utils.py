"""Shared Accessibility API helper functions.

Provides a unified interface for common AX operations used across
input_observer.py and text_injector.py.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import ApplicationServices
    from ApplicationServices import (
        AXUIElementGetPid,
        AXValueGetValue,
        kAXValueTypeCFRange,
        kAXValueTypeCGPoint,
        kAXValueTypeCGSize,
    )

    HAS_ACCESSIBILITY = True
except ImportError:
    HAS_ACCESSIBILITY = False


def ax_get_attribute(element, attribute: str):
    """Safely get an accessibility attribute from an element."""
    if not HAS_ACCESSIBILITY:
        return None
    err, value = ApplicationServices.AXUIElementCopyAttributeValue(
        element, attribute, None
    )
    if err == 0:
        return value
    return None


# Sentinel for missing attributes in batch results
_AX_MISSING = object()

# Common attribute sets used by tree walks
_TREE_ATTRS = [
    "AXRole", "AXSubrole", "AXRoleDescription", "AXValue",
    "AXTitle", "AXDescription", "AXPlaceholderValue",
    "AXNumberOfCharacters", "AXVisibleChildren",
]
# Fallback when AXVisibleChildren is not supported
_TREE_ATTRS_FALLBACK = [
    "AXRole", "AXSubrole", "AXRoleDescription", "AXValue",
    "AXTitle", "AXDescription", "AXPlaceholderValue",
    "AXNumberOfCharacters", "AXChildren",
]


def ax_get_multiple_attributes(
    element, attributes: list[str]
) -> dict[str, object]:
    """Batch-fetch multiple attributes in a single IPC call.

    Returns a dict mapping attribute names to values.  Attributes that
    are not supported or have errors map to ``None``.

    Uses ``AXUIElementCopyMultipleAttributeValues`` which makes one
    cross-process round-trip instead of N, dramatically reducing latency
    for tree walks (9 calls per node → 1).
    """
    if not HAS_ACCESSIBILITY:
        return {a: None for a in attributes}
    try:
        err, values = ApplicationServices.AXUIElementCopyMultipleAttributeValues(
            element, attributes, 0, None
        )
        if err == 0 and values is not None:
            result = {}
            for i, attr in enumerate(attributes):
                if i < len(values):
                    v = values[i]
                    # pyobjc wraps missing/error values as NSNull or similar
                    if v is None or type(v).__name__ == "NSNull":
                        result[attr] = None
                    else:
                        result[attr] = v
                else:
                    result[attr] = None
            return result
    except Exception:
        logger.debug("Batch attribute fetch failed, falling back to individual calls", exc_info=True)
    # Fallback: fetch individually (shouldn't normally happen)
    return {a: ax_get_attribute(element, a) for a in attributes}


def ax_get_tree_attrs(element) -> dict[str, object]:
    """Fetch the standard set of tree-walk attributes in one IPC call.

    Prefers ``AXVisibleChildren`` (only on-screen elements) and falls
    back to ``AXChildren`` if the element doesn't support it.

    Returns a dict with keys: AXRole, AXSubrole, AXRoleDescription,
    AXValue, AXTitle, AXDescription, AXPlaceholderValue,
    AXNumberOfCharacters, and either AXVisibleChildren or AXChildren
    (stored under the key ``"children"`` for convenience).
    """
    attrs = ax_get_multiple_attributes(element, _TREE_ATTRS)
    children = attrs.get("AXVisibleChildren")
    # AXVisibleChildren may return an AXValueRef or other non-list type
    # on some elements — only trust it if it's a usable sequence.
    if not _is_ax_list(children) or len(children) == 0:
        # AXVisibleChildren not supported or empty — try AXChildren
        fallback_children = ax_get_attribute(element, "AXChildren")
        if _is_ax_list(fallback_children) and len(fallback_children) > 0:
            children = list(fallback_children)
        else:
            children = []
    else:
        children = list(children)
    attrs["children"] = children
    return attrs


def _is_ax_list(obj) -> bool:
    """Check if an AX return value is a usable list of elements.

    pyobjc may return ``NSArray`` (or ``NSCFArray``) which is iterable
    and supports ``len()`` but fails ``isinstance(obj, list)``.  We
    accept anything that looks like a sequence of AX elements while
    rejecting scalar wrappers like ``AXValueRef``.
    """
    if obj is None:
        return False
    if isinstance(obj, (list, tuple)):
        return True
    # pyobjc NSArray: has __len__ and __getitem__ but is not list/tuple.
    # AXValueRef does NOT have __len__, so this distinguishes them.
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and not isinstance(obj, (str, bytes))


def ax_get_children(element) -> list:
    """Get children of an AX element, preferring visible-only children.

    Tries ``AXVisibleChildren`` first (returns only on-screen elements,
    dramatically reducing node count for scrollable containers).  Falls
    back to ``AXChildren`` if the element doesn't support visible children.
    """
    children = ax_get_attribute(element, "AXVisibleChildren")
    if _is_ax_list(children) and len(children) > 0:
        return list(children)
    children = ax_get_attribute(element, "AXChildren")
    if _is_ax_list(children):
        return list(children)
    return []


def ax_get_visible_text(element) -> str | None:
    """Get the visible text of a text element using AXVisibleCharacterRange.

    Uses ``AXVisibleCharacterRange`` to determine which characters are
    on-screen, then ``AXStringForRange`` (a parameterized attribute) to
    fetch just that text.  Returns the visible text string, or ``None``
    if the element doesn't support these attributes.

    This is far more efficient than walking child elements for text
    areas, editors, and similar elements with large text content.
    """
    if not HAS_ACCESSIBILITY:
        return None
    try:
        vis_range = ax_get_attribute(element, "AXVisibleCharacterRange")
        if vis_range is None:
            return None
        # vis_range is an AXValueRef wrapping a CFRange
        ok, cf_range = AXValueGetValue(vis_range, kAXValueTypeCFRange, None)
        if not ok:
            return None
        # Use parameterized attribute AXStringForRange
        err, text = ApplicationServices.AXUIElementCopyParameterizedAttributeValue(
            element, "AXStringForRange", vis_range, None
        )
        if err == 0 and isinstance(text, str):
            return text
    except Exception:
        logger.debug("Failed to get visible text via AXStringForRange", exc_info=True)
    return None


def ax_set_attribute(element, attribute: str, value) -> bool:
    """Safely set an accessibility attribute."""
    if not HAS_ACCESSIBILITY:
        return False
    err = ApplicationServices.AXUIElementSetAttributeValue(
        element, attribute, value
    )
    return err == 0


def ax_is_attribute_settable(element, attribute: str) -> bool:
    """Check if an accessibility attribute is settable."""
    if not HAS_ACCESSIBILITY:
        return False
    err, settable = ApplicationServices.AXUIElementIsAttributeSettable(
        element, attribute, None
    )
    return err == 0 and settable


def ax_get_position(element) -> tuple:
    """Get the screen position of an accessibility element.

    Returns (x, y) tuple or None.
    """
    pos = ax_get_attribute(element, "AXPosition")
    if pos is not None:
        try:
            success, point = AXValueGetValue(pos, kAXValueTypeCGPoint, None)
            if success:
                logger.log(5, f"AXPosition: ({point.x:.0f}, {point.y:.0f})")
                return (point.x, point.y)
        except Exception:
            logger.debug("Failed to extract AXPosition", exc_info=True)
    return None


def ax_get_size(element) -> tuple:
    """Get the size of an accessibility element.

    Returns (width, height) tuple or None.
    """
    size = ax_get_attribute(element, "AXSize")
    if size is not None:
        try:
            success, sz = AXValueGetValue(size, kAXValueTypeCGSize, None)
            if success:
                logger.log(5, f"AXSize: ({sz.width:.0f}, {sz.height:.0f})")
                return (sz.width, sz.height)
        except Exception:
            logger.debug("Failed to extract AXSize", exc_info=True)
    return None


def ax_get_pid(element) -> int:
    """Get the PID of the process owning an accessibility element."""
    if not HAS_ACCESSIBILITY:
        return 0
    try:
        err, pid = AXUIElementGetPid(element, None)
        if err == 0:
            return pid
    except Exception:
        pass
    return 0


def dump_ax_tree(
    element,
    out,
    max_depth: int = 12,
    max_children: int = 50,
    depth: int = 0,
) -> None:
    """Recursively dump the AX tree with useful attributes.

    Writes a human-readable representation to *out* (any object with a
    ``write`` method — a file, ``sys.stdout``, or ``io.StringIO``).
    Uses batch attribute fetching for performance.
    """
    if depth > max_depth:
        out.write(f"{'  ' * depth}... (depth limit {max_depth})\n")
        return

    a = ax_get_tree_attrs(element)
    role = a.get("AXRole") or "?"
    subrole = a.get("AXSubrole")
    value = a.get("AXValue")
    title = a.get("AXTitle")
    desc = a.get("AXDescription")
    role_desc = a.get("AXRoleDescription")
    placeholder = a.get("AXPlaceholderValue")
    num_chars = a.get("AXNumberOfCharacters")
    children = a.get("children") or []
    n_children = len(children) if children else 0

    attrs: list[str] = []
    if isinstance(value, str):
        display = value.replace("\n", "\\n")
        if len(display) > 120:
            display = display[:120] + "..."
        attrs.append(f'val="{display}"')
    elif value is not None:
        attrs.append(f"val=({type(value).__name__})")
    if isinstance(title, str) and title.strip():
        attrs.append(f'title="{title[:80]}"')
    if isinstance(desc, str) and desc.strip():
        attrs.append(f'desc="{desc[:80]}"')
    if subrole:
        attrs.append(f"subrole={subrole}")
    if isinstance(role_desc, str) and role_desc.strip():
        attrs.append(f'rdesc="{role_desc}"')
    if placeholder is not None:
        attrs.append(f'placeholder="{placeholder}"')
    if num_chars is not None:
        attrs.append(f"numChars={num_chars}")

    indent = "  " * depth
    attr_str = "  " + " | ".join(attrs) if attrs else ""
    out.write(f"{indent}[{depth}] {role} ({n_children} ch){attr_str}\n")

    if children:
        for child in children[:max_children]:
            dump_ax_tree(child, out, max_depth, max_children, depth + 1)
        if len(children) > max_children:
            out.write(
                f"{indent}  ... +{len(children) - max_children} more children\n"
            )


def serialize_ax_tree(
    element,
    max_depth: int = 12,
    max_children: int = 50,
    depth: int = 0,
    focused_element=None,
) -> dict | None:
    """Recursively serialize the AX tree to a JSON-compatible dict.

    Mirrors ``dump_ax_tree`` but returns structured data instead of text.
    Returns ``None`` if *max_depth* is exceeded.

    If *focused_element* is provided (an AXUIElement), the node matching
    that element is marked with ``"focused": true`` along with cursor
    metadata (``cursorPosition``, ``selectionLength``).  Every ancestor
    of the focused node is marked with ``"ancestorOfFocused": true`` so
    the path from root to the input field is easy to trace.
    """
    if depth > max_depth:
        return None

    a = ax_get_tree_attrs(element)
    role = a.get("AXRole") or ""
    subrole = a.get("AXSubrole") or ""
    role_desc = a.get("AXRoleDescription") or ""
    value = a.get("AXValue")
    title = a.get("AXTitle") or ""
    desc = a.get("AXDescription") or ""
    placeholder = a.get("AXPlaceholderValue")
    num_chars = a.get("AXNumberOfCharacters")
    children = a.get("children") or []

    node: dict = {
        "role": role,
        "subrole": subrole,
        "roleDescription": role_desc,
        "value": value if isinstance(value, str) else None,
        "title": title,
        "description": desc,
        "placeholderValue": placeholder if isinstance(placeholder, str) else None,
        "numberOfCharacters": num_chars if isinstance(num_chars, (int, float)) else None,
        "children": [],
    }

    # Check if this element is the focused (keyboard-active) element.
    is_focused = False
    if focused_element is not None:
        try:
            is_focused = element == focused_element
        except Exception:
            pass

    if is_focused:
        node["focused"] = True
        # Capture cursor / selection range for the focused element.
        try:
            if HAS_ACCESSIBILITY:
                from ApplicationServices import AXValueGetValue

                sel_range = ax_get_attribute(element, "AXSelectedTextRange")
                if sel_range is not None:
                    ok, cf_range = AXValueGetValue(
                        sel_range, kAXValueTypeCFRange, None
                    )
                    if ok:
                        node["cursorPosition"] = cf_range.location
                        node["selectionLength"] = cf_range.length
        except Exception:
            pass

    # Recurse into children, tracking whether any descendant is focused.
    child_has_focus = False
    if children:
        for child in children[:max_children]:
            child_node = serialize_ax_tree(
                child, max_depth, max_children, depth + 1, focused_element
            )
            if child_node is not None:
                node["children"].append(child_node)
                if child_node.get("focused") or child_node.get("ancestorOfFocused"):
                    child_has_focus = True

    if child_has_focus:
        node["ancestorOfFocused"] = True

    return node
