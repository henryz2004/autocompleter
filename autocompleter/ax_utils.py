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
    """
    if depth > max_depth:
        out.write(f"{'  ' * depth}... (depth limit {max_depth})\n")
        return

    role = ax_get_attribute(element, "AXRole") or "?"
    subrole = ax_get_attribute(element, "AXSubrole")
    value = ax_get_attribute(element, "AXValue")
    title = ax_get_attribute(element, "AXTitle")
    desc = ax_get_attribute(element, "AXDescription")
    role_desc = ax_get_attribute(element, "AXRoleDescription")
    placeholder = ax_get_attribute(element, "AXPlaceholderValue")
    num_chars = ax_get_attribute(element, "AXNumberOfCharacters")
    children = ax_get_attribute(element, "AXChildren") or []
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
