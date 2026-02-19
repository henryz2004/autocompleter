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
