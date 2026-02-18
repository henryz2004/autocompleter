"""Input Observer using macOS Accessibility API via pyobjc.

Detects the currently focused text field, reads visible text content
in the active window, and monitors for typing pauses or hotkey triggers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

try:
    import AppKit
    import ApplicationServices
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
    )
    from CoreFoundation import CFEqual

    HAS_ACCESSIBILITY = True
except ImportError:
    HAS_ACCESSIBILITY = False

logger = logging.getLogger(__name__)


@dataclass
class FocusedElement:
    """Information about the currently focused text input."""

    app_name: str
    app_pid: int
    role: str
    value: str
    selected_text: str
    position: tuple[float, float] | None  # (x, y) screen coordinates
    size: tuple[float, float] | None  # (width, height)


@dataclass
class VisibleContent:
    """Content visible in the active window."""

    app_name: str
    app_pid: int
    window_title: str
    text_elements: list[str]
    url: str


def _ax_get_attribute(element, attribute: str):
    """Safely get an accessibility attribute from an element."""
    if not HAS_ACCESSIBILITY:
        return None
    err, value = ApplicationServices.AXUIElementCopyAttributeValue(
        element, attribute, None
    )
    if err == 0:
        return value
    return None


def _ax_get_position(element) -> tuple[float, float] | None:
    """Get the screen position of an accessibility element."""
    pos = _ax_get_attribute(element, "AXPosition")
    if pos is not None:
        try:
            point = AppKit.NSValue.valueWithPoint_(
                AppKit.NSPoint(0, 0)
            ).pointValue()
            # AXPosition returns an AXValue wrapping a CGPoint
            import Quartz

            success, point = Quartz.AXValueGetValue(
                pos, Quartz.kAXValueTypeCGPoint, None
            )
            if success:
                return (point.x, point.y)
        except Exception:
            pass
    return None


def _ax_get_size(element) -> tuple[float, float] | None:
    """Get the size of an accessibility element."""
    size = _ax_get_attribute(element, "AXSize")
    if size is not None:
        try:
            import Quartz

            success, sz = Quartz.AXValueGetValue(
                size, Quartz.kAXValueTypeCGSize, None
            )
            if success:
                return (sz.width, sz.height)
        except Exception:
            pass
    return None


class InputObserver:
    """Observes focused text inputs using the macOS Accessibility API."""

    def __init__(self):
        if not HAS_ACCESSIBILITY:
            logger.warning(
                "pyobjc accessibility modules not available. "
                "Input observation will use stub data."
            )
            self._system_wide = None
        else:
            self._system_wide = AXUIElementCreateSystemWide()
        self._last_value: str = ""
        self._last_change_time: float = 0.0

    @staticmethod
    def check_accessibility_permissions() -> bool:
        """Check if the app has accessibility permissions."""
        if not HAS_ACCESSIBILITY:
            return False
        return bool(AXIsProcessTrusted())

    def get_focused_element(self) -> FocusedElement | None:
        """Get information about the currently focused text element."""
        if not HAS_ACCESSIBILITY or self._system_wide is None:
            return None

        focused = _ax_get_attribute(self._system_wide, "AXFocusedUIElement")
        if focused is None:
            return None

        role = _ax_get_attribute(focused, "AXRole") or ""
        # Only care about text-input-like roles
        text_roles = {
            "AXTextField",
            "AXTextArea",
            "AXComboBox",
            "AXWebArea",
            "AXGroup",  # contenteditable divs often show as AXGroup
        }
        if role not in text_roles:
            return None

        value = _ax_get_attribute(focused, "AXValue") or ""
        selected = _ax_get_attribute(focused, "AXSelectedText") or ""

        # Get the owning application
        pid = _ax_get_attribute(focused, "AXPid") or 0
        app_name = self._get_app_name(pid) if pid else "Unknown"

        position = _ax_get_position(focused)
        size = _ax_get_size(focused)

        # Track value changes for typing-pause detection
        if value != self._last_value:
            self._last_value = value
            self._last_change_time = time.time()

        return FocusedElement(
            app_name=app_name,
            app_pid=pid,
            role=role,
            value=value,
            selected_text=selected,
            position=position,
            size=size,
        )

    def get_visible_content(self) -> VisibleContent | None:
        """Read visible text content from the active window.

        Traverses the accessibility tree of the frontmost application
        to extract text elements visible in the current window.
        """
        if not HAS_ACCESSIBILITY:
            return None

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if front_app is None:
            return None

        pid = front_app.processIdentifier()
        app_name = front_app.localizedName() or "Unknown"

        app_element = AXUIElementCreateApplication(pid)

        # Get the focused window
        window = _ax_get_attribute(app_element, "AXFocusedWindow")
        if window is None:
            # Fall back to first window
            windows = _ax_get_attribute(app_element, "AXWindows")
            if windows and len(windows) > 0:
                window = windows[0]
            else:
                return None

        window_title = _ax_get_attribute(window, "AXTitle") or ""

        # Extract text elements from the window
        text_elements = []
        self._collect_text(window, text_elements, max_depth=10, max_items=100)

        # Try to get URL from browser
        url = self._get_browser_url(app_element, app_name)

        return VisibleContent(
            app_name=app_name,
            app_pid=pid,
            window_title=window_title,
            text_elements=text_elements,
            url=url,
        )

    def _collect_text(
        self,
        element,
        results: list[str],
        max_depth: int,
        max_items: int,
        depth: int = 0,
    ) -> None:
        """Recursively collect text content from accessibility elements."""
        if depth > max_depth or len(results) >= max_items:
            return

        # Get text value
        value = _ax_get_attribute(element, "AXValue")
        if isinstance(value, str) and value.strip():
            results.append(value.strip())

        # Also check AXTitle and AXDescription
        title = _ax_get_attribute(element, "AXTitle")
        if isinstance(title, str) and title.strip():
            results.append(title.strip())

        description = _ax_get_attribute(element, "AXDescription")
        if isinstance(description, str) and description.strip():
            results.append(description.strip())

        # Check static text
        role = _ax_get_attribute(element, "AXRole")
        if role == "AXStaticText":
            st_value = _ax_get_attribute(element, "AXValue")
            if isinstance(st_value, str) and st_value.strip():
                if st_value.strip() not in results:
                    results.append(st_value.strip())

        # Recurse into children
        children = _ax_get_attribute(element, "AXChildren")
        if children:
            for child in children:
                if len(results) >= max_items:
                    break
                self._collect_text(
                    child, results, max_depth, max_items, depth + 1
                )

    def _get_browser_url(self, app_element, app_name: str) -> str:
        """Try to extract the current URL from a browser app."""
        browser_names = {"Safari", "Google Chrome", "Chromium", "Arc", "Firefox"}
        if app_name not in browser_names:
            return ""

        # For Safari
        if app_name == "Safari":
            # Try AXDocument attribute on the focused window
            window = _ax_get_attribute(app_element, "AXFocusedWindow")
            if window:
                doc = _ax_get_attribute(window, "AXDocument")
                if isinstance(doc, str):
                    return doc

        # For Chrome-based browsers, try the address bar
        window = _ax_get_attribute(app_element, "AXFocusedWindow")
        if window:
            toolbar = self._find_element_by_role(window, "AXToolbar", max_depth=3)
            if toolbar:
                text_field = self._find_element_by_role(
                    toolbar, "AXTextField", max_depth=3
                )
                if text_field:
                    value = _ax_get_attribute(text_field, "AXValue")
                    if isinstance(value, str):
                        return value

        return ""

    def _find_element_by_role(self, element, role: str, max_depth: int, depth: int = 0):
        """Find the first child element matching a given role."""
        if depth > max_depth:
            return None

        el_role = _ax_get_attribute(element, "AXRole")
        if el_role == role:
            return element

        children = _ax_get_attribute(element, "AXChildren")
        if children:
            for child in children:
                result = self._find_element_by_role(
                    child, role, max_depth, depth + 1
                )
                if result is not None:
                    return result

        return None

    @staticmethod
    def _get_app_name(pid: int) -> str:
        """Get application name from PID."""
        if not HAS_ACCESSIBILITY:
            return "Unknown"
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        for app in workspace.runningApplications():
            if app.processIdentifier() == pid:
                return app.localizedName() or "Unknown"
        return "Unknown"

    def has_typing_paused(self, pause_ms: int = 500) -> bool:
        """Check if the user has paused typing for at least pause_ms."""
        if self._last_change_time == 0:
            return False
        elapsed_ms = (time.time() - self._last_change_time) * 1000
        return elapsed_ms >= pause_ms
