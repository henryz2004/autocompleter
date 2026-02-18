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
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
    )

    HAS_ACCESSIBILITY = True
except ImportError:
    HAS_ACCESSIBILITY = False

from .ax_utils import ax_get_attribute, ax_get_pid, ax_get_position, ax_get_size

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
    insertion_point: int | None = None  # Caret position in value (chars from start)
    selection_length: int = 0  # Length of selected text range (0 = no selection)

    @property
    def before_cursor(self) -> str:
        """Text before the cursor/selection."""
        if self.insertion_point is None:
            return self.value
        return self.value[:self.insertion_point]

    @property
    def after_cursor(self) -> str:
        """Text after the cursor/selection."""
        if self.insertion_point is None:
            return ""
        return self.value[self.insertion_point + self.selection_length:]


@dataclass
class ConversationTurn:
    """A single message in a conversation."""
    speaker: str
    text: str


@dataclass
class VisibleContent:
    """Content visible in the active window."""

    app_name: str
    app_pid: int
    window_title: str
    text_elements: list[str]
    url: str
    conversation_turns: list[ConversationTurn] | None = None


def _collect_child_text(
    element, max_depth: int = 5, max_chars: int = 2000, depth: int = 0
) -> str:
    """Collect text from child elements.

    Used as a fallback when AXValue is empty on contenteditable divs
    (common in Chromium-based browsers like Edge and Chrome).
    """
    if depth > max_depth:
        return ""

    parts: list[str] = []
    total = 0
    children = ax_get_attribute(element, "AXChildren")
    if children:
        for child in children:
            if total >= max_chars:
                break
            role = ax_get_attribute(child, "AXRole")
            if role == "AXStaticText":
                val = ax_get_attribute(child, "AXValue")
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                    total += len(val)
            else:
                child_text = _collect_child_text(
                    child, max_depth, max_chars - total, depth + 1
                )
                if child_text:
                    parts.append(child_text)
                    total += len(child_text)

    return "\n".join(parts)


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

        focused = ax_get_attribute(self._system_wide, "AXFocusedUIElement")
        if focused is None:
            logger.debug("No AXFocusedUIElement found")
            return None

        role = ax_get_attribute(focused, "AXRole") or ""
        # Only care about text-input-like roles
        text_roles = {
            "AXTextField",
            "AXTextArea",
            "AXComboBox",
            "AXWebArea",
            "AXGroup",  # contenteditable divs often show as AXGroup
        }
        if role not in text_roles:
            logger.debug(f"Focused element role '{role}' is not a text input")
            return None

        value = ax_get_attribute(focused, "AXValue") or ""
        # Chromium-based browsers (Edge, Chrome) often return empty or
        # whitespace-only AXValue for contenteditable divs. Fall back to
        # collecting child text.
        if not value.strip() and role in {"AXTextArea", "AXWebArea", "AXGroup"}:
            value = _collect_child_text(focused)
            if value:
                logger.debug(
                    f"AXValue was empty, collected {len(value)} chars from children"
                )
        selected = ax_get_attribute(focused, "AXSelectedText") or ""

        # Extract cursor position from AXSelectedTextRange
        insertion_point = None
        selection_length = 0
        sel_range = ax_get_attribute(focused, "AXSelectedTextRange")
        if sel_range is not None:
            try:
                ns_range = sel_range.rangeValue()
                insertion_point = ns_range.location
                selection_length = ns_range.length
                logger.debug(
                    f"Cursor: insertion_point={insertion_point}, "
                    f"selection_length={selection_length}"
                )
            except Exception:
                logger.debug("Could not extract range from AXSelectedTextRange",
                             exc_info=True)

        # Get PID via AXUIElementGetPid
        pid = ax_get_pid(focused)
        app_name = self._get_app_name(pid) if pid else "Unknown"

        position = ax_get_position(focused)
        size = ax_get_size(focused)

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
            insertion_point=insertion_point,
            selection_length=selection_length,
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
        window = ax_get_attribute(app_element, "AXFocusedWindow")
        if window is None:
            # Fall back to first window
            windows = ax_get_attribute(app_element, "AXWindows")
            if windows and len(windows) > 0:
                window = windows[0]
            else:
                return None

        window_title = ax_get_attribute(window, "AXTitle") or ""

        # Extract text elements from the window
        text_elements = []
        self._collect_text(window, text_elements, max_depth=10, max_items=100)

        # Try to get URL from browser
        url = self._get_browser_url(app_element, app_name)

        # Try structured conversation extraction for chat-like apps
        conversation_turns = self._extract_conversation_turns(window)

        return VisibleContent(
            app_name=app_name,
            app_pid=pid,
            window_title=window_title,
            text_elements=text_elements,
            url=url,
            conversation_turns=conversation_turns,
        )

    # Roles that are UI chrome — skip their entire subtree
    _SKIP_ROLES = frozenset({
        "AXToolbar", "AXMenuBar", "AXMenu", "AXMenuItem",
        "AXButton", "AXScrollBar", "AXSlider", "AXIncrementor",
        "AXPopUpButton", "AXCheckBox", "AXRadioButton",
        "AXTabGroup", "AXTab",
    })

    # Roles that carry meaningful text content
    _CONTENT_ROLES = frozenset({
        "AXStaticText", "AXTextField", "AXTextArea",
        "AXWebArea", "AXGroup", "AXCell", "AXRow",
        "AXHeading", "AXLink", "AXParagraph",
    })

    _MIN_TEXT_LEN = 3  # Skip strings <= 2 chars

    def _collect_text(
        self,
        element,
        results: list[str],
        max_depth: int,
        max_items: int,
        depth: int = 0,
    ) -> None:
        """Recursively collect text content from accessibility elements.

        Filters out UI chrome (toolbars, buttons, menus, scrollbars) and
        only extracts from content-bearing roles. Skips very short strings.
        """
        if depth > max_depth or len(results) >= max_items:
            return

        role = ax_get_attribute(element, "AXRole") or ""

        # Skip entire subtrees for UI chrome roles
        if role in self._SKIP_ROLES:
            return

        # Extract text only from content-bearing roles
        if role in self._CONTENT_ROLES:
            value = ax_get_attribute(element, "AXValue")
            if isinstance(value, str):
                stripped = value.strip()
                if len(stripped) >= self._MIN_TEXT_LEN and stripped not in results:
                    results.append(stripped)

        # Recurse into children
        children = ax_get_attribute(element, "AXChildren")
        if children:
            for child in children:
                if len(results) >= max_items:
                    break
                self._collect_text(
                    child, results, max_depth, max_items, depth + 1
                )

    def _extract_conversation_turns(
        self, window, max_turns: int = 15
    ) -> list[ConversationTurn] | None:
        """Try to extract structured conversation turns from a chat window.

        Walks the AX tree looking for message-like groups: containers with
        child elements where one is short (speaker name) and another is
        longer (message body). Falls back to None if structure can't be detected.
        """
        try:
            turns = self._walk_for_messages(window, max_turns, max_depth=8)
            if len(turns) >= 2:
                return turns
        except Exception:
            logger.debug("Conversation extraction failed", exc_info=True)
        return None

    def _walk_for_messages(
        self, element, max_turns: int, max_depth: int, depth: int = 0
    ) -> list[ConversationTurn]:
        """Recursively search for message-like group elements."""
        if depth > max_depth:
            return []

        role = ax_get_attribute(element, "AXRole") or ""
        turns: list[ConversationTurn] = []

        # Look for groups/cells that might be message containers
        if role in {"AXGroup", "AXCell", "AXRow"}:
            turn = self._try_parse_message_group(element)
            if turn is not None:
                turns.append(turn)
                if len(turns) >= max_turns:
                    return turns

        children = ax_get_attribute(element, "AXChildren")
        if children:
            for child in children:
                if len(turns) >= max_turns:
                    break
                child_turns = self._walk_for_messages(
                    child, max_turns - len(turns), max_depth, depth + 1
                )
                turns.extend(child_turns)

        return turns

    def _try_parse_message_group(self, element) -> ConversationTurn | None:
        """Check if an AX element looks like a chat message container.

        Heuristic: a group containing at least two text children where one
        is short (likely a speaker name, <= 40 chars) and another is longer
        (likely the message body, > 5 chars).
        """
        children = ax_get_attribute(element, "AXChildren")
        if not children or len(children) < 2:
            return None

        texts: list[tuple[str, str]] = []  # (role, value)
        for child in children[:10]:  # limit scan
            child_role = ax_get_attribute(child, "AXRole") or ""
            if child_role == "AXStaticText":
                val = ax_get_attribute(child, "AXValue")
                if isinstance(val, str) and val.strip():
                    texts.append((child_role, val.strip()))
            elif child_role in {"AXGroup", "AXTextArea"}:
                # Nested group might contain the message text
                nested_text = _collect_child_text(child, max_depth=3, max_chars=500)
                if nested_text.strip():
                    texts.append((child_role, nested_text.strip()))

        if len(texts) < 2:
            return None

        # Heuristic: find a short text (speaker) and a longer text (body)
        speaker = None
        body = None
        for _, text in texts:
            if len(text) <= 40 and speaker is None:
                speaker = text
            elif len(text) > 5 and body is None:
                body = text

        if speaker and body:
            return ConversationTurn(speaker=speaker, text=body)
        return None

    def _get_browser_url(self, app_element, app_name: str) -> str:
        """Try to extract the current URL from a browser app."""
        browser_names = {"Safari", "Google Chrome", "Chromium", "Arc", "Firefox"}
        if app_name not in browser_names:
            return ""

        # For Safari
        if app_name == "Safari":
            window = ax_get_attribute(app_element, "AXFocusedWindow")
            if window:
                doc = ax_get_attribute(window, "AXDocument")
                if isinstance(doc, str):
                    return doc

        # For Chrome-based browsers, try the address bar
        window = ax_get_attribute(app_element, "AXFocusedWindow")
        if window:
            toolbar = self._find_element_by_role(window, "AXToolbar", max_depth=3)
            if toolbar:
                text_field = self._find_element_by_role(
                    toolbar, "AXTextField", max_depth=3
                )
                if text_field:
                    value = ax_get_attribute(text_field, "AXValue")
                    if isinstance(value, str):
                        return value

        return ""

    def _find_element_by_role(self, element, role: str, max_depth: int, depth: int = 0):
        """Find the first child element matching a given role."""
        if depth > max_depth:
            return None

        el_role = ax_get_attribute(element, "AXRole")
        if el_role == role:
            return element

        children = ax_get_attribute(element, "AXChildren")
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
