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
from .conversation_extractors import (
    ConversationTurn,
    _collect_child_text,
    get_extractor,
)

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
    placeholder_detected: bool = False  # True when value was cleared due to placeholder detection

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
class VisibleContent:
    """Content visible in the active window."""

    app_name: str
    app_pid: int
    window_title: str
    text_elements: list[str]
    url: str
    conversation_turns: list[ConversationTurn] | None = None


# App-specific known placeholder prefixes.  Some Electron apps (Gemini Desktop)
# bake placeholder text into AXValue with no distinguishing AX attributes.
# Keyed by app name (as reported by NSWorkspace.localizedName()).
_APP_PLACEHOLDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "Google Gemini": ("Ask Gemini",),
}


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

        # Extract cursor position from AXSelectedTextRange BEFORE placeholder
        # detection — Strategy 3 below needs insertion_point.
        # The range is an AXValueRef wrapping a CFRange.  pyobjc's
        # AXValueGetValue returns it as a (location, length) tuple.
        insertion_point = None
        selection_length = 0
        sel_range = ax_get_attribute(focused, "AXSelectedTextRange")
        if sel_range is not None:
            try:
                from ApplicationServices import AXValueGetValue, kAXValueTypeCFRange
                success, cf_range = AXValueGetValue(sel_range, kAXValueTypeCFRange, None)
                if success:
                    # cf_range may be a tuple (location, length) or a struct
                    if isinstance(cf_range, tuple):
                        insertion_point = cf_range[0]
                        selection_length = cf_range[1]
                    else:
                        insertion_point = cf_range.location
                        selection_length = cf_range.length
                    logger.debug(
                        f"Cursor: insertion_point={insertion_point}, "
                        f"selection_length={selection_length}"
                    )
                else:
                    logger.debug("AXValueGetValue failed for AXSelectedTextRange")
            except Exception:
                logger.debug("Could not extract range from AXSelectedTextRange",
                             exc_info=True)

        # Get PID via AXUIElementGetPid (needed by Strategy 4 below)
        pid = ax_get_pid(focused)
        app_name = self._get_app_name(pid) if pid else "Unknown"

        # Detect placeholder text — many apps expose placeholder strings
        # like "Reply..." as AXValue when the field is actually empty.
        # Strategies (in order):
        #   1. AXPlaceholderValue match (reliable, used by ChatGPT etc.)
        #   2. AXNumberOfCharacters == 0 but AXValue is non-empty (web pages)
        #   3. AXPlaceholderValue attr exists (even empty) + cursor at pos 0
        #      + short value — catches Electron apps like Claude Desktop that
        #      don't properly expose their placeholder string.
        #   4. App-specific known placeholder prefixes (Gemini Desktop etc.)
        placeholder_raw = ax_get_attribute(focused, "AXPlaceholderValue")
        placeholder = placeholder_raw or ""
        num_chars = ax_get_attribute(focused, "AXNumberOfCharacters")
        placeholder_detected = False
        if placeholder and value.strip().rstrip("\n") == placeholder.strip():
            logger.debug(f"[CTX] Value matches AXPlaceholderValue: {placeholder!r}, treating as empty")
            value = ""
            placeholder_detected = True
        elif num_chars is not None and num_chars == 0 and value.strip():
            logger.debug(
                f"[CTX] AXNumberOfCharacters=0 but AXValue={value.strip()!r}, treating as placeholder"
            )
            value = ""
            placeholder_detected = True
        elif (insertion_point == 0 and selection_length == 0
              and value.strip() and len(value.strip()) < 50):
            # Strategy 3: cursor at position 0 with short text and no selection.
            # The user hasn't started typing (cursor at start), so the field
            # content is almost certainly placeholder/decoration text.
            # This catches Electron apps (Claude Desktop, Slack) that don't
            # properly expose AXPlaceholderValue.
            logger.debug(
                f"[CTX] Cursor at 0 with short value={value.strip()!r}, "
                f"placeholder_raw={placeholder_raw!r} — treating as placeholder"
            )
            value = ""
            placeholder_detected = True
        elif app_name in _APP_PLACEHOLDER_PREFIXES:
            # Strategy 4: App-specific known placeholder patterns.
            # Some Electron apps (Gemini Desktop) bake placeholder text into
            # AXValue with no distinguishing AX attributes.
            prefixes = _APP_PLACEHOLDER_PREFIXES[app_name]
            stripped = value.strip()
            if any(stripped.startswith(p) for p in prefixes) and len(stripped) < 50:
                logger.debug(
                    f"[CTX] App={app_name!r} value={stripped!r} matches known "
                    f"placeholder prefix — treating as placeholder"
                )
                value = ""
                placeholder_detected = True
        else:
            if value.strip():
                logger.debug(
                    f"[CTX] Placeholder check passed: AXPlaceholderValue={placeholder_raw!r}, "
                    f"AXNumberOfCharacters={num_chars}, insertion_point={insertion_point}"
                )

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
            placeholder_detected=placeholder_detected,
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
                logger.debug("[CTX] No window found for %s", app_name)
                return None

        window_title = ax_get_attribute(window, "AXTitle") or ""
        logger.debug("[CTX] app=%r window=%r", app_name, window_title)

        # Extract text elements.  For web apps (PWAs, Electron) skip the
        # browser scaffolding by starting from the AXWebArea — this avoids
        # needing an artificially high depth limit to punch through the
        # many wrapper <div>/AXGroup layers that Chromium generates.
        text_elements: list[str] = []
        stats: dict[str, int] = {"visited": 0, "max_depth_hit": 0, "skipped_chrome": 0,
                                  "no_value": 0, "too_short": 0, "placeholder": 0,
                                  "from_desc": 0}
        web_area = self._find_element_by_role(window, "AXWebArea", max_depth=15)
        content_root = web_area if web_area else window
        self._collect_text(content_root, text_elements, max_depth=20, max_items=100, _stats=stats)
        logger.debug(
            "[CTX] _collect_text: %d elements | visited=%d skipped_chrome=%d "
            "no_value=%d too_short=%d placeholder=%d from_desc=%d max_depth_hit=%d",
            len(text_elements), stats["visited"], stats["skipped_chrome"],
            stats["no_value"], stats["too_short"], stats["placeholder"],
            stats["from_desc"], stats["max_depth_hit"],
        )

        # Fallback for Electron/Chromium apps: if _collect_text returned nothing,
        # use _collect_child_text which aggregates text from AXStaticText children
        # regardless of parent role. This handles:
        #  - Electron apps that fragment text into 1-2 char AXStaticText elements
        #  - Apps where content elements have no AXValue but have text children
        #  - Deeply nested trees without AXWebArea
        if not text_elements:
            # Try AXWebArea first (browsers), then fall back to window
            target = self._find_element_by_role(window, "AXWebArea", max_depth=10)
            target_label = "AXWebArea"
            if target is None:
                target = window
                target_label = "window"
            logger.debug("[CTX] Fallback: collecting child text from %s", target_label)
            raw = _collect_child_text(target, max_depth=15, max_chars=6000)
            if raw.strip():
                for line in raw.splitlines():
                    stripped = line.strip()
                    if len(stripped) >= self._MIN_TEXT_LEN:
                        text_elements.append(stripped)
                logger.debug(
                    "[CTX] Fallback: collected %d elements from %s (%d chars)",
                    len(text_elements), target_label, len(raw),
                )
            else:
                logger.debug("[CTX] Fallback: child text from %s was empty", target_label)

        # Reverse so that content elements (deeper in the tree, rendered later)
        # come first. When truncated, this keeps actual content and drops
        # top-of-tree UI chrome (headers, navigation, toolbars that passed
        # the role filter).
        text_elements.reverse()

        # Try to get URL from browser
        url = self._get_browser_url(app_element, app_name)

        # Try structured conversation extraction for chat-like apps
        conversation_turns = self._extract_conversation_turns(window, app_name=app_name)

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

    # Cap children iterated per node and total nodes visited to prevent
    # runaway traversal on highly branched AX trees.
    _MAX_CHILDREN_PER_NODE = 50
    _MAX_VISITS = 600

    def _collect_text(
        self,
        element,
        results: list[str],
        max_depth: int,
        max_items: int,
        depth: int = 0,
        _seen: set[str] | None = None,
        _stats: dict[str, int] | None = None,
    ) -> None:
        """Recursively collect text content from accessibility elements.

        Filters out UI chrome (toolbars, buttons, menus, scrollbars) and
        only extracts from content-bearing roles. Skips very short strings.
        Uses a set for O(1) dedup checks.
        """
        if depth > max_depth or len(results) >= max_items:
            if _stats and depth > max_depth:
                _stats["max_depth_hit"] = _stats.get("max_depth_hit", 0) + 1
            return

        # Global visit budget — stop traversal if we've visited too many nodes
        if _stats and _stats.get("visited", 0) >= self._MAX_VISITS:
            return

        if _seen is None:
            _seen = set()

        role = ax_get_attribute(element, "AXRole") or ""
        if _stats:
            _stats["visited"] = _stats.get("visited", 0) + 1

        # Skip entire subtrees for UI chrome roles
        if role in self._SKIP_ROLES:
            if _stats:
                _stats["skipped_chrome"] = _stats.get("skipped_chrome", 0) + 1
            return

        # Extract text only from content-bearing roles
        if role in self._CONTENT_ROLES:
            value = ax_get_attribute(element, "AXValue")
            text: str | None = None
            if isinstance(value, str) and value.strip():
                text = value.strip()
            else:
                # Fallback: many Electron apps (ChatGPT, Slack) put message
                # text in AXDescription instead of AXValue.
                desc = ax_get_attribute(element, "AXDescription")
                if isinstance(desc, str) and desc.strip():
                    text = desc.strip()
                    if _stats:
                        _stats["from_desc"] = _stats.get("from_desc", 0) + 1
                else:
                    if _stats:
                        _stats["no_value"] = _stats.get("no_value", 0) + 1

            if text is not None:
                if len(text) < self._MIN_TEXT_LEN:
                    if _stats:
                        _stats["too_short"] = _stats.get("too_short", 0) + 1
                elif text in _seen:
                    pass  # dedup
                else:
                    # For input fields, skip placeholder text so it doesn't
                    # pollute visible context (e.g. "Reply..." in chat sidebars)
                    if role in {"AXTextField", "AXTextArea"}:
                        ph = ax_get_attribute(element, "AXPlaceholderValue")
                        nc = ax_get_attribute(element, "AXNumberOfCharacters")
                        if (ph and text.rstrip("\n") == ph.strip()) or (
                            nc is not None and nc == 0 and text
                        ):
                            if _stats:
                                _stats["placeholder"] = _stats.get("placeholder", 0) + 1
                        else:
                            _seen.add(text)
                            results.append(text)
                    else:
                        _seen.add(text)
                        results.append(text)

        # Recurse into children (capped per node to avoid wide-tree blowup)
        children = ax_get_attribute(element, "AXChildren")
        if children:
            for child in children[:self._MAX_CHILDREN_PER_NODE]:
                if len(results) >= max_items:
                    break
                if _stats and _stats.get("visited", 0) >= self._MAX_VISITS:
                    break
                self._collect_text(
                    child, results, max_depth, max_items, depth + 1, _seen,
                    _stats,
                )

    def _extract_conversation_turns(
        self, window, max_turns: int = 15, app_name: str = ""
    ) -> list[ConversationTurn] | None:
        """Try to extract structured conversation turns from a chat window.

        Uses app-specific extractors when available (Gemini, Slack, ChatGPT,
        Claude Desktop, iMessage). Falls back to a generic heuristic for
        unknown apps.
        """
        extractor = get_extractor(app_name)
        logger.debug(
            "[CTX] Using %s for app %r",
            type(extractor).__name__, app_name,
        )
        try:
            return extractor.extract(window, max_turns)
        except Exception:
            logger.debug(
                "Conversation extraction failed with %s",
                type(extractor).__name__, exc_info=True,
            )
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

    _MAX_FIND_VISITS = 500  # visit budget for _find_element_by_role

    def _find_element_by_role(
        self, element, role: str, max_depth: int, depth: int = 0,
        _visits: list | None = None,
    ):
        """Find the first child element matching a given role."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_FIND_VISITS:
            return None
        if depth > max_depth:
            return None

        el_role = ax_get_attribute(element, "AXRole")
        if el_role == role:
            return element

        children = ax_get_attribute(element, "AXChildren")
        if children:
            for child in children[:self._MAX_CHILDREN_PER_NODE]:
                result = self._find_element_by_role(
                    child, role, max_depth, depth + 1, _visits,
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
