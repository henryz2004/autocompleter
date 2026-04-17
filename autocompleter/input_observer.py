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

from .ax_utils import ax_get_attribute, ax_get_children, ax_get_pid, ax_get_position, ax_get_size
from .conversation_extractors import (
    ConversationTurn,
    _collect_child_text,
    get_extractor,
)
from .suggestion_engine import is_shell_app

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
    raw_value: str = ""
    raw_placeholder_value: str = ""
    raw_number_of_characters: int | None = None

    # Zero-width characters injected by some apps (e.g. Discord \ufeff BOM)
    # that should be stripped from text sent to the LLM.
    _ZERO_WIDTH_CHARS = "\ufeff\u200b\u200c\u200d\u2060\ufffe"

    @property
    def before_cursor(self) -> str:
        """Text before the cursor/selection (zero-width chars stripped)."""
        if self.insertion_point is None:
            return self.value.strip(self._ZERO_WIDTH_CHARS)
        return self.value[:self.insertion_point].strip(self._ZERO_WIDTH_CHARS)

    @property
    def after_cursor(self) -> str:
        """Text after the cursor/selection (zero-width chars stripped)."""
        if self.insertion_point is None:
            return ""
        return self.value[self.insertion_point + self.selection_length:].strip(self._ZERO_WIDTH_CHARS)


@dataclass
class VisibleContent:
    """Content visible in the active window."""

    app_name: str
    app_pid: int
    window_title: str
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
        # Cache conversation extraction results (expensive for deep AX trees)
        self._cached_conversation_key: str = ""  # app+window key
        self._cached_conversation_turns: list | None = None
        self._cached_conversation_time: float = 0.0
        self._CONVERSATION_CACHE_TTL: float = 5.0  # seconds

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
        raw_value = value
        placeholder_raw = ax_get_attribute(focused, "AXPlaceholderValue")
        placeholder = placeholder_raw or ""
        num_chars = ax_get_attribute(focused, "AXNumberOfCharacters")
        placeholder_detected = False
        if placeholder and value.strip().rstrip("\n") == placeholder.strip():
            logger.debug(f"[CTX] Value matches AXPlaceholderValue: {placeholder!r}, treating as empty")
            value = ""
            placeholder_detected = True
        elif num_chars is not None and num_chars == 0 and value.strip() and not is_shell_app(app_name):
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
            raw_value=raw_value,
            raw_placeholder_value=placeholder,
            raw_number_of_characters=num_chars,
        )

    def get_visible_content(self) -> VisibleContent | None:
        """Read visible content metadata from the active window.

        Extracts conversation turns (for chat apps) and URL. Text context
        is provided by get_subtree_context() instead of flat text collection.
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
            windows = ax_get_attribute(app_element, "AXWindows")
            if windows and len(windows) > 0:
                window = windows[0]
            else:
                logger.debug("[CTX] No window found for %s", app_name)
                return None

        window_title = ax_get_attribute(window, "AXTitle") or ""
        logger.debug("[CTX] app=%r window=%r", app_name, window_title)

        url = self._get_browser_url(app_element, app_name)

        conversation_turns = self._extract_conversation_turns(
            window, app_name=app_name, window_title=window_title,
        )

        return VisibleContent(
            app_name=app_name,
            app_pid=pid,
            window_title=window_title,
            url=url,
            conversation_turns=conversation_turns,
        )

    def get_subtree_context(self, token_budget: int = 1000) -> str | None:
        """Extract context XML by walking up from the focused element.

        Uses the subtree walker to produce minimal XML context from the
        focused element's nearby content — no per-app extractor needed.

        Returns XML string or None if unavailable.
        """
        if not HAS_ACCESSIBILITY:
            return None

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if front_app is None:
            return None

        pid = front_app.processIdentifier()
        app_element = AXUIElementCreateApplication(pid)

        window = ax_get_attribute(app_element, "AXFocusedWindow")
        if window is None:
            windows = ax_get_attribute(app_element, "AXWindows")
            if windows and len(windows) > 0:
                window = windows[0]
            else:
                return None

        focused_el = ax_get_attribute(app_element, "AXFocusedUIElement")
        if focused_el is None:
            return None

        try:
            from .subtree_context import extract_context_live

            xml = extract_context_live(
                window, focused_el, max_depth=40, token_budget=token_budget,
            )
            if xml:
                logger.debug(
                    "[CTX] Subtree context: %d chars (~%d tokens)",
                    len(xml), len(xml) // 4,
                )
            return xml
        except Exception:
            logger.debug("Subtree context extraction failed", exc_info=True)
            return None

    def get_context_bundle(
        self,
        focused_state: FocusedElement,
        token_budget: int = 1000,
        overview_token_budget: int = 120,
    ):
        """Build the focus-aware tree context bundle used by prompts/dumps."""
        if not HAS_ACCESSIBILITY:
            return None

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if front_app is None:
            return None

        pid = front_app.processIdentifier()
        app_element = AXUIElementCreateApplication(pid)

        window = ax_get_attribute(app_element, "AXFocusedWindow")
        if window is None:
            windows = ax_get_attribute(app_element, "AXWindows")
            if windows and len(windows) > 0:
                window = windows[0]
            else:
                return None

        focused_el = ax_get_attribute(app_element, "AXFocusedUIElement")
        if focused_el is None:
            return None

        try:
            from .subtree_context import build_context_bundle_live

            return build_context_bundle_live(
                window,
                focused_el,
                focused_value=focused_state.value,
                placeholder_detected=focused_state.placeholder_detected,
                insertion_point=focused_state.insertion_point,
                selection_length=focused_state.selection_length,
                raw_value=focused_state.raw_value,
                raw_placeholder_value=focused_state.raw_placeholder_value,
                raw_number_of_characters=focused_state.raw_number_of_characters,
                max_depth=40,
                token_budget=token_budget,
                overview_token_budget=overview_token_budget,
            )
        except Exception:
            logger.debug("Tree context bundle extraction failed", exc_info=True)
            return None

    def get_focus_debug_info(
        self,
        *,
        max_depth: int = 12,
        candidate_limit: int = 8,
    ) -> dict[str, object]:
        """Collect rich focus diagnostics for remote debugging."""
        if not HAS_ACCESSIBILITY:
            return {"has_accessibility": False}

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if front_app is None:
            return {"has_accessibility": True, "frontmost_app": None}

        app_name = front_app.localizedName() or "Unknown"
        pid = front_app.processIdentifier()
        app_element = AXUIElementCreateApplication(pid)

        system_focused = (
            ax_get_attribute(self._system_wide, "AXFocusedUIElement")
            if self._system_wide is not None
            else None
        )
        app_local_focused = ax_get_attribute(app_element, "AXFocusedUIElement")

        window = ax_get_attribute(app_element, "AXFocusedWindow")
        if window is None:
            windows = ax_get_attribute(app_element, "AXWindows")
            if windows and len(windows) > 0:
                window = windows[0]

        window_title = ax_get_attribute(window, "AXTitle") or "" if window is not None else ""
        source_url = self._get_browser_url(app_element, app_name) if window is not None else ""

        window_tree = None
        app_local_tree = None
        try:
            from .ax_utils import serialize_ax_tree

            if window is not None:
                window_tree = serialize_ax_tree(
                    window,
                    max_depth=max_depth,
                    max_children=20,
                    focused_element=system_focused,
                )
                if app_local_focused is not None:
                    app_local_tree = serialize_ax_tree(
                        window,
                        max_depth=max_depth,
                        max_children=20,
                        focused_element=app_local_focused,
                    )
        except Exception:
            logger.debug("Failed to serialize window tree for focus diagnostics", exc_info=True)

        return {
            "has_accessibility": True,
            "frontmost_app": {
                "name": app_name,
                "pid": pid,
                "window_title": window_title,
                "source_url": source_url,
            },
            "system_wide_focus_present": system_focused is not None,
            "system_wide_focus_role": (
                ax_get_attribute(system_focused, "AXRole") if system_focused is not None else None
            ),
            "app_local_focus_present": app_local_focused is not None,
            "app_local_focus_role": (
                ax_get_attribute(app_local_focused, "AXRole")
                if app_local_focused is not None
                else None
            ),
            "editable_candidates": self._summarize_editable_candidates(
                window_tree or app_local_tree or {},
                limit=candidate_limit,
            ),
            "window_tree": window_tree,
            "app_local_tree": app_local_tree,
        }

    _MAX_CHILDREN_PER_NODE = 50
    _MAX_FIND_VISITS = 500

    def _extract_conversation_turns(
        self, window, max_turns: int = 15, app_name: str = "",
        window_title: str = "",
    ) -> list[ConversationTurn] | None:
        """Try to extract structured conversation turns from a chat window.

        Uses app-specific extractors when available (Gemini, Slack, ChatGPT,
        Claude Desktop, Codex, iMessage, WhatsApp, Discord). Falls back to a
        generic heuristic for unknown apps. For browsers, *window_title* is
        used to pick the right extractor.

        Results are cached for ``_CONVERSATION_CACHE_TTL`` seconds to avoid
        expensive AX tree traversals on every observer poll cycle.
        """
        import time as _time

        cache_key = f"{app_name}\0{window_title}"
        now = _time.monotonic()
        if (
            cache_key == self._cached_conversation_key
            and (now - self._cached_conversation_time) < self._CONVERSATION_CACHE_TTL
        ):
            return self._cached_conversation_turns

        extractor = get_extractor(app_name, window_title=window_title)
        logger.debug(
            "[CTX] Using %s for app %r",
            type(extractor).__name__, app_name,
        )
        try:
            turns = extractor.extract(window, max_turns)
        except Exception:
            logger.debug(
                "Conversation extraction failed with %s",
                type(extractor).__name__, exc_info=True,
            )
            turns = None

        # Cache the result
        self._cached_conversation_key = cache_key
        self._cached_conversation_turns = turns
        self._cached_conversation_time = now
        return turns

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

    @staticmethod
    def _summarize_editable_candidates(
        tree: dict,
        *,
        limit: int,
    ) -> list[dict[str, object]]:
        if not tree:
            return []

        results: list[dict[str, object]] = []

        def _walk(node: dict, path: str, depth: int) -> None:
            if len(results) >= limit:
                return
            role = str(node.get("role") or "")
            role_desc = str(node.get("roleDescription") or "")
            desc = str(node.get("description") or "")
            title = str(node.get("title") or "")
            placeholder = str(node.get("placeholderValue") or "")
            value = str(node.get("value") or "")
            text = " ".join(
                piece for piece in (role_desc, desc, title, placeholder) if piece
            ).lower()
            editableish = (
                role in {"AXTextArea", "AXTextField", "AXWebArea", "AXGroup"}
                and (
                    "text entry" in text
                    or "search text field" in text
                    or "search" in text
                    or bool(placeholder)
                    or bool(value.strip())
                )
            )
            if editableish:
                results.append(
                    {
                        "path": path,
                        "depth": depth,
                        "role": role,
                        "role_description": role_desc,
                        "description": desc,
                        "title": title,
                        "placeholder_value": placeholder,
                        "value_preview": value[:120],
                        "focused": bool(node.get("focused")),
                    }
                )

            for index, child in enumerate(node.get("children") or []):
                if not isinstance(child, dict):
                    continue
                _walk(child, f"{path}/{child.get('role', '?')}[{index}]", depth + 1)

        _walk(tree, "root", 0)
        return results

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

        children = ax_get_children(element)
        if children:
            for child in children[:self._MAX_CHILDREN_PER_NODE]:
                result = self._find_element_by_role(
                    child, role, max_depth, depth + 1, _visits,
                )
                if result is not None:
                    return result

        return None

    def _find_all_elements_by_role(
        self, element, role: str, max_depth: int, depth: int = 0,
        _visits: list | None = None, _results: list | None = None,
    ) -> list:
        """Find ALL child elements matching a given role."""
        if _visits is None:
            _visits = [0]
        if _results is None:
            _results = []
        _visits[0] += 1
        if _visits[0] > self._MAX_FIND_VISITS:
            return _results
        if depth > max_depth:
            return _results

        el_role = ax_get_attribute(element, "AXRole")
        if el_role == role:
            _results.append(element)
            # Don't recurse into children of a match — nested matches
            # would be children of this one anyway.
            return _results

        children = ax_get_children(element)
        if children:
            for child in children[:self._MAX_CHILDREN_PER_NODE]:
                self._find_all_elements_by_role(
                    child, role, max_depth, depth + 1, _visits, _results,
                )
        return _results

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
