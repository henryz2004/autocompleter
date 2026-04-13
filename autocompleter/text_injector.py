"""Text Injector - inserts accepted suggestion text into focused input fields.

Handles contenteditable divs and custom input components used by chat interfaces
(ChatGPT, Claude, Gemini). Must trigger the app's internal state update,
not just visual insertion.

Strategy:
1. Primary: Use the Accessibility API's AXValue setter + AXConfirmAction
2. CDP injection: Use Chrome DevTools Protocol for Chromium-based apps
3. Fallback: Use NSPasteboard (clipboard) + Cmd+V simulation
4. Fallback: Use CGEvents to simulate keyboard input character-by-character
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import AppKit
    import ApplicationServices
    from ApplicationServices import AXUIElementCreateSystemWide
    import Quartz

    HAS_INJECTION = True
except ImportError:
    HAS_INJECTION = False

from .ax_utils import ax_get_attribute, ax_is_attribute_settable, ax_set_attribute
from .cdp_injector import CDPConnection, find_debug_port, is_chromium_app


class TextInjector:
    """Injects text into the currently focused input field."""

    def __init__(self):
        if HAS_INJECTION:
            self._system_wide = AXUIElementCreateSystemWide()
        else:
            self._system_wide = None

    def inject(
        self, text: str, replace: bool = False, insertion_point: Optional[int] = None,
        app_name: str = "", app_pid: int = 0,
    ) -> bool:
        """Inject text into the currently focused input.

        Uses simulated keystrokes as the primary strategy.  AX value
        setting and CDP are avoided because they bypass the app's normal
        input handling — rich text editors (Claude Desktop, ChatGPT,
        ProseMirror/React apps) break when their internal state is mutated
        externally.  Keystrokes go through the standard input pipeline so
        the app processes them natively.

        Falls back to clipboard paste if keystrokes fail.

        Returns True if injection succeeded.
        """
        if not text:
            return False

        # Primary: simulated keystrokes (works with all rich text editors)
        if self._inject_via_keystrokes(text):
            logger.debug("Injected text via simulated keystrokes")
            return True

        # Fallback: clipboard paste (Cmd+V)
        if self._inject_via_clipboard(text):
            logger.debug("Injected text via clipboard paste")
            return True

        logger.warning("All injection methods failed")
        return False

    @staticmethod
    def _prefer_insert_only(app_name: str) -> bool:
        """Return True for editors where AX value rewrites are destructive."""
        return app_name == "Codex"

    def _inject_via_cdp(
        self, text: str, app_name: str = "", app_pid: int = 0,
    ) -> bool:
        """Inject text via the Chrome DevTools Protocol.

        Only attempted for Chromium-based apps.  Tries Input.insertText
        first, falling back to document.execCommand('insertText').

        Returns True if injection succeeded.
        """
        if not app_name or not is_chromium_app(app_name):
            return False

        port = find_debug_port(app_pid)
        if port is None:
            logger.debug(
                f"No CDP debug port found for {app_name!r} (PID {app_pid})"
            )
            return False

        cdp = CDPConnection(port=port)
        try:
            if not cdp.connect_to_target():
                logger.debug(f"CDP: could not connect to target for {app_name!r}")
                return False

            # Primary: Input.insertText
            if cdp.insert_text(text):
                logger.debug("CDP: injected via Input.insertText")
                return True

            # Fallback: document.execCommand
            if cdp.insert_text_via_js(text):
                logger.debug("CDP: injected via execCommand")
                return True

            logger.debug("CDP: both insertion methods failed")
            return False
        except Exception:
            logger.debug("CDP injection error", exc_info=True)
            return False
        finally:
            cdp.close()

    def _inject_via_ax(
        self, text: str, insertion_point: Optional[int] = None, replace: bool = False,
    ) -> bool:
        """Inject by setting AXValue on the focused element.

        Args:
            text: The text to inject.
            insertion_point: Character offset at which to splice the text
                into the existing value. When None, text is appended to
                the end. Clamped to [0, len(current_value)] to avoid
                out-of-bounds errors.
            replace: If True, replace the entire field value with *text*,
                ignoring insertion_point.
        """
        if not HAS_INJECTION or self._system_wide is None:
            return False

        focused = ax_get_attribute(self._system_wide, "AXFocusedUIElement")
        if focused is None:
            return False

        if not ax_is_attribute_settable(focused, "AXValue"):
            return False

        current_value = ax_get_attribute(focused, "AXValue") or ""

        if replace:
            new_value = text
            cursor_after = len(text)
        elif insertion_point is not None:
            # For continuation/splice mode we need to reposition the cursor
            # after setting AXValue. If AXSelectedTextRange isn't settable
            # (common in web/Electron apps), bail out so the caller falls
            # through to CDP/clipboard which insert at the OS cursor position.
            if not ax_is_attribute_settable(focused, "AXSelectedTextRange"):
                logger.debug("AXSelectedTextRange not settable — skipping AX for continuation")
                return False
            # Clamp insertion_point to valid range
            clamped = max(0, min(insertion_point, len(current_value)))
            new_value = current_value[:clamped] + text + current_value[clamped:]
            cursor_after = clamped + len(text)
        else:
            # Legacy append behaviour
            new_value = current_value + text
            cursor_after = len(new_value)

        if ax_set_attribute(focused, "AXValue", new_value):
            # Try to trigger a confirm/change notification first — some apps
            # reset cursor position during confirm, so we set cursor AFTER.
            ApplicationServices.AXUIElementPerformAction(
                focused, "AXConfirm"
            )

            # Let the app process the confirm before repositioning the cursor
            # — some apps handle it asynchronously.
            time.sleep(0.05)
            self._set_cursor_position(focused, cursor_after)
            return True

        return False

    @staticmethod
    def _set_cursor_position(element, position: int) -> bool:
        """Move the text cursor to *position* in the focused element.

        Tries two strategies:
        1. NSValue.valueWithRange_ — most reliable with PyObjC
        2. AXValueCreate with kAXValueTypeCFRange — fallback
        """
        # Strategy 1: NSValue (preferred — well-supported in PyObjC)
        try:
            from Foundation import NSRange, NSValue
            range_value = NSValue.valueWithRange_(NSRange(position, 0))
            if ax_set_attribute(element, "AXSelectedTextRange", range_value):
                logger.debug(f"Cursor set to {position} via NSValue")
                return True
            else:
                logger.debug(f"NSValue cursor set to {position} returned False")
        except Exception:
            logger.debug("NSValue cursor set failed", exc_info=True)

        # Strategy 2: AXValueCreate (older API)
        try:
            import ApplicationServices as _AS
            range_value = _AS.AXValueCreate(
                _AS.kAXValueTypeCFRange, (position, 0),
            )
            if range_value is not None:
                if ax_set_attribute(element, "AXSelectedTextRange", range_value):
                    logger.debug(f"Cursor set to {position} via AXValueCreate")
                    return True
                else:
                    logger.debug(
                        f"AXValueCreate cursor set to {position} returned False"
                    )
            else:
                logger.debug("AXValueCreate returned None")
        except Exception:
            logger.debug("AXValueCreate cursor set failed", exc_info=True)

        logger.warning(f"Could not set cursor position to {position}")
        return False

    def _inject_via_clipboard(self, text: str) -> bool:
        """Inject by placing text on the clipboard and simulating Cmd+V.

        Note: This overwrites the user's clipboard. We intentionally do NOT
        restore the previous clipboard contents because Electron and web apps
        process paste events asynchronously — restoring too early causes the
        app to paste the old content instead of the suggestion text.
        """
        if not HAS_INJECTION:
            return False

        pasteboard = AppKit.NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)
        self._simulate_cmd_v()
        time.sleep(0.1)
        return True

    def _inject_via_keystrokes(self, text: str) -> bool:
        """Inject by simulating individual key press events."""
        if not HAS_INJECTION:
            return False

        try:
            for char in text:
                self._type_character(char)
                time.sleep(0.01)  # Small delay between characters
            return True
        except Exception:
            logger.exception("Failed to inject via keystrokes")
            return False

    @staticmethod
    def _simulate_cmd_v() -> None:
        """Simulate pressing Cmd+V."""
        if not HAS_INJECTION:
            return

        source = Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState
        )

        # Key code for 'V' is 9
        v_keycode = 9

        # Key down with Cmd modifier
        event_down = Quartz.CGEventCreateKeyboardEvent(
            source, v_keycode, True
        )
        Quartz.CGEventSetFlags(
            event_down, Quartz.kCGEventFlagMaskCommand
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)

        # Key up
        event_up = Quartz.CGEventCreateKeyboardEvent(
            source, v_keycode, False
        )
        Quartz.CGEventSetFlags(
            event_up, Quartz.kCGEventFlagMaskCommand
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)

    @staticmethod
    def _type_character(char: str) -> None:
        """Simulate typing a single character using CGEvents."""
        if not HAS_INJECTION:
            return

        source = Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState
        )

        # Use CGEventKeyboardSetUnicodeString for arbitrary characters
        event_down = Quartz.CGEventCreateKeyboardEvent(source, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(
            event_down, len(char), char
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)

        event_up = Quartz.CGEventCreateKeyboardEvent(source, 0, False)
        Quartz.CGEventKeyboardSetUnicodeString(
            event_up, len(char), char
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
