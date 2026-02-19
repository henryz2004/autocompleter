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

import logging
import time

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
        self, text: str, replace: bool = False,
        app_name: str = "", app_pid: int = 0,
    ) -> bool:
        """Inject text into the currently focused input.

        Tries multiple strategies in order of preference.
        When *replace* is True (field has a baked-in placeholder), skip
        AXValue setting — it bypasses the web app's JS event handlers so
        the PWA/Electron app never clears its placeholder.  Clipboard
        paste and keystrokes go through normal input handling, which
        triggers the placeholder-clearing behaviour.

        *app_name* and *app_pid* are used to determine whether CDP
        injection should be attempted (for Chromium-based apps).

        Returns True if injection succeeded.
        """
        if not text:
            return False

        # Try AX API value setting first (skip when replacing placeholder)
        if not replace:
            if self._inject_via_ax(text):
                logger.debug("Injected text via AX API")
                return True

        # Try CDP injection for Chromium-based apps
        if self._inject_via_cdp(text, app_name=app_name, app_pid=app_pid):
            logger.debug("Injected text via CDP")
            return True

        # Fall back to clipboard paste
        if self._inject_via_clipboard(text):
            logger.debug("Injected text via clipboard paste")
            return True

        # Last resort: simulated keystrokes
        if self._inject_via_keystrokes(text):
            logger.debug("Injected text via simulated keystrokes")
            return True

        logger.warning("All injection methods failed")
        return False

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

    def _inject_via_ax(self, text: str) -> bool:
        """Inject by setting AXValue on the focused element."""
        if not HAS_INJECTION or self._system_wide is None:
            return False

        focused = ax_get_attribute(self._system_wide, "AXFocusedUIElement")
        if focused is None:
            return False

        if not ax_is_attribute_settable(focused, "AXValue"):
            return False

        current_value = ax_get_attribute(focused, "AXValue") or ""
        new_value = current_value + text

        if ax_set_attribute(focused, "AXValue", new_value):
            # Try to trigger a confirm/change notification
            ApplicationServices.AXUIElementPerformAction(
                focused, "AXConfirm"
            )
            return True

        return False

    def _inject_via_clipboard(self, text: str) -> bool:
        """Inject by placing text on the clipboard and simulating Cmd+V."""
        if not HAS_INJECTION:
            return False

        pasteboard = AppKit.NSPasteboard.generalPasteboard()

        # Save current clipboard content
        old_types = pasteboard.types()
        old_content = None
        if (
            old_types
            and AppKit.NSPasteboardTypeString in old_types
        ):
            old_content = pasteboard.stringForType_(
                AppKit.NSPasteboardTypeString
            )

        # Set our text, paste, and always restore the clipboard
        try:
            pasteboard.clearContents()
            pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)
            self._simulate_cmd_v()
            time.sleep(0.1)  # Brief pause for the paste to register
        finally:
            if old_content is not None:
                time.sleep(0.05)
                pasteboard.clearContents()
                pasteboard.setString_forType_(
                    old_content, AppKit.NSPasteboardTypeString
                )

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
