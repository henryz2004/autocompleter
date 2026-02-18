"""Text Injector - inserts accepted suggestion text into focused input fields.

Handles contenteditable divs and custom input components used by chat interfaces
(ChatGPT, Claude, Gemini). Must trigger the app's internal state update,
not just visual insertion.

Strategy:
1. Primary: Use the Accessibility API's AXValue setter + AXConfirmAction
2. Fallback: Use CGEvents to simulate keyboard input character-by-character
3. Fallback: Use NSPasteboard (clipboard) + Cmd+V simulation
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


def _ax_get_attribute(element, attribute: str):
    """Safely get an accessibility attribute."""
    if not HAS_INJECTION:
        return None
    err, value = ApplicationServices.AXUIElementCopyAttributeValue(
        element, attribute, None
    )
    if err == 0:
        return value
    return None


def _ax_set_attribute(element, attribute: str, value) -> bool:
    """Safely set an accessibility attribute."""
    if not HAS_INJECTION:
        return False
    err = ApplicationServices.AXUIElementSetAttributeValue(
        element, attribute, value
    )
    return err == 0


def _ax_is_attribute_settable(element, attribute: str) -> bool:
    """Check if an accessibility attribute is settable."""
    if not HAS_INJECTION:
        return False
    err, settable = ApplicationServices.AXUIElementIsAttributeSettable(
        element, attribute, None
    )
    return err == 0 and settable


class TextInjector:
    """Injects text into the currently focused input field."""

    def __init__(self):
        if HAS_INJECTION:
            self._system_wide = AXUIElementCreateSystemWide()
        else:
            self._system_wide = None

    def inject(self, text: str) -> bool:
        """Inject text into the currently focused input.

        Tries multiple strategies in order of preference.
        Returns True if injection succeeded.
        """
        if not text:
            return False

        # Try AX API value setting first
        if self._inject_via_ax(text):
            logger.debug("Injected text via AX API")
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

    def _inject_via_ax(self, text: str) -> bool:
        """Inject by setting AXValue on the focused element."""
        if not HAS_INJECTION or self._system_wide is None:
            return False

        focused = _ax_get_attribute(self._system_wide, "AXFocusedUIElement")
        if focused is None:
            return False

        if not _ax_is_attribute_settable(focused, "AXValue"):
            return False

        current_value = _ax_get_attribute(focused, "AXValue") or ""
        new_value = current_value + text

        if _ax_set_attribute(focused, "AXValue", new_value):
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

        # Set our text
        pasteboard.clearContents()
        pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)

        # Simulate Cmd+V
        self._simulate_cmd_v()
        time.sleep(0.1)  # Brief pause for the paste to register

        # Restore clipboard
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
