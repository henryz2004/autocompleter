"""Text Injector - inserts accepted suggestion text into focused input fields.

Handles contenteditable divs and custom input components used by chat interfaces
(ChatGPT, Claude, Gemini). Must trigger the app's internal state update,
not just visual insertion.

Strategy:
1. Primary: Use the Accessibility API's AXValue setter + AXConfirmAction
2. Fallback: Use CGEvents to simulate keyboard input character-by-character
3. Fallback: Use NSPasteboard (clipboard) + Cmd+V simulation
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


class TextInjector:
    """Injects text into the currently focused input field."""

    def __init__(self):
        if HAS_INJECTION:
            self._system_wide = AXUIElementCreateSystemWide()
        else:
            self._system_wide = None

    def inject(
        self, text: str, replace: bool = False, insertion_point: Optional[int] = None,
    ) -> bool:
        """Inject text into the currently focused input.

        Tries multiple strategies in order of preference.
        When *replace* is True (field has a baked-in placeholder), skip
        AXValue setting — it bypasses the web app's JS event handlers so
        the PWA/Electron app never clears its placeholder.  Clipboard
        paste and keystrokes go through normal input handling, which
        triggers the placeholder-clearing behaviour.

        Args:
            text: The text to inject.
            replace: If True, skip AXValue setting and use clipboard/keystrokes.
            insertion_point: Character offset at which to splice the text.
                When None, text is appended to the end (backward-compatible).
                Only used by the AX API strategy; clipboard paste and
                keystrokes naturally inject at the OS cursor position.

        Returns True if injection succeeded.
        """
        if not text:
            return False

        # Try AX API value setting first (skip when replacing placeholder)
        if not replace:
            if self._inject_via_ax(text, insertion_point=insertion_point):
                logger.debug("Injected text via AX API")
                return True

        # Fall back to clipboard paste.
        # NOTE: Clipboard paste simulates Cmd+V, which inserts at the
        # current OS cursor position. The insertion_point parameter is
        # not needed here — the OS handles cursor-aware insertion natively.
        if self._inject_via_clipboard(text):
            logger.debug("Injected text via clipboard paste")
            return True

        # Last resort: simulated keystrokes.
        # NOTE: Like clipboard paste, simulated keystrokes are typed at
        # the current OS cursor position, so insertion_point is not needed.
        if self._inject_via_keystrokes(text):
            logger.debug("Injected text via simulated keystrokes")
            return True

        logger.warning("All injection methods failed")
        return False

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
            # Clamp insertion_point to valid range
            clamped = max(0, min(insertion_point, len(current_value)))
            new_value = current_value[:clamped] + text + current_value[clamped:]
            cursor_after = clamped + len(text)
        else:
            # Legacy append behaviour
            new_value = current_value + text
            cursor_after = len(new_value)

        if ax_set_attribute(focused, "AXValue", new_value):
            # Move the cursor to just after the injected text so the user
            # can continue typing seamlessly.
            try:
                import ApplicationServices as _AS
                range_value = _AS.AXValueCreate(
                    _AS.kAXValueTypeCFRange, (cursor_after, 0),
                )
                ax_set_attribute(focused, "AXSelectedTextRange", range_value)
            except Exception:
                logger.debug(
                    "Could not set AXSelectedTextRange after injection",
                    exc_info=True,
                )

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
