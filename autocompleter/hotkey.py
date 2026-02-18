"""Hotkey listener for triggering suggestions.

Uses Quartz event taps to listen for global keyboard shortcuts without
requiring the app to be focused. Runs the event tap on a background thread
and dispatches callbacks to the main thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

try:
    import Quartz
    from AppKit import NSEvent

    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False

# Key code constants
KEY_CODES = {
    "space": 49,
    "tab": 48,
    "return": 36,
    "escape": 53,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
}

# Modifier flag masks
MODIFIER_FLAGS = {
    "ctrl": Quartz.kCGEventFlagMaskControl if HAS_QUARTZ else 0,
    "cmd": Quartz.kCGEventFlagMaskCommand if HAS_QUARTZ else 0,
    "alt": Quartz.kCGEventFlagMaskAlternate if HAS_QUARTZ else 0,
    "shift": Quartz.kCGEventFlagMaskShift if HAS_QUARTZ else 0,
}


def parse_hotkey(hotkey_str: str) -> tuple[int, int]:
    """Parse a hotkey string like 'ctrl+space' into (keycode, modifier_flags).

    Returns:
        Tuple of (keycode, combined_modifier_flags).
    """
    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    keycode = 0
    flags = 0

    for part in parts:
        if part in MODIFIER_FLAGS:
            flags |= MODIFIER_FLAGS[part]
        elif part in KEY_CODES:
            keycode = KEY_CODES[part]
        else:
            # Try to interpret as a single character
            if len(part) == 1:
                # Map common characters to key codes
                char_codes = {
                    "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3,
                    "g": 5, "h": 4, "i": 34, "j": 38, "k": 40, "l": 37,
                    "m": 46, "n": 45, "o": 31, "p": 35, "q": 12, "r": 15,
                    "s": 1, "t": 17, "u": 32, "v": 9, "w": 13, "x": 7,
                    "y": 16, "z": 6,
                }
                keycode = char_codes.get(part, 0)
            else:
                logger.warning(f"Unknown hotkey component: {part}")

    return keycode, flags


class HotkeyListener:
    """Listens for global hotkey events using Quartz event taps."""

    def __init__(self):
        self._callbacks: dict[tuple[int, int], Callable] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._tap = None

    def register(self, hotkey_str: str, callback: Callable) -> None:
        """Register a callback for a hotkey combination."""
        keycode, flags = parse_hotkey(hotkey_str)
        self._callbacks[(keycode, flags)] = callback
        logger.info(f"Registered hotkey: {hotkey_str} (code={keycode}, flags={flags})")

    def start(self) -> None:
        """Start listening for hotkey events in a background thread."""
        if not HAS_QUARTZ:
            logger.warning("Quartz not available; hotkey listener disabled.")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        self._running = False
        if self._tap is not None:
            Quartz.CGEventTapEnable(self._tap, False)

    def _run_event_loop(self) -> None:
        """Run the Quartz event tap loop."""
        mask = (
            1 << Quartz.kCGEventKeyDown
            | 1 << Quartz.kCGEventKeyUp
        )

        def callback(proxy, event_type, event, refcon):
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                flags = Quartz.CGEventGetFlags(event)

                # Check each registered hotkey
                for (reg_keycode, reg_flags), cb in self._callbacks.items():
                    if keycode == reg_keycode and (flags & reg_flags) == reg_flags:
                        try:
                            cb()
                        except Exception:
                            logger.exception("Error in hotkey callback")
                        # Suppress the event so it doesn't reach the app
                        return None

            return event

        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            mask,
            callback,
            None,
        )

        if self._tap is None:
            logger.error(
                "Failed to create event tap. "
                "Ensure the app has accessibility permissions."
            )
            return

        run_loop_source = Quartz.CFMachPortCreateRunLoopSource(
            None, self._tap, 0
        )
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(),
            run_loop_source,
            Quartz.kCFRunLoopCommonModes,
        )
        Quartz.CGEventTapEnable(self._tap, True)

        logger.info("Hotkey event tap started")

        # Run the event loop
        while self._running:
            Quartz.CFRunLoopRunInMode(
                Quartz.kCFRunLoopDefaultMode, 1.0, False
            )

        logger.info("Hotkey event tap stopped")


class OverlayKeyHandler:
    """Handles keyboard events while the suggestion overlay is visible.

    Arrow keys navigate, Tab/Enter accept, Esc dismisses.
    """

    def __init__(
        self,
        on_move_up: Callable,
        on_move_down: Callable,
        on_accept: Callable,
        on_dismiss: Callable,
    ):
        self.on_move_up = on_move_up
        self.on_move_down = on_move_down
        self.on_accept = on_accept
        self.on_dismiss = on_dismiss

    def register(self, listener: HotkeyListener) -> None:
        """Register overlay navigation keys."""
        # These are registered as additional hotkeys. The main app
        # should only enable them when the overlay is visible.
        listener.register("up", self.on_move_up)
        listener.register("down", self.on_move_down)
        listener.register("tab", self.on_accept)
        listener.register("return", self.on_accept)
        listener.register("escape", self.on_dismiss)
