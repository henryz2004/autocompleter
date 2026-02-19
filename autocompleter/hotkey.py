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


# Callback type: returns True to suppress the event, False to pass it through
HotkeyCallback = Callable[[], bool]


class HotkeyListener:
    """Listens for global hotkey events using Quartz event taps.

    Callbacks must return True to suppress the event (prevent it from
    reaching the target app) or False to let it pass through.
    """

    def __init__(self):
        self._callbacks: dict[tuple[int, int], HotkeyCallback] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._tap = None

    def register(self, hotkey_str: str, callback: HotkeyCallback) -> None:
        """Register a callback for a hotkey combination.

        The callback must return True to suppress the key event, or False
        to let it pass through to the focused application.
        """
        keycode, flags = parse_hotkey(hotkey_str)
        self._callbacks[(keycode, flags)] = callback
        logger.info(f"Registered hotkey: {hotkey_str} (code={keycode}, flags=0x{flags:x})")

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
            # macOS sends kCGEventTapDisabledByTimeout when the tap blocks
            # too long. Re-enable it automatically.
            if event_type == Quartz.kCGEventTapDisabledByTimeout:
                logger.warning("Event tap was disabled by timeout, re-enabling")
                Quartz.CGEventTapEnable(self._tap, True)
                return event

            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                flags = Quartz.CGEventGetFlags(event)
                for (reg_keycode, reg_flags), cb in self._callbacks.items():
                    # For hotkeys with modifiers: match keycode and
                    # require the modifier flags to be present.
                    # For hotkeys without modifiers (reg_flags == 0):
                    # match keycode only when NO modifiers are held
                    # (ignore device-dependent bits like caps lock).
                    if keycode != reg_keycode:
                        continue

                    if reg_flags != 0:
                        # Modifier hotkey: check required flags are present
                        if (flags & reg_flags) != reg_flags:
                            continue
                    else:
                        # Plain key: only match if no modifier keys held
                        modifier_mask = (
                            Quartz.kCGEventFlagMaskControl
                            | Quartz.kCGEventFlagMaskCommand
                            | Quartz.kCGEventFlagMaskAlternate
                            | Quartz.kCGEventFlagMaskShift
                        )
                        if flags & modifier_mask:
                            continue

                    try:
                        suppress = cb()
                        if suppress:
                            logger.debug(
                                f"Hotkey suppressed: code={keycode} flags=0x{flags:x}"
                            )
                            return None
                        else:
                            logger.debug(
                                f"Hotkey handled but passed through: code={keycode}"
                            )
                    except Exception:
                        logger.exception("Error in hotkey callback")

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

        while self._running:
            Quartz.CFRunLoopRunInMode(
                Quartz.kCFRunLoopDefaultMode, 1.0, False
            )

        logger.info("Hotkey event tap stopped")
