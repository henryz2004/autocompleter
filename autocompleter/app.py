"""Main application orchestrator.

Wires together the input observer, context store, suggestion engine,
overlay renderer, text injector, and hotkey listener into a cohesive
autocomplete tool.
"""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time

logger = logging.getLogger(__name__)

try:
    import AppKit

    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False

from .config import Config, load_config
from .context_store import ContextStore
from .hotkey import HotkeyListener
from .input_observer import InputObserver
from .overlay import OverlayConfig, SuggestionOverlay
from .suggestion_engine import Suggestion, SuggestionEngine
from .text_injector import TextInjector

# How often the background observer polls for visible content (seconds)
OBSERVE_INTERVAL = 2.0


def _get_caret_screen_position() -> tuple[float, float] | None:
    """Try to get the text cursor (caret) position on screen.

    Uses the focused element's AXSelectedTextRange + AXBoundsForRange
    to get the actual caret location, which is more accurate than using
    the element's position.
    """
    if not HAS_APPKIT:
        return None
    try:
        import ApplicationServices
        from ApplicationServices import (
            AXUIElementCreateSystemWide,
            AXValueGetValue,
            kAXValueTypeCGRect,
        )

        system_wide = AXUIElementCreateSystemWide()
        err, focused = ApplicationServices.AXUIElementCopyAttributeValue(
            system_wide, "AXFocusedUIElement", None
        )
        if err != 0 or focused is None:
            logger.debug("Caret: no focused element")
            return None

        # Get the selected text range (caret position)
        err, sel_range = ApplicationServices.AXUIElementCopyAttributeValue(
            focused, "AXSelectedTextRange", None
        )
        if err != 0 or sel_range is None:
            logger.debug(f"Caret: no AXSelectedTextRange (err={err})")
            return None

        # Get the bounds for that range
        err, bounds = ApplicationServices.AXUIElementCopyParameterizedAttributeValue(
            focused, "AXBoundsForRange", sel_range, None
        )
        if err != 0 or bounds is None:
            logger.debug(f"Caret: no AXBoundsForRange (err={err})")
            return None

        success, rect = AXValueGetValue(bounds, kAXValueTypeCGRect, None)
        if success:
            logger.debug(
                f"Caret rect: x={rect.origin.x:.0f} y={rect.origin.y:.0f} "
                f"w={rect.size.width:.0f} h={rect.size.height:.0f}"
            )
            # Validate: reject degenerate rects (zero size or negative coords)
            if rect.size.height > 0 and rect.origin.x > 0 and rect.origin.y >= 0:
                return (rect.origin.x, rect.origin.y + rect.size.height)
            else:
                logger.debug("Caret rect is degenerate, ignoring")
        else:
            logger.debug("Caret: AXValueGetValue failed")
    except Exception:
        logger.debug("Could not get caret position via AXBoundsForRange", exc_info=True)

    return None


class Autocompleter:
    """Main application class that orchestrates all components."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()

        self.context_store = ContextStore(self.config.db_path)
        self.observer = InputObserver()
        self.suggestion_engine = SuggestionEngine(self.config)
        self.overlay = SuggestionOverlay(
            OverlayConfig(
                width=self.config.overlay_width,
                max_height=self.config.overlay_max_height,
                font_size=self.config.overlay_font_size,
                opacity=self.config.overlay_opacity,
            )
        )
        self.injector = TextInjector()
        self.hotkey_listener = HotkeyListener()

        self._running = False
        self._observer_thread: threading.Thread | None = None
        self._current_suggestions: list[Suggestion] = []
        self._main_queue: queue.Queue = queue.Queue()

    def start(self) -> None:
        """Start the autocompleter."""
        logger.info("Starting autocompleter...")

        # Check accessibility permissions
        if not self.observer.check_accessibility_permissions():
            logger.error(
                "Accessibility permissions not granted. "
                "Go to System Preferences > Security & Privacy > "
                "Privacy > Accessibility and add this application."
            )
            print(
                "ERROR: Accessibility permissions required.\n"
                "Go to System Preferences > Security & Privacy > "
                "Privacy > Accessibility\n"
                "and grant access to this application (or Terminal).",
                file=sys.stderr,
            )
            sys.exit(1)

        # Open the context store
        self.context_store.open()
        logger.info(f"Context store opened at {self.config.db_path}")
        logger.info(f"Context store has {self.context_store.entry_count()} entries")

        # Prune old entries on startup
        pruned = self.context_store.prune(
            max_age_hours=self.config.max_context_age_hours,
            max_entries=self.config.max_context_entries,
        )
        if pruned:
            logger.info(f"Pruned {pruned} old context entries")

        # Register hotkeys — callbacks return True to suppress, False to pass through
        self.hotkey_listener.register(
            self.config.hotkey, self._on_trigger
        )
        self.hotkey_listener.register("up", self._on_nav_up)
        self.hotkey_listener.register("down", self._on_nav_down)
        self.hotkey_listener.register("tab", self._on_nav_accept)
        self.hotkey_listener.register("return", self._on_nav_accept)
        self.hotkey_listener.register("escape", self._on_nav_dismiss)

        # Start the hotkey listener
        self.hotkey_listener.start()

        # Start background observer thread
        self._running = True
        self._observer_thread = threading.Thread(
            target=self._observe_loop, daemon=True
        )
        self._observer_thread.start()

        logger.info(
            f"Autocompleter running. Trigger: {self.config.hotkey} | "
            f"Provider: {self.config.llm_provider} | Model: {self.config.llm_model}"
        )
        print(f"Autocompleter running. Press {self.config.hotkey} to get suggestions.")
        print("Press Ctrl+C to exit.")

        # Run the main application loop
        if HAS_APPKIT:
            self._run_appkit_loop()
        else:
            self._run_simple_loop()

    def stop(self) -> None:
        """Stop the autocompleter."""
        logger.info("Stopping autocompleter...")
        self._running = False
        self.hotkey_listener.stop()
        self.overlay.hide()
        self.context_store.close()

    def _run_appkit_loop(self) -> None:
        """Run the AppKit event loop (needed for overlay rendering)."""
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

        # Handle Ctrl+C
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Run the loop in short intervals, draining the main queue each tick
        try:
            while self._running:
                self._drain_main_queue()
                AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.05)
                )
        except KeyboardInterrupt:
            self.stop()

    def _run_on_main(self, fn) -> None:
        """Schedule a function to run on the main thread."""
        self._main_queue.put(fn)

    def _drain_main_queue(self) -> None:
        """Execute all pending functions on the main thread."""
        while not self._main_queue.empty():
            try:
                fn = self._main_queue.get_nowait()
                fn()
            except Exception:
                logger.exception("Error in main-thread callback")

    def _run_simple_loop(self) -> None:
        """Simple fallback loop when AppKit is not available."""
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def _observe_loop(self) -> None:
        """Background loop that observes visible content and stores context."""
        while self._running:
            try:
                content = self.observer.get_visible_content()
                if content and content.text_elements:
                    combined = "\n".join(content.text_elements[:20])
                    if combined.strip():
                        self.context_store.add_entry(
                            source_app=content.app_name,
                            content=combined,
                            entry_type="visible_text",
                            source_url=content.url,
                        )

                # Also store the current input field value
                focused = self.observer.get_focused_element()
                if focused and focused.value.strip():
                    self.context_store.add_entry(
                        source_app=focused.app_name,
                        content=focused.value,
                        entry_type="user_input",
                    )

            except Exception:
                logger.exception("Error in observer loop")

            time.sleep(OBSERVE_INTERVAL)

    # ---- Hotkey callbacks (run on background event-tap thread) ----
    # Each returns True to suppress the key event, False to pass it through.

    def _on_trigger(self) -> bool:
        """Called when the suggestion hotkey is pressed.

        IMPORTANT: This runs on the event tap thread. It must return
        immediately — macOS will disable the tap if it blocks for >1s.
        The actual LLM call is dispatched to a worker thread.
        """
        logger.debug("Trigger hotkey pressed")

        if self.overlay.is_visible:
            logger.debug("Overlay visible, hiding it")
            self._run_on_main(self.overlay.hide)
            return True

        # Gather info quickly (AX calls are fast) then dispatch heavy work
        focused = self.observer.get_focused_element()
        if focused is None:
            logger.debug("No focused text element found")
            return True

        logger.debug(
            f"Focused element: app={focused.app_name} role={focused.role} "
            f"pos={focused.position} size={focused.size} "
            f"value_len={len(focused.value)}"
        )

        current_input = focused.value
        if not current_input.strip():
            logger.debug("Input field is empty, skipping")
            return True

        # Capture position now (while we're on the event tap thread with AX access)
        caret_pos = _get_caret_screen_position()
        if caret_pos:
            x, y = caret_pos
            logger.debug(f"Using caret position: ({x:.0f}, {y:.0f})")
        elif focused.position:
            x, y = focused.position
            if focused.size:
                y += focused.size[1]
            logger.debug(f"Using element position: ({x:.0f}, {y:.0f})")
        else:
            x, y = 100.0, 100.0
            logger.debug("No position available, using default (100, 100)")

        # Dispatch the LLM call to a worker thread so we don't block the tap
        threading.Thread(
            target=self._generate_and_show,
            args=(current_input, focused.app_name, x, y),
            daemon=True,
        ).start()

        return True

    def _generate_and_show(
        self, current_input: str, app_name: str, x: float, y: float
    ) -> None:
        """Run the LLM call on a worker thread and show the overlay."""
        context = self.context_store.get_sliced_context(
            source_app=app_name,
            max_chars=self.config.context_window_chars,
        )
        logger.debug(f"Context length: {len(context)} chars")

        t0 = time.time()
        suggestions = self.suggestion_engine.generate_suggestions(
            current_input=current_input,
            context=context,
            app_name=app_name,
        )
        elapsed = time.time() - t0
        logger.info(f"LLM call took {elapsed:.2f}s, got {len(suggestions)} suggestions")

        if not suggestions:
            logger.debug("No suggestions generated")
            return

        for i, s in enumerate(suggestions):
            logger.debug(f"  Suggestion [{i}]: {s.text[:80]}")

        self._current_suggestions = suggestions

        # Dispatch overlay show to main thread
        self._run_on_main(lambda: self.overlay.show(suggestions, x, y))

    def _on_nav_up(self) -> bool:
        """Handle up arrow — only intercept when overlay is visible."""
        if not self.overlay.is_visible:
            return False
        self._run_on_main(lambda: self.overlay.move_selection(-1))
        return True

    def _on_nav_down(self) -> bool:
        """Handle down arrow — only intercept when overlay is visible."""
        if not self.overlay.is_visible:
            return False
        self._run_on_main(lambda: self.overlay.move_selection(1))
        return True

    def _on_nav_accept(self) -> bool:
        """Handle tab/enter — only intercept when overlay is visible."""
        if not self.overlay.is_visible:
            return False

        def _accept():
            suggestion = self.overlay.accept_selection()
            if suggestion:
                success = self.injector.inject(suggestion.text)
                if success:
                    logger.info(f"Injected: {suggestion.text[:60]}")
                    focused = self.observer.get_focused_element()
                    app_name = focused.app_name if focused else "Unknown"
                    self.context_store.add_entry(
                        source_app=app_name,
                        content=suggestion.text,
                        entry_type="accepted_suggestion",
                    )
                else:
                    logger.warning("Failed to inject suggestion")

        self._run_on_main(_accept)
        return True

    def _on_nav_dismiss(self) -> bool:
        """Handle escape — only intercept when overlay is visible."""
        if not self.overlay.is_visible:
            return False
        self._run_on_main(self.overlay.hide)
        return True


def main():
    """Entry point for the autocompleter."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet down noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    config = load_config()
    app = Autocompleter(config)

    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
