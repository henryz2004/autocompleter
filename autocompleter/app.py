"""Main application orchestrator.

Wires together the input observer, context store, suggestion engine,
overlay renderer, text injector, and hotkey listener into a cohesive
autocomplete tool.
"""

import logging
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
from .hotkey import HotkeyListener, OverlayKeyHandler
from .input_observer import InputObserver
from .overlay import OverlayConfig, SuggestionOverlay
from .suggestion_engine import Suggestion, SuggestionEngine
from .text_injector import TextInjector

# How often the background observer polls for visible content (seconds)
OBSERVE_INTERVAL = 2.0


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

        # Prune old entries on startup
        pruned = self.context_store.prune(
            max_age_hours=self.config.max_context_age_hours,
            max_entries=self.config.max_context_entries,
        )
        if pruned:
            logger.info(f"Pruned {pruned} old context entries")

        # Register hotkeys
        self.hotkey_listener.register(
            self.config.hotkey, self._on_trigger
        )

        # Register overlay navigation keys
        overlay_handler = OverlayKeyHandler(
            on_move_up=lambda: self._on_overlay_navigate(-1),
            on_move_down=lambda: self._on_overlay_navigate(1),
            on_accept=self._on_overlay_accept,
            on_dismiss=self._on_overlay_dismiss,
        )
        overlay_handler.register(self.hotkey_listener)

        # Start the hotkey listener
        self.hotkey_listener.start()

        # Start background observer thread
        self._running = True
        self._observer_thread = threading.Thread(
            target=self._observe_loop, daemon=True
        )
        self._observer_thread.start()

        logger.info(
            f"Autocompleter running. Trigger hotkey: {self.config.hotkey}"
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

        # Use a timer to allow signal handling
        timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.5, app, None, None, True
        )
        try:
            AppKit.NSRunLoop.currentRunLoop().run()
        except KeyboardInterrupt:
            self.stop()

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

    def _on_trigger(self) -> None:
        """Called when the suggestion hotkey is pressed."""
        logger.debug("Suggestion triggered")

        if self.overlay.is_visible:
            self.overlay.hide()
            return

        focused = self.observer.get_focused_element()
        if focused is None:
            logger.debug("No focused text element found")
            return

        current_input = focused.value
        if not current_input.strip():
            logger.debug("Input field is empty, skipping")
            return

        # Get context from the store
        context = self.context_store.get_sliced_context(
            source_app=focused.app_name,
            max_chars=self.config.context_window_chars,
        )

        # Generate suggestions
        suggestions = self.suggestion_engine.generate_suggestions(
            current_input=current_input,
            context=context,
            app_name=focused.app_name,
        )

        if not suggestions:
            logger.debug("No suggestions generated")
            return

        self._current_suggestions = suggestions

        # Show overlay near the focused element
        x, y = 100.0, 100.0  # Default position
        if focused.position:
            x, y = focused.position
            if focused.size:
                y += focused.size[1]  # Position below the input field

        self.overlay.show(suggestions, x, y)

    def _on_overlay_navigate(self, delta: int) -> None:
        """Handle arrow key navigation in the overlay."""
        if self.overlay.is_visible:
            self.overlay.move_selection(delta)

    def _on_overlay_accept(self) -> None:
        """Handle accepting a suggestion from the overlay."""
        if not self.overlay.is_visible:
            return

        suggestion = self.overlay.accept_selection()
        if suggestion:
            success = self.injector.inject(suggestion.text)
            if success:
                logger.info(f"Injected suggestion: {suggestion.text[:50]}...")
                # Store the accepted suggestion as context
                focused = self.observer.get_focused_element()
                app_name = focused.app_name if focused else "Unknown"
                self.context_store.add_entry(
                    source_app=app_name,
                    content=suggestion.text,
                    entry_type="accepted_suggestion",
                )
            else:
                logger.warning("Failed to inject suggestion")

    def _on_overlay_dismiss(self) -> None:
        """Handle dismissing the overlay."""
        self.overlay.hide()


def main():
    """Entry point for the autocompleter."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    app = Autocompleter(config)

    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
