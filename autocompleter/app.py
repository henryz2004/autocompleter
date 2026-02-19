"""Main application orchestrator.

Wires together the input observer, context store, suggestion engine,
overlay renderer, text injector, and hotkey listener into a cohesive
autocomplete tool.
"""

from __future__ import annotations

import hashlib
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
from .suggestion_engine import AutocompleteMode, Suggestion, SuggestionEngine, detect_mode
from .text_injector import TextInjector

# How often the background observer polls for visible content (seconds)
OBSERVE_INTERVAL = 2.0


def _get_caret_screen_position() -> tuple[float, float, float] | None:
    """Try to get the text cursor (caret) position on screen.

    Uses the focused element's AXSelectedTextRange + AXBoundsForRange
    to get the actual caret location, which is more accurate than using
    the element's position.

    Returns:
        (x, y, caret_height) in AX screen coordinates, or None.
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
                return (rect.origin.x, rect.origin.y + rect.size.height, rect.size.height)
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
        self._last_content_hash: str = ""
        self._last_input_hash: str = ""
        self._generation_id: int = 0  # Monotonic counter; only latest generation updates overlay
        self._replace_on_inject: bool = False  # True when focused field had baked-in placeholder

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

    @staticmethod
    def _hash_content(text: str) -> str:
        """Fast content hash for dedup."""
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

    # Cap stored content to avoid DB bloat from terminal scrollback buffers
    _MAX_STORE_CHARS = 4000

    def _observe_loop(self) -> None:
        """Background loop that observes visible content and stores context."""
        while self._running:
            try:
                content = self.observer.get_visible_content()
                if content and content.text_elements:
                    combined = "\n".join(content.text_elements[:50])
                    if len(combined) > self._MAX_STORE_CHARS:
                        combined = combined[:self._MAX_STORE_CHARS]
                    if combined.strip():
                        content_hash = self._hash_content(combined)
                        if content_hash != self._last_content_hash:
                            self._last_content_hash = content_hash
                            logger.debug(
                                f"Observer: stored {len(content.text_elements)} "
                                f"elements from {content.app_name!r} "
                                f"({len(combined)} chars)"
                            )
                            self.context_store.add_entry(
                                source_app=content.app_name,
                                content=combined,
                                entry_type="visible_text",
                                source_url=content.url,
                                window_title=content.window_title,
                            )

                # Also store the current input field value (capped)
                focused = self.observer.get_focused_element()
                if focused and focused.value.strip():
                    value = focused.value
                    if len(value) > self._MAX_STORE_CHARS:
                        value = value[:self._MAX_STORE_CHARS]
                    input_hash = self._hash_content(value)
                    if input_hash != self._last_input_hash:
                        self._last_input_hash = input_hash
                        logger.debug(
                            f"Observer: stored user_input from {focused.app_name!r} "
                            f"({len(value)} chars)"
                        )
                        self.context_store.add_entry(
                            source_app=focused.app_name,
                            content=value,
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
        logger.info("--- TRIGGER ---")

        if self.overlay.is_visible:
            logger.debug("Overlay visible, hiding it")
            self._run_on_main(self.overlay.hide)
            return True

        # Gather info quickly (AX calls are fast) then dispatch heavy work
        focused = self.observer.get_focused_element()
        if focused is None:
            logger.info("No focused text element found")
            return True

        # Remember whether the field has a baked-in placeholder so the
        # injector can skip AXValue setting (which bypasses the web app's
        # placeholder-clearing JS) and use clipboard/keystrokes instead.
        self._replace_on_inject = focused.placeholder_detected

        logger.info(
            f"Focused: app={focused.app_name!r} role={focused.role} "
            f"cursor_pos={focused.insertion_point} value_len={len(focused.value)} "
            f"placeholder_detected={focused.placeholder_detected}"
        )
        logger.info(
            f"Before cursor ({len(focused.before_cursor)} chars): "
            f"{focused.before_cursor[-120:]!r}"
        )
        if focused.after_cursor.strip():
            logger.info(
                f"After cursor ({len(focused.after_cursor)} chars): "
                f"{focused.after_cursor[:120]!r}"
            )

        current_input = focused.value

        # Detect mode early so the entire pipeline knows
        mode = detect_mode(before_cursor=focused.before_cursor)
        logger.info(f"Mode: {mode.value} (before_cursor len={len(focused.before_cursor.strip())})")

        # Capture position now (while we're on the event tap thread with AX access)
        caret_pos = _get_caret_screen_position()
        caret_height = 20.0  # default fallback
        if caret_pos:
            x, y, caret_height = caret_pos
            logger.debug(f"Using caret position: ({x:.0f}, {y:.0f}), caret_h={caret_height:.0f}")
            # Sanity-check: some apps (Electron) report stale/shifted caret
            # rects. If the caret bottom is above the element bottom, prefer
            # the element bottom — it's more reliable across apps.
            if focused.position and focused.size:
                elem_bottom = focused.position[1] + focused.size[1]
                if y < elem_bottom:
                    logger.debug(
                        f"Caret bottom ({y:.0f}) < element bottom ({elem_bottom:.0f}), "
                        f"using element bottom instead"
                    )
                    y = elem_bottom
        elif focused.position:
            x, y = focused.position
            if focused.size:
                y += focused.size[1]
            logger.debug(f"Using element position: ({x:.0f}, {y:.0f})")
        else:
            x, y = 100.0, 100.0
            logger.debug("No position available, using default (100, 100)")

        # Show loading indicator immediately
        self._run_on_main(lambda: self.overlay.show(
            [Suggestion(text="Generating...", index=0)], x, y,
            caret_height=caret_height,
        ))

        # Capture visible content metadata for context assembly
        visible = self.observer.get_visible_content()
        window_title = visible.window_title if visible else ""
        source_url = visible.url if visible else ""
        visible_text_elements: list[str] = []
        conversation_turns = None
        if visible:
            visible_text_elements = visible.text_elements or []
            logger.info(
                f"[CTX] Window: {visible.window_title!r} | URL: {visible.url!r} | "
                f"text_elements: {len(visible_text_elements)} | "
                f"conversation_turns: {len(visible.conversation_turns) if visible.conversation_turns else 0}"
            )
            if visible.conversation_turns:
                conversation_turns = [
                    {"speaker": t.speaker, "text": t.text}
                    for t in visible.conversation_turns
                ]
                for i, t in enumerate(visible.conversation_turns):
                    logger.debug(f"[CTX]   Turn [{i}]: {t.speaker}: {t.text[:100]!r}")
            # Log a sample of visible text elements
            for i, elem in enumerate(visible_text_elements[:10]):
                logger.debug(f"[CTX]   Visible text [{i}]: {elem[:120]!r}")
            if len(visible_text_elements) > 10:
                logger.debug(f"[CTX]   ... and {len(visible_text_elements) - 10} more elements")

            # Store fresh observation in context store so the worker thread
            # sees up-to-date data (fixes stale context at trigger time).
            if visible.text_elements:
                combined = "\n".join(visible.text_elements[:50])
                if combined.strip():
                    content_hash = self._hash_content(combined)
                    if content_hash != self._last_content_hash:
                        self._last_content_hash = content_hash
                        self.context_store.add_entry(
                            source_app=visible.app_name,
                            content=combined,
                            entry_type="visible_text",
                            source_url=visible.url,
                            window_title=visible.window_title,
                        )
        else:
            logger.info("[CTX] No visible content captured")

        # Bump generation counter — only the latest generation updates the overlay
        self._generation_id += 1
        gen_id = self._generation_id

        # Dispatch the LLM call to a worker thread so we don't block the tap.
        # Use the streaming path by default for faster perceived response.
        threading.Thread(
            target=self._generate_and_show_streaming,
            args=(
                focused, x, y, caret_height, mode,
                window_title, source_url, conversation_turns,
                visible_text_elements, gen_id,
            ),
            daemon=True,
        ).start()

        return True

    def _generate_and_show(
        self,
        focused,  # FocusedElement (duck typed to avoid circular import)
        x: float, y: float,
        caret_height: float = 20.0,
        mode: AutocompleteMode | None = None,
        window_title: str = "",
        source_url: str = "",
        conversation_turns: list[dict[str, str]] | None = None,
        visible_text_elements: list[str] | None = None,
        generation_id: int = 0,
    ) -> None:
        """Run the LLM call on a worker thread and show the overlay."""
        app_name = focused.app_name
        current_input = focused.value

        if mode is None:
            mode = detect_mode(before_cursor=focused.before_cursor)

        # Assemble context based on mode
        if mode == AutocompleteMode.CONTINUATION:
            context = self.context_store.get_continuation_context(
                before_cursor=focused.before_cursor,
                after_cursor=focused.after_cursor,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                visible_text=visible_text_elements,
            )
        else:
            context = self.context_store.get_reply_context(
                conversation_turns=conversation_turns or [],
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=focused.before_cursor if focused.insertion_point is not None else "",
                visible_text=visible_text_elements,
            )

        logger.info(
            f"[CTX] --- CONTEXT (gen={generation_id}, mode={mode.value}, "
            f"{len(context)} chars) ---"
        )
        # Log the full assembled context so we can inspect exactly what the LLM sees
        for line in context.splitlines():
            logger.info(f"[CTX]   | {line}")
        logger.info("[CTX] --- END CONTEXT ---")

        t0 = time.time()
        suggestions = self.suggestion_engine.generate_suggestions(
            current_input=current_input,
            context=context,
            app_name=app_name,
            mode=mode,
        )
        elapsed = time.time() - t0

        # Check if a newer trigger has superseded this one
        if generation_id != self._generation_id:
            logger.info(
                f"Generation {generation_id} superseded by {self._generation_id}, "
                f"discarding {len(suggestions)} results after {elapsed:.2f}s"
            )
            return

        if not suggestions:
            logger.info(f"No suggestions generated (took {elapsed:.2f}s)")
            self._run_on_main(self.overlay.hide)
            return

        logger.info(
            f"--- SUGGESTIONS (gen={generation_id}, {elapsed:.2f}s) ---"
        )
        for i, s in enumerate(suggestions):
            logger.info(f"  [{i}]: {s.text[:120]}")

        self._current_suggestions = suggestions

        # Dispatch overlay show to main thread (re-check generation to avoid race)
        def _show():
            if generation_id == self._generation_id:
                self.overlay.show(suggestions, x, y, caret_height=caret_height)

        self._run_on_main(_show)

    def _generate_and_show_streaming(
        self,
        focused,  # FocusedElement (duck typed to avoid circular import)
        x: float, y: float,
        caret_height: float = 20.0,
        mode: AutocompleteMode | None = None,
        window_title: str = "",
        source_url: str = "",
        conversation_turns: list[dict[str, str]] | None = None,
        visible_text_elements: list[str] | None = None,
        generation_id: int = 0,
    ) -> None:
        """Run the streaming LLM call on a worker thread, updating the overlay incrementally."""
        app_name = focused.app_name
        current_input = focused.value

        if mode is None:
            mode = detect_mode(before_cursor=focused.before_cursor)

        # Assemble context based on mode
        if mode == AutocompleteMode.CONTINUATION:
            context = self.context_store.get_continuation_context(
                before_cursor=focused.before_cursor,
                after_cursor=focused.after_cursor,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                visible_text=visible_text_elements,
            )
        else:
            context = self.context_store.get_reply_context(
                conversation_turns=conversation_turns or [],
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=focused.before_cursor if focused.insertion_point is not None else "",
                visible_text=visible_text_elements,
            )

        logger.info(
            f"[CTX] --- CONTEXT (gen={generation_id}, mode={mode.value}, "
            f"{len(context)} chars) ---"
        )
        for line in context.splitlines():
            logger.info(f"[CTX]   | {line}")
        logger.info("[CTX] --- END CONTEXT ---")

        t0 = time.time()
        suggestions: list[Suggestion] = []

        try:
            for suggestion in self.suggestion_engine.generate_suggestions_stream(
                current_input=current_input,
                context=context,
                app_name=app_name,
                mode=mode,
            ):
                # Check if a newer trigger has superseded this one
                if generation_id != self._generation_id:
                    elapsed = time.time() - t0
                    logger.info(
                        f"Generation {generation_id} superseded by {self._generation_id}, "
                        f"abandoning stream after {len(suggestions)} suggestions ({elapsed:.2f}s)"
                    )
                    return

                suggestions.append(suggestion)
                logger.info(
                    f"Stream [{generation_id}]: suggestion {suggestion.index} arrived "
                    f"({time.time() - t0:.2f}s): {suggestion.text[:80]!r}"
                )

                # Capture a snapshot of suggestions for the closure
                snapshot = list(suggestions)

                def _update(snp=snapshot):
                    if generation_id == self._generation_id:
                        self.overlay.show(snp, x, y, caret_height=caret_height)

                self._run_on_main(_update)

        except Exception:
            logger.exception(f"Error during streaming generation {generation_id}")

        elapsed = time.time() - t0

        # Final check: if superseded, discard
        if generation_id != self._generation_id:
            logger.info(
                f"Generation {generation_id} superseded by {self._generation_id}, "
                f"discarding after stream completed ({elapsed:.2f}s)"
            )
            return

        if not suggestions:
            logger.info(f"No suggestions from stream (took {elapsed:.2f}s)")
            self._run_on_main(self.overlay.hide)
            return

        logger.info(
            f"--- STREAM COMPLETE (gen={generation_id}, {elapsed:.2f}s, "
            f"{len(suggestions)} suggestions) ---"
        )
        for i, s in enumerate(suggestions):
            logger.info(f"  [{i}]: {s.text[:120]}")

        self._current_suggestions = suggestions

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
                success = self.injector.inject(
                    suggestion.text, replace=self._replace_on_inject,
                )
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
    import argparse

    parser = argparse.ArgumentParser(description="macOS contextual autocompleter")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to this file (in addition to stderr)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level (default: INFO). Log file always gets DEBUG.",
    )
    args = parser.parse_args()

    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Allow everything through; handlers filter

    # Stderr: show INFO+ by default (clean output)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, args.log_level))
    console.setFormatter(logging.Formatter(log_fmt))
    root.addHandler(console)

    # File: always DEBUG (full diagnostics)
    if args.log_file:
        fh = logging.FileHandler(args.log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(log_fmt))
        root.addHandler(fh)

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    if args.log_file:
        print(f"Logging to {args.log_file} (file=DEBUG, console={args.log_level})")

    config = load_config()
    app = Autocompleter(config)

    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
