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
from .context_trail import ContextTrail
from .embeddings import (
    AnthropicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    TFIDFEmbeddingProvider,
)
from .hotkey import HotkeyListener
from .input_observer import InputObserver, VisibleContent
from .overlay import OverlayConfig, SuggestionOverlay
from .suggestion_engine import AutocompleteMode, Suggestion, SuggestionEngine, detect_mode
from .text_injector import TextInjector


class _Debouncer:
    """Single-thread debouncer. poke() resets the deadline; fires callback after delay."""

    def __init__(self, delay_s: float, callback):
        self._delay = delay_s
        self._callback = callback
        self._deadline: float = 0.0       # monotonic timestamp; 0 = disarmed
        self._event = threading.Event()    # wakes the thread on poke/stop
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def poke(self) -> None:
        """Called on each keystroke — resets the deadline."""
        self._deadline = time.monotonic() + self._delay
        self._event.set()

    def cancel(self) -> None:
        """Cancel pending trigger without stopping the thread."""
        self._deadline = 0.0

    def stop(self) -> None:
        """Permanently stop the debouncer thread."""
        self._stopped = True
        self._deadline = 0.0
        self._event.set()

    def _loop(self) -> None:
        while not self._stopped:
            self._event.wait()          # block until poke() or stop()
            self._event.clear()
            while not self._stopped and self._deadline > 0:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    self._deadline = 0.0
                    self._callback()
                    break
                # Sleep until deadline OR a new poke resets it
                self._event.wait(timeout=remaining)
                self._event.clear()


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
        self.context_trail = ContextTrail()

        # Initialize embedding provider for semantic context
        self._embedding_provider = None
        if self.config.use_semantic_context:
            provider_name = self.config.embedding_provider
            if provider_name == "anthropic":
                self._embedding_provider = AnthropicEmbeddingProvider(
                    api_key=self.config.anthropic_api_key,
                )
            elif provider_name == "openai":
                self._embedding_provider = OpenAIEmbeddingProvider(
                    api_key=self.config.openai_api_key,
                )
            else:
                self._embedding_provider = TFIDFEmbeddingProvider()
            logger.info(
                f"Semantic context enabled with {provider_name} embeddings"
            )

        self._running = False
        self._observer_thread: threading.Thread | None = None
        self._current_suggestions: list[Suggestion] = []
        self._main_queue: queue.Queue = queue.Queue()
        self._last_content_hash: str = ""
        self._last_input_hash: str = ""
        self._generation_id: int = 0  # Monotonic counter; only latest generation updates overlay
        self._replace_on_inject: bool = False  # True when focused field had baked-in placeholder
        self._trigger_time: float | None = None  # Timestamp of last trigger for latency tracking
        self._trigger_mode: str = "continuation"  # Mode used for the current suggestions
        self._trigger_app: str = "Unknown"  # App name for the current suggestions

        # Observer loop state
        self._observe_iteration: int = 0  # Counter for periodic pruning
        self._observe_consecutive_errors: int = 0  # For exponential backoff
        self._last_visible_content: VisibleContent | None = None  # Cached for trigger reuse
        self._last_visible_content_time: float = 0.0  # Timestamp of cached content

        # Dynamic polling state
        self._current_poll_interval: float = self._POLL_MIN  # Start fast
        self._last_observed_app: str = ""
        self._last_observed_window: str = ""

        # Auto-trigger state
        self._auto_trigger_enabled: bool = self.config.auto_trigger_enabled
        self._last_dismiss_time: float = 0.0
        self._auto_trigger_debouncer = _Debouncer(
            delay_s=self.config.auto_trigger_delay_ms / 1000.0,
            callback=lambda: self._run_on_main(self._fire_auto_trigger),
        )

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

        # Wire up click-outside-to-dismiss (reuses existing dismiss logic)
        def _click_dismiss():
            self._on_nav_dismiss()
        self.overlay.set_dismiss_callback(_click_dismiss)

        # Dismiss overlay when user starts typing (any non-hotkey keydown),
        # and poke the auto-trigger debouncer.
        def _on_typing():
            if self.overlay.is_visible:
                self._on_nav_dismiss()
            if self._auto_trigger_enabled:
                cooldown_ok = (time.time() - self._last_dismiss_time) * 1000 >= self.config.auto_trigger_cooldown_ms
                if cooldown_ok:
                    self._auto_trigger_debouncer.poke()
        self.hotkey_listener.set_unhandled_key_callback(_on_typing)

        # Register hotkeys — callbacks return True to suppress, False to pass through
        self.hotkey_listener.register(
            self.config.hotkey, self._on_trigger
        )
        self.hotkey_listener.register(
            f"shift+{self.config.hotkey}", self._on_toggle_auto_trigger
        )
        self.hotkey_listener.register("up", self._on_nav_up)
        self.hotkey_listener.register("down", self._on_nav_down)
        self.hotkey_listener.register("tab", self._on_nav_accept)
        self.hotkey_listener.register("return", self._on_nav_accept)
        self.hotkey_listener.register("shift+tab", self._on_partial_accept)
        self.hotkey_listener.register("escape", self._on_nav_dismiss)

        # Start the hotkey listener
        self.hotkey_listener.start()

        # Start background observer thread
        self._running = True
        self._observer_thread = threading.Thread(
            target=self._observe_loop, daemon=True
        )
        self._observer_thread.start()

        auto_state = "ON" if self._auto_trigger_enabled else "OFF"
        logger.info(
            f"Autocompleter running. Trigger: {self.config.hotkey} | "
            f"Auto-trigger: {auto_state} | "
            f"Provider: {self.config.llm_provider} | Model: {self.config.llm_model}"
        )
        print(f"Autocompleter running. Press {self.config.hotkey} to get suggestions.")
        print(f"Press Shift+{self.config.hotkey} to toggle auto-trigger (currently {auto_state}).")
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
        self._auto_trigger_debouncer.stop()
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

        # Pump NSApplication events properly so that global event monitors
        # (e.g. the click-outside-to-dismiss monitor on the overlay) receive
        # their callbacks.  NSRunLoop.runUntilDate_ alone does not dispatch
        # NSApplication-level events like global mouse monitors.
        mask = getattr(AppKit, "NSEventMaskAny", AppKit.NSAnyEventMask)
        try:
            while self._running:
                self._drain_main_queue()
                event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                    mask,
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.05),
                    AppKit.NSDefaultRunLoopMode,
                    True,
                )
                if event is not None:
                    app.sendEvent_(event)
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

    # Prune the DB every N observer iterations
    _PRUNE_EVERY = 100

    # Error backoff: max sleep multiplier
    _MAX_ERROR_BACKOFF = 16

    # Dynamic polling bounds (seconds)
    _POLL_MIN = 0.5     # Fastest poll rate (right after a change)
    _POLL_MAX = 4.0     # Slowest poll rate (idle)
    _POLL_DECAY = 1.5   # Multiply interval by this each idle tick

    def _observe_loop(self) -> None:
        """Background loop that observes visible content and stores context."""
        while self._running:
            content_changed = False
            input_changed = False
            activity_detected = False

            try:
                content = self.observer.get_visible_content()
                if content and content.text_elements:
                    # Detect app/window switch
                    if (content.app_name != self._last_observed_app
                            or content.window_title != self._last_observed_window):
                        activity_detected = True
                    self._last_observed_app = content.app_name
                    self._last_observed_window = content.window_title

                    # Cache for trigger reuse
                    self._last_visible_content = content
                    self._last_visible_content_time = time.time()

                    self.context_trail.record(content)

                    combined = "\n".join(content.text_elements[:50])
                    if len(combined) > self._MAX_STORE_CHARS:
                        combined = combined[:self._MAX_STORE_CHARS]
                    if combined.strip():
                        content_hash = self._hash_content(
                            f"{content.app_name}\0{content.window_title}\0{content.url}\0{combined}"
                        )
                        if content_hash != self._last_content_hash:
                            content_changed = True
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
                    input_hash = self._hash_content(f"{focused.app_name}\0{value}")
                    if input_hash != self._last_input_hash:
                        input_changed = True
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

                # Periodic DB pruning
                self._observe_iteration += 1
                if self._observe_iteration % self._PRUNE_EVERY == 0:
                    pruned = self.context_store.prune(
                        max_age_hours=self.config.max_context_age_hours,
                        max_entries=self.config.max_context_entries,
                    )
                    if pruned:
                        logger.debug(f"Observer: pruned {pruned} old entries")

                # Reset error backoff on success
                self._observe_consecutive_errors = 0

            except Exception:
                self._observe_consecutive_errors += 1
                logger.exception("Error in observer loop")
                if self._observe_consecutive_errors == 5:
                    logger.warning(
                        "Observer: 5 consecutive errors — AX permissions "
                        "may have been revoked"
                    )

            # Dynamic polling: fast after changes, decay toward max when idle
            if activity_detected or content_changed or input_changed:
                self._current_poll_interval = self._POLL_MIN
            else:
                self._current_poll_interval = min(
                    self._current_poll_interval * self._POLL_DECAY,
                    self._POLL_MAX,
                )

            # Error backoff still multiplies on top
            backoff = min(
                2 ** self._observe_consecutive_errors,
                self._MAX_ERROR_BACKOFF,
            ) if self._observe_consecutive_errors > 0 else 1
            time.sleep(self._current_poll_interval * backoff)

    # ---- Hotkey callbacks (run on background event-tap thread) ----
    # Each returns True to suppress the key event, False to pass it through.

    def _on_trigger(self) -> bool:
        """Called when the suggestion hotkey is pressed.

        IMPORTANT: This runs on the event tap thread. It must return
        immediately — macOS will disable the tap if it blocks for >1s.
        The actual LLM call is dispatched to a worker thread.
        """
        self._auto_trigger_debouncer.cancel()
        logger.info("--- TRIGGER ---")
        self._trigger_time = time.time()

        if self.overlay.is_visible:
            logger.debug("Overlay visible, hiding it")
            # Bump generation_id so any in-flight LLM stream is discarded
            self._generation_id += 1
            self._run_on_main(self.overlay.hide)
            return True

        # Gather info quickly (AX calls are fast) then dispatch heavy work
        focused = self.observer.get_focused_element()
        if focused is None:
            logger.info("No focused text element found")
            return True

        # Skip non-editable elements. AXWebArea and AXGroup can be either
        # contenteditable inputs or read-only content areas. If the value is
        # empty and no placeholder was detected, it's almost certainly a
        # read-only area (e.g. conversation view) — not a text input.
        _AMBIGUOUS_ROLES = {"AXWebArea", "AXGroup"}
        if (focused.role in _AMBIGUOUS_ROLES
                and not focused.value.strip()
                and not focused.placeholder_detected):
            logger.info(
                f"Focused element role={focused.role!r} has no editable content, skipping"
            )
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
        self._trigger_mode = mode.value
        self._trigger_app = focused.app_name
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

        # Reuse cached visible content if fresh enough, otherwise re-fetch.
        # The observer loop polls dynamically, so if the cache is younger
        # than the current poll interval, an AX tree re-traversal is redundant.
        cache_age = time.time() - self._last_visible_content_time
        if (self._last_visible_content is not None
                and cache_age < self._current_poll_interval
                and self._last_visible_content.app_name == focused.app_name):
            visible = self._last_visible_content
            logger.debug(f"Using cached visible content ({cache_age:.1f}s old)")
        else:
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
                    content_hash = self._hash_content(
                        f"{visible.app_name}\0{visible.window_title}\0{visible.url}\0{combined}"
                    )
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

        # Fetch cross-app context from the trail
        cross_app_snapshots = self.context_trail.get_recent_cross_app_context(
            current_app=focused.app_name,
            max_age_seconds=60.0,
            max_entries=3,
        )
        cross_app_context = ContextTrail.format_cross_app_context(cross_app_snapshots)
        if cross_app_context:
            logger.info(
                f"[CTX] Cross-app context ({len(cross_app_snapshots)} snapshots):\n"
                + cross_app_context
            )

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
                visible_text_elements, gen_id, cross_app_context,
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
        cross_app_context: str = "",
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
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context,
                cross_app_context=cross_app_context,
            )
        else:
            context = self.context_store.get_reply_context(
                conversation_turns=conversation_turns or [],
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=focused.before_cursor if focused.insertion_point is not None else "",
                visible_text=visible_text_elements,
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context,
                cross_app_context=cross_app_context,
            )

        logger.info(
            f"[CTX] --- CONTEXT (gen={generation_id}, mode={mode.value}, "
            f"{len(context)} chars) ---"
        )
        # Log the full assembled context so we can inspect exactly what the LLM sees
        for line in context.splitlines():
            logger.info(f"[CTX]   | {line}")
        logger.info("[CTX] --- END CONTEXT ---")

        # Fetch feedback stats and dismissed patterns for suggestion tuning
        try:
            feedback_stats = self.context_store.get_feedback_stats(
                source_app=app_name,
            )
            negative_patterns = self.context_store.get_recent_dismissed_patterns(
                source_app=app_name,
            )
        except Exception:
            logger.debug("Could not fetch feedback data", exc_info=True)
            feedback_stats = None
            negative_patterns = None

        t0 = time.time()
        suggestions = self.suggestion_engine.generate_suggestions(
            current_input=current_input,
            context=context,
            app_name=app_name,
            mode=mode,
            feedback_stats=feedback_stats,
            negative_patterns=negative_patterns,
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
        cross_app_context: str = "",
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
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context,
                cross_app_context=cross_app_context,
            )
        else:
            context = self.context_store.get_reply_context(
                conversation_turns=conversation_turns or [],
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=focused.before_cursor if focused.insertion_point is not None else "",
                visible_text=visible_text_elements,
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context,
                cross_app_context=cross_app_context,
            )

        logger.info(
            f"[CTX] --- CONTEXT (gen={generation_id}, mode={mode.value}, "
            f"{len(context)} chars) ---"
        )
        for line in context.splitlines():
            logger.info(f"[CTX]   | {line}")
        logger.info("[CTX] --- END CONTEXT ---")

        # Fetch feedback stats and dismissed patterns for suggestion tuning
        try:
            feedback_stats = self.context_store.get_feedback_stats(
                source_app=app_name,
            )
            negative_patterns = self.context_store.get_recent_dismissed_patterns(
                source_app=app_name,
            )
        except Exception:
            logger.debug("Could not fetch feedback data", exc_info=True)
            feedback_stats = None
            negative_patterns = None

        t0 = time.time()
        suggestions: list[Suggestion] = []

        try:
            for suggestion in self.suggestion_engine.generate_suggestions_stream(
                current_input=current_input,
                context=context,
                app_name=app_name,
                mode=mode,
                feedback_stats=feedback_stats,
                negative_patterns=negative_patterns,
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

    def _fire_auto_trigger(self) -> None:
        """Fire an auto-trigger suggestion (runs on main thread)."""
        if not self._auto_trigger_enabled:
            return
        if self.overlay.is_visible:
            return
        # Recheck cooldown (dismiss may have happened while timer was sleeping)
        cooldown_ms = (time.time() - self._last_dismiss_time) * 1000
        if cooldown_ms < self.config.auto_trigger_cooldown_ms:
            return

        logger.info("--- AUTO-TRIGGER ---")
        focused = self.observer.get_focused_element()
        if focused is None:
            logger.debug("Auto-trigger: no focused element")
            return

        _AMBIGUOUS_ROLES = {"AXWebArea", "AXGroup"}
        if (focused.role in _AMBIGUOUS_ROLES
                and not focused.value.strip()
                and not focused.placeholder_detected):
            logger.debug("Auto-trigger: non-editable element, skipping")
            return

        # Need some text to complete
        if not focused.before_cursor.strip():
            logger.debug("Auto-trigger: empty input, skipping")
            return

        self._replace_on_inject = focused.placeholder_detected
        self._trigger_time = time.time()

        mode = detect_mode(before_cursor=focused.before_cursor)
        self._trigger_mode = mode.value
        self._trigger_app = focused.app_name

        caret_pos = _get_caret_screen_position()
        caret_height = 20.0
        if caret_pos:
            x, y, caret_height = caret_pos
            if focused.position and focused.size:
                elem_bottom = focused.position[1] + focused.size[1]
                if y < elem_bottom:
                    y = elem_bottom
        elif focused.position:
            x, y = focused.position
            if focused.size:
                y += focused.size[1]
        else:
            x, y = 100.0, 100.0

        # Show loading indicator
        self.overlay.show(
            [Suggestion(text="Generating...", index=0)], x, y,
            caret_height=caret_height,
        )

        # Reuse cached visible content if fresh enough
        cache_age = time.time() - self._last_visible_content_time
        if (self._last_visible_content is not None
                and cache_age < self._current_poll_interval
                and self._last_visible_content.app_name == focused.app_name):
            visible = self._last_visible_content
        else:
            visible = self.observer.get_visible_content()
        window_title = visible.window_title if visible else ""
        source_url = visible.url if visible else ""
        visible_text_elements: list[str] = visible.text_elements if visible else []
        conversation_turns = None
        if visible and visible.conversation_turns:
            conversation_turns = [
                {"speaker": t.speaker, "text": t.text}
                for t in visible.conversation_turns
            ]

        cross_app_snapshots = self.context_trail.get_recent_cross_app_context(
            current_app=focused.app_name,
            max_age_seconds=60.0,
            max_entries=3,
        )
        cross_app_context = ContextTrail.format_cross_app_context(cross_app_snapshots)

        self._generation_id += 1
        gen_id = self._generation_id

        threading.Thread(
            target=self._generate_and_show_streaming,
            args=(
                focused, x, y, caret_height, mode,
                window_title, source_url, conversation_turns,
                visible_text_elements, gen_id, cross_app_context,
            ),
            daemon=True,
        ).start()

    def _on_toggle_auto_trigger(self) -> bool:
        """Toggle auto-trigger mode on/off."""
        self._auto_trigger_enabled = not self._auto_trigger_enabled
        if not self._auto_trigger_enabled:
            self._auto_trigger_debouncer.cancel()
        state = "ON" if self._auto_trigger_enabled else "OFF"
        logger.info(f"Auto-trigger toggled: {state}")
        self._run_on_main(lambda: self._show_auto_toggle_feedback(state))
        return True

    def _show_auto_toggle_feedback(self, state: str) -> None:
        """Briefly show auto-trigger toggle status near the caret."""
        self.overlay.set_auto_trigger_active(self._auto_trigger_enabled)
        feedback = Suggestion(text=f"Auto-trigger: {state}", index=0)
        # Position near the caret
        caret_pos = _get_caret_screen_position()
        caret_height = 20.0
        if caret_pos:
            x, y, caret_height = caret_pos
        else:
            focused = self.observer.get_focused_element()
            if focused and focused.position:
                x, y = focused.position
                if focused.size:
                    y += focused.size[1]
            else:
                x, y = 200.0, 200.0
        # Bump generation so the feedback won't be overwritten by stale LLM results
        self._generation_id += 1
        feedback_gen = self._generation_id
        self.overlay.show([feedback], x, y, caret_height=caret_height)
        # Schedule hide after 1 second, but only if nothing else has shown since
        import Foundation
        Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1.0, False,
            lambda timer: self.overlay.hide() if self._generation_id == feedback_gen else None,
        )

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
            # Capture the focused element once BEFORE injection so we know
            # the current cursor position, app name, and PID. Reading it
            # twice risks the user switching apps between reads.
            focused = self.observer.get_focused_element()
            cursor_pos = focused.insertion_point if focused else None
            app_name = focused.app_name if focused else "Unknown"
            app_pid = focused.app_pid if focused else 0

            suggestion = self.overlay.accept_selection()
            if suggestion:
                success = self.injector.inject(
                    suggestion.text,
                    replace=self._replace_on_inject,
                    insertion_point=cursor_pos,
                    app_name=app_name,
                    app_pid=app_pid,
                )
                if success:
                    logger.info(f"Injected: {suggestion.text[:60]}")
                    self.context_store.add_entry(
                        source_app=app_name,
                        content=suggestion.text,
                        entry_type="accepted_suggestion",
                    )
                    # Record accepted feedback
                    latency_ms = None
                    if self._trigger_time is not None:
                        latency_ms = (time.time() - self._trigger_time) * 1000
                    try:
                        self.context_store.record_feedback(
                            source_app=self._trigger_app,
                            mode=self._trigger_mode,
                            suggestion_text=suggestion.text,
                            action="accepted",
                            suggestion_index=suggestion.index,
                            total_suggestions=len(self._current_suggestions),
                            latency_ms=latency_ms,
                        )
                    except Exception:
                        logger.debug("Failed to record accepted feedback", exc_info=True)
                else:
                    logger.warning("Failed to inject suggestion")

        self._run_on_main(_accept)
        return True

    @staticmethod
    def _extract_first_segment(text: str) -> str:
        """Extract the first sentence or line from a suggestion text.

        Splits on newline first, then on '. ' (sentence boundary).
        If the text is a single sentence with no newlines, returns
        the full text.
        """
        # Split on newline first
        first_line = text.split("\n", 1)[0].strip()
        # If the original text had a newline, use just the first line
        if "\n" in text:
            return first_line
        # Otherwise try splitting on sentence boundary '. '
        dot_pos = first_line.find(". ")
        if dot_pos >= 0:
            return first_line[: dot_pos + 1]  # include the period
        # Single sentence — return all
        return text

    def _on_partial_accept(self) -> bool:
        """Handle shift+tab — inject only the first sentence/line."""
        if not self.overlay.is_visible:
            return False

        def _accept_partial():
            suggestion = self.overlay.accept_selection()
            if suggestion:
                partial_text = self._extract_first_segment(suggestion.text)
                success = self.injector.inject(
                    partial_text, replace=self._replace_on_inject,
                )
                if success:
                    logger.info(f"Partial inject: {partial_text[:60]}")
                    focused = self.observer.get_focused_element()
                    app_name = focused.app_name if focused else "Unknown"
                    self.context_store.add_entry(
                        source_app=app_name,
                        content=partial_text,
                        entry_type="accepted_suggestion",
                    )
                else:
                    logger.warning("Failed to inject partial suggestion")

        self._run_on_main(_accept_partial)
        return True

    def _on_nav_dismiss(self) -> bool:
        """Handle escape / click-outside / typing — only intercept when overlay is visible."""
        if not self.overlay.is_visible:
            return False

        # Cancel any pending auto-trigger
        self._auto_trigger_debouncer.cancel()

        # Record dismiss time for auto-trigger cooldown
        self._last_dismiss_time = time.time()

        # Bump generation_id so any in-flight LLM stream is discarded
        self._generation_id += 1

        # Record dismissed feedback for all currently shown suggestions
        def _dismiss():
            latency_ms = None
            if self._trigger_time is not None:
                latency_ms = (time.time() - self._trigger_time) * 1000
            for suggestion in self._current_suggestions:
                try:
                    self.context_store.record_feedback(
                        source_app=self._trigger_app,
                        mode=self._trigger_mode,
                        suggestion_text=suggestion.text,
                        action="dismissed",
                        suggestion_index=suggestion.index,
                        total_suggestions=len(self._current_suggestions),
                        latency_ms=latency_ms,
                    )
                except Exception:
                    logger.debug("Failed to record dismissed feedback", exc_info=True)
            self.overlay.hide()

        self._run_on_main(_dismiss)
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
