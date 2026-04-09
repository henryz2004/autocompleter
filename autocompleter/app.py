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
from .context_trail import ContextTrail, _filter_ui_chrome
from .embeddings import (
    AnthropicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    TFIDFEmbeddingProvider,
)
from .hotkey import HotkeyListener
from .input_observer import InputObserver, VisibleContent
from .latency_tracker import LatencyStore, LatencyTracker
from .overlay import OverlayConfig, SuggestionOverlay
from .quality_review import (
    LIVE_CONTINUATION_VARIANT,
    LIVE_REPLY_VARIANT,
    apply_quality_variant_to_context,
)
from .shell_parser import parse_terminal_buffer, parse_tui_buffer
from .suggestion_engine import AutocompleteMode, Suggestion, SuggestionEngine, detect_mode, is_shell_app
from .memory import MemoryStore
from .text_injector import TextInjector
from .trigger_dump import TriggerDumper, TriggerSnapshot

# Max chars of terminal buffer to send as context for TUI (non-shell) apps.
# Shell parser uses 1500 for structured history; 3000 gives ~50-60 lines of
# unstructured conversation context without the 145K+ noise from full buffers.
_TUI_CONTEXT_TAIL_CHARS = 3000


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
            # Validate: reject rects with invalid coordinates, but accept
            # zero-height rects (Chromium sometimes reports h=0 with valid x,y).
            _DEFAULT_CARET_HEIGHT = 20.0
            if rect.origin.x > 0 and rect.origin.y >= 0:
                height = rect.size.height if rect.size.height > 0 else _DEFAULT_CARET_HEIGHT
                return (rect.origin.x, rect.origin.y + height, height)
            else:
                logger.debug("Caret rect has invalid coordinates, ignoring")
        else:
            logger.debug("Caret: AXValueGetValue failed")
    except Exception:
        logger.debug("Could not get caret position via AXBoundsForRange", exc_info=True)

    return None


class Autocompleter:
    """Main application class that orchestrates all components."""

    def __init__(self, config: Config | None = None, dump_dir: str | None = None):
        self.config = config or load_config()
        self._dumper: TriggerDumper | None = TriggerDumper(dump_dir) if dump_dir else None

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
        self.memory = MemoryStore(self.config)

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
        self._latency_tracker = LatencyTracker()
        self._latency_store = LatencyStore(self.config.data_dir / "context.db")
        self._trigger_before_cursor: str = ""  # before_cursor at trigger time (for leading-space stripping)
        self._trigger_after_cursor: str = ""   # after_cursor at trigger time (for trailing-space stripping)

        # Observer loop state
        self._observe_iteration: int = 0  # Counter for periodic pruning
        self._observe_consecutive_errors: int = 0  # For exponential backoff
        self._last_visible_content: VisibleContent | None = None  # Cached for trigger reuse
        self._last_visible_content_time: float = 0.0  # Timestamp of cached content

        # Dynamic polling state
        self._current_poll_interval: float = self._POLL_MIN  # Start fast
        self._last_observed_app: str = ""
        self._last_observed_window: str = ""

        # Regenerate state — stores the last trigger's arguments so we can
        # replay the LLM call with a fresh generation_id (new sampling).
        self._last_trigger_args: dict | None = None

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
                self._on_nav_dismiss(set_cooldown=False)
            if self._auto_trigger_enabled:
                self._auto_trigger_debouncer.poke()
        self.hotkey_listener.set_unhandled_key_callback(_on_typing)

        # Register hotkeys — callbacks return True to suppress, False to pass through
        self.hotkey_listener.register(
            self.config.hotkey, self._on_trigger
        )
        self.hotkey_listener.register(
            f"shift+{self.config.hotkey}", self._on_toggle_auto_trigger
        )
        self.hotkey_listener.register(
            self.config.regenerate_hotkey, self._on_regenerate
        )
        self.hotkey_listener.register("up", self._on_nav_up)
        self.hotkey_listener.register("down", self._on_nav_down)
        self.hotkey_listener.register("tab", self._on_nav_accept)
        self.hotkey_listener.register("return", self._on_nav_accept)
        self.hotkey_listener.register("shift+tab", self._on_partial_accept)
        self.hotkey_listener.register("escape", self._on_nav_dismiss)
        # Number keys for direct selection (1-indexed)
        for i in range(1, 4):
            self.hotkey_listener.register(str(i), lambda idx=i: self._on_nav_accept_index(idx - 1))

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
        print(f"Press {self.config.regenerate_hotkey} to regenerate suggestions.")
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

    def _visible_content_hash(self, visible: VisibleContent | None) -> str:
        """Build a stable hash for a visible-content snapshot."""
        if visible is None:
            return ""
        joined = "\n".join((visible.text_elements or [])[:50])
        parts = [
            visible.app_name,
            visible.window_title,
            visible.url,
            joined,
        ]
        return self._hash_content("\0".join(parts))

    def _resolve_visible_content(
        self,
        focused,
    ) -> tuple[VisibleContent | None, dict[str, object]]:
        """Return visible content plus measurement metadata for this trigger."""
        self._latency_tracker.mark("visible_start")
        now = time.time()
        cache_age_ms = None
        if self._last_visible_content_time > 0:
            cache_age_ms = max(0.0, (now - self._last_visible_content_time) * 1000)

        metadata: dict[str, object] = {
            "visible_source": "fresh",
            "visible_cache_age_ms": cache_age_ms,
            "visible_content_changed": None,
        }

        use_cache = (
            self._last_visible_content is not None
            and (now - self._last_visible_content_time) < self._current_poll_interval
            and self._last_visible_content.app_name == focused.app_name
        )
        cached = self._last_visible_content if use_cache else None
        cached_hash = self._visible_content_hash(cached)

        if use_cache:
            visible = cached
            metadata["visible_source"] = "cache"
            logger.debug(
                "Using cached visible content (age=%.0fms)",
                cache_age_ms or 0.0,
            )
        else:
            visible = self.observer.get_visible_content()
            fresh_hash = self._visible_content_hash(visible)
            if (
                self._last_visible_content is not None
                and visible is not None
                and self._last_visible_content.app_name == visible.app_name
                and self._last_visible_content.window_title == visible.window_title
            ):
                metadata["visible_content_changed"] = bool(
                    cached_hash and fresh_hash and cached_hash != fresh_hash
                )
        self._latency_tracker.mark("visible_ready")
        return visible, metadata

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

                # Pre-warm memory cache (fire-and-forget, non-blocking).
                # Uses app name + window title + recent visible text as a
                # composite query — much richer than cursor text alone.
                if self.memory.enabled and self._last_observed_app:
                    from autocompleter.memory import MemoryStore
                    _snippet = ""
                    if content and content.text_elements:
                        _snippet = "\n".join(content.text_elements[-5:])
                    mem_query = MemoryStore.build_query(
                        app_name=self._last_observed_app,
                        window_title=self._last_observed_window,
                        visible_snippet=_snippet,
                    )
                    if mem_query:
                        self.memory.pre_warm(mem_query)

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
        self._latency_tracker.start(generation_id=self._generation_id + 1)
        self._latency_tracker.mark("trigger")

        if self.overlay.is_visible:
            logger.debug("Overlay visible, hiding it")
            # Bump generation_id so any in-flight LLM stream is discarded
            self._generation_id += 1
            self._run_on_main(self.overlay.hide)
            return True

        # Gather info quickly (AX calls are fast) then dispatch heavy work
        focused = self.observer.get_focused_element()
        self._latency_tracker.mark("focused_ready")
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

        # Remember cursor context at trigger time for space dedup on accept
        self._trigger_before_cursor = focused.before_cursor
        self._trigger_after_cursor = focused.after_cursor

        # Remember whether the field has a baked-in placeholder so the
        # injector can skip AXValue setting (which bypasses the web app's
        # placeholder-clearing JS) and use clipboard/keystrokes instead.
        self._replace_on_inject = focused.placeholder_detected

        # Force clipboard/keystroke injection for terminals — AX value
        # setting would replace the entire terminal buffer.
        if is_shell_app(focused.app_name):
            self._replace_on_inject = True

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
        self._latency_tracker.mark("caret_ready")

        # Show loading indicator immediately
        self._run_on_main(lambda: self.overlay.show(
            [Suggestion(text="Generating...", index=0)], x, y,
            caret_height=caret_height,
        ))

        # Fetch subtree context (XML from walking up from focused element).
        # Done before visible content because it's fast and more reliable.
        self._latency_tracker.mark("subtree_start")
        subtree_context = self.observer.get_subtree_context(token_budget=500)
        self._latency_tracker.mark("subtree_ready")
        if subtree_context:
            logger.info(
                f"[CTX] Subtree context: {len(subtree_context)} chars "
                f"(~{len(subtree_context) // 4} tokens)"
            )

        visible, visible_meta = self._resolve_visible_content(focused)
        window_title = visible.window_title if visible else ""
        source_url = visible.url if visible else ""
        visible_text_elements: list[str] = []
        conversation_turns = None
        if visible:
            visible_text_elements = visible.text_elements or []
            # Filter UI chrome (buttons, sidebar labels, status indicators) from
            # visible text before it reaches the LLM context.
            if visible_text_elements:
                filtered = _filter_ui_chrome("\n".join(visible_text_elements))
                visible_text_elements = [
                    line for line in filtered.split("\n") if line.strip()
                ]
            logger.info(
                f"[CTX] Window: {visible.window_title!r} | URL: {visible.url!r} | "
                f"text_elements: {len(visible_text_elements)} | "
                f"conversation_turns: {len(visible.conversation_turns) if visible.conversation_turns else 0}"
            )
            if visible.conversation_turns:
                conversation_turns = [
                    {"speaker": t.speaker, "text": t.text, "timestamp": t.timestamp}
                    for t in visible.conversation_turns
                ]
                for i, t in enumerate(visible.conversation_turns):
                    ts_info = f" [{t.timestamp}]" if t.timestamp else ""
                    logger.debug(f"[CTX]   Turn [{i}]: {t.speaker}{ts_info}: {t.text[:100]!r}")
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

        # Build trigger dump snapshot if dumping is enabled
        snapshot: TriggerSnapshot | None = None
        if self._dumper:
            snapshot = self._dumper.new_snapshot(gen_id)
            snapshot.app_name = focused.app_name
            snapshot.window_title = window_title
            snapshot.source_url = source_url
            snapshot.trigger_type = "manual"
            snapshot.role = focused.role
            snapshot.before_cursor = focused.before_cursor
            snapshot.after_cursor = focused.after_cursor
            snapshot.insertion_point = focused.insertion_point
            snapshot.value_length = len(focused.value)
            snapshot.placeholder_detected = focused.placeholder_detected
            snapshot.visible_source = str(visible_meta.get("visible_source", ""))
            snapshot.visible_cache_age_ms = (
                float(visible_meta["visible_cache_age_ms"])
                if visible_meta.get("visible_cache_age_ms") is not None
                else None
            )
            snapshot.visible_content_changed = (
                bool(visible_meta["visible_content_changed"])
                if visible_meta.get("visible_content_changed") is not None
                else None
            )
            snapshot.visible_text_elements = list(visible_text_elements or [])
            snapshot.request["live_context_variant"] = LIVE_REPLY_VARIANT.name if mode == AutocompleteMode.REPLY else LIVE_CONTINUATION_VARIANT.name
            if conversation_turns:
                snapshot.conversation_turns = list(conversation_turns)
                snapshot.has_conversation_turns = True
                snapshot.conversation_turn_count = len(conversation_turns)
            # Capture AX tree on a background thread to avoid blocking the tap
            self._dumper.capture_ax_tree(snapshot)

        # Save trigger state so _on_regenerate can replay the LLM call
        self._last_trigger_args = dict(
            focused=focused,
            x=x, y=y, caret_height=caret_height,
            mode=mode,
            window_title=window_title,
            source_url=source_url,
            conversation_turns=conversation_turns,
            visible_text_elements=visible_text_elements,
            cross_app_context=cross_app_context,
            subtree_context=subtree_context,
            visible_meta=visible_meta,
            trigger_type="manual",
        )

        # Dispatch the LLM call to a worker thread so we don't block the tap.
        # Use the streaming path by default for faster perceived response.
        threading.Thread(
            target=self._generate_and_show_streaming,
            args=(
                focused, x, y, caret_height, mode,
                window_title, source_url, conversation_turns,
                visible_text_elements, gen_id, cross_app_context,
                snapshot, subtree_context, 0.0, None, visible_meta, "manual",
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
        subtree_context: str | None = None,
    ) -> None:
        """Run the LLM call on a worker thread and show the overlay."""
        app_name = focused.app_name
        current_input = focused.value

        if mode is None:
            mode = detect_mode(before_cursor=focused.before_cursor)

        # Terminal context detection: TUI first, then shell.
        # TUI detection must run before shell detection because in tmux
        # split-pane layouts, _strip_tmux_split_panes may pick the wrong
        # pane — if it picks the shell pane, parse_terminal_buffer sees a
        # prompt and we'd never try TUI detection.
        use_shell = False
        use_tui = False
        effective_before_cursor = focused.before_cursor
        # Read pre-warmed memory context (populated by observer loop — instant,
        # no API call).  Falls back to "" if cache is cold or memory disabled.
        memory_context = self.memory.get_cached_context()
        if memory_context:
            logger.info(f"[MEM] Using cached memory context ({len(memory_context)} chars)")

        if is_shell_app(app_name):
            # Try TUI detection first (e.g. Claude Code running inside terminal)
            tui = parse_tui_buffer(focused.before_cursor, focused.after_cursor)
            if tui.is_tui:
                use_tui = True
                logger.info(f"[CTX] TUI detected: {tui.tui_name} | user_input={len(tui.user_input)} chars | turns={len(tui.conversation_turns)}")
                effective_before_cursor = tui.user_input
                # Always use reply mode for TUI apps — the user is composing
                # a message to an AI agent, not continuing prose.
                mode = AutocompleteMode.REPLY
                # Use conversation turns from the TUI
                if not conversation_turns and tui.conversation_turns:
                    conversation_turns = tui.conversation_turns
            else:
                # Fall back to shell detection
                parsed = parse_terminal_buffer(focused.before_cursor)
                if parsed.is_likely_shell_context:
                    use_shell = True
                    context = self.context_store.get_shell_context(
                        parsed=parsed,
                        source_app=app_name,
                        window_title=window_title,
                        cross_app_context=cross_app_context,
                        memory_context=memory_context,
                    )
                    mode = AutocompleteMode.CONTINUATION if parsed.current_command.strip() else AutocompleteMode.REPLY
                else:
                    logger.info("[CTX] Terminal detected but not at shell prompt — using generic mode")
                    # Trim to tail to avoid sending 145K+ chars of terminal buffer
                    effective_before_cursor = focused.before_cursor[-_TUI_CONTEXT_TAIL_CHARS:]

        # Messaging apps: force reply mode when structured conversation turns
        # were extracted.  The user is composing a message in a chat, not
        # continuing prose.  Suppress raw visible text ONLY when the
        # conversation has multiple speakers (i.e. we captured both sides).
        # When turns are one-sided (e.g. Gemini only exposes user messages),
        # keep visible text so the LLM can see assistant responses.
        if not use_shell and not use_tui and conversation_turns:
            mode = AutocompleteMode.REPLY
            speakers = {t.speaker for t in conversation_turns}
            if len(speakers) > 1:
                visible_text_elements = []
                logger.info(
                    "[CTX] Messaging app detected with %d turns (%d speakers) — "
                    "forcing reply mode, suppressing raw visible text",
                    len(conversation_turns), len(speakers),
                )
            else:
                logger.info(
                    "[CTX] Messaging app detected with %d turns (one-sided: %s) — "
                    "forcing reply mode, KEEPING visible text for assistant context",
                    len(conversation_turns),
                    speakers.pop() if speakers else "?",
                )

        # Standard context assembly based on mode
        if not use_shell and mode == AutocompleteMode.CONTINUATION:
            context = self.context_store.get_continuation_context(
                before_cursor=effective_before_cursor,
                after_cursor=focused.after_cursor,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                visible_text=visible_text_elements,
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context,
                cross_app_context=cross_app_context,
                subtree_context=subtree_context,
                memory_context=memory_context,
            )
        elif not use_shell:
            # For TUI apps, suppress visible text and semantic context —
            # the observer stores raw terminal buffer (including adjacent
            # tmux panes) in the DB which pollutes semantic search results.
            context = self.context_store.get_reply_context(
                conversation_turns=conversation_turns or [],
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=effective_before_cursor if focused.insertion_point is not None else "",
                visible_text=visible_text_elements if not use_tui else [],
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context and not use_tui,
                cross_app_context=cross_app_context,
                subtree_context=subtree_context if not use_tui else None,
                memory_context=memory_context,
            )

        live_variant = (
            LIVE_CONTINUATION_VARIANT
            if mode == AutocompleteMode.CONTINUATION
            else LIVE_REPLY_VARIANT
        )
        context = apply_quality_variant_to_context(context, live_variant)
        logger.info(
            f"[CTX] --- CONTEXT (gen={generation_id}, mode={mode.value}, "
            f"{len(context)} chars, variant={live_variant.name}) ---"
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

        self._latency_tracker.mark("context_ready")
        self._latency_tracker.mark("llm_start")

        t0 = time.time()
        suggestions = self.suggestion_engine.generate_suggestions(
            current_input=current_input,
            context=context,
            app_name=app_name,
            mode=mode,
            feedback_stats=feedback_stats,
            negative_patterns=negative_patterns,
            shell_mode=use_shell,
            prompt_placeholder_aware=True,
        )
        elapsed = time.time() - t0

        # Blocking path: first_suggestion and llm_done are the same moment
        if suggestions:
            self._latency_tracker.mark("first_suggestion")
        self._latency_tracker.mark("llm_done")

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
        self._latency_tracker.mark("displayed")

        record = self._latency_tracker.finish(
            app_name=app_name,
            mode=mode.value,
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            suggestion_count=len(suggestions),
        )
        self._latency_store.save(record)

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
        snapshot: TriggerSnapshot | None = None,
        subtree_context: str | None = None,
        temperature_boost: float = 0.0,
        extra_negative_patterns: list[str] | None = None,
        visible_meta: dict[str, object] | None = None,
        trigger_type: str = "",
    ) -> None:
        """Run the streaming LLM call on a worker thread, updating the overlay incrementally."""
        app_name = focused.app_name
        current_input = focused.value

        if mode is None:
            mode = detect_mode(before_cursor=focused.before_cursor)

        # Terminal context detection: TUI first, then shell.
        # TUI detection must run before shell detection because in tmux
        # split-pane layouts, _strip_tmux_split_panes may pick the wrong
        # pane — if it picks the shell pane, parse_terminal_buffer sees a
        # prompt and we'd never try TUI detection.
        use_shell = False
        use_tui = False
        effective_before_cursor = focused.before_cursor

        # Read pre-warmed memory context (populated by observer loop — instant,
        # no API call).  Falls back to "" if cache is cold or memory disabled.
        memory_context = self.memory.get_cached_context()
        if memory_context:
            logger.info(f"[MEM] Using cached memory context ({len(memory_context)} chars)")

        if is_shell_app(app_name):
            # Try TUI detection first (e.g. Claude Code running inside terminal)
            tui = parse_tui_buffer(focused.before_cursor, focused.after_cursor)
            if tui.is_tui:
                use_tui = True
                logger.info(f"[CTX] TUI detected: {tui.tui_name} | user_input={len(tui.user_input)} chars | turns={len(tui.conversation_turns)}")
                effective_before_cursor = tui.user_input
                # Always use reply mode for TUI apps — the user is composing
                # a message to an AI agent, not continuing prose.
                mode = AutocompleteMode.REPLY
                # Use conversation turns from the TUI
                if not conversation_turns and tui.conversation_turns:
                    conversation_turns = tui.conversation_turns
            else:
                # Fall back to shell detection
                parsed = parse_terminal_buffer(focused.before_cursor)
                if parsed.is_likely_shell_context:
                    use_shell = True
                    context = self.context_store.get_shell_context(
                        parsed=parsed,
                        source_app=app_name,
                        window_title=window_title,
                        cross_app_context=cross_app_context,
                        memory_context=memory_context,
                    )
                    mode = AutocompleteMode.CONTINUATION if parsed.current_command.strip() else AutocompleteMode.REPLY
                else:
                    logger.info("[CTX] Terminal detected but not at shell prompt — using generic mode")
                    # Trim to tail to avoid sending 145K+ chars of terminal buffer
                    effective_before_cursor = focused.before_cursor[-_TUI_CONTEXT_TAIL_CHARS:]

        # Messaging apps: force reply mode when structured conversation turns
        # were extracted.  The user is composing a message in a chat, not
        # continuing prose.  Suppress raw visible text ONLY when the
        # conversation has multiple speakers (i.e. we captured both sides).
        # When turns are one-sided (e.g. Gemini only exposes user messages),
        # keep visible text so the LLM can see assistant responses.
        if not use_shell and not use_tui and conversation_turns:
            mode = AutocompleteMode.REPLY
            speakers = {t.speaker for t in conversation_turns}
            if len(speakers) > 1:
                visible_text_elements = []
                logger.info(
                    "[CTX] Messaging app detected with %d turns (%d speakers) — "
                    "forcing reply mode, suppressing raw visible text",
                    len(conversation_turns), len(speakers),
                )
            else:
                logger.info(
                    "[CTX] Messaging app detected with %d turns (one-sided: %s) — "
                    "forcing reply mode, KEEPING visible text for assistant context",
                    len(conversation_turns),
                    speakers.pop() if speakers else "?",
                )

        # Standard context assembly based on mode
        self._latency_tracker.mark("context_build_start")
        if not use_shell and mode == AutocompleteMode.CONTINUATION:
            context = self.context_store.get_continuation_context(
                before_cursor=effective_before_cursor,
                after_cursor=focused.after_cursor,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                visible_text=visible_text_elements,
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context,
                cross_app_context=cross_app_context,
                subtree_context=subtree_context,
                memory_context=memory_context,
            )
            used_semantic_context = bool(self.config.use_semantic_context)
        elif not use_shell:
            # For TUI apps, suppress visible text and semantic context —
            # the observer stores raw terminal buffer (including adjacent
            # tmux panes) in the DB which pollutes semantic search results.
            context = self.context_store.get_reply_context(
                conversation_turns=conversation_turns or [],
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=effective_before_cursor if focused.insertion_point is not None else "",
                visible_text=visible_text_elements if not use_tui else [],
                embedding_provider=self._embedding_provider,
                use_semantic_context=self.config.use_semantic_context and not use_tui,
                cross_app_context=cross_app_context,
                subtree_context=subtree_context if not use_tui else None,
                memory_context=memory_context,
            )
            used_semantic_context = bool(
                self.config.use_semantic_context and not use_tui
            )
        else:
            used_semantic_context = False

        live_variant = (
            LIVE_CONTINUATION_VARIANT
            if mode == AutocompleteMode.CONTINUATION
            else LIVE_REPLY_VARIANT
        )
        context = apply_quality_variant_to_context(context, live_variant)
        logger.info(
            f"[CTX] --- CONTEXT (gen={generation_id}, mode={mode.value}, "
            f"{len(context)} chars, variant={live_variant.name}) ---"
        )
        for line in context.splitlines():
            logger.info(f"[CTX]   | {line}")
        logger.info("[CTX] --- END CONTEXT ---")

        # Populate snapshot with detection results and context
        if snapshot:
            snapshot.mode = mode.value
            snapshot.use_shell = use_shell
            snapshot.use_tui = use_tui
            snapshot.request.setdefault("live_context_variant", live_variant.name)
            if use_tui:
                snapshot.tui_name = tui.tui_name
                snapshot.tui_user_input = tui.user_input
            snapshot.context = context
            if conversation_turns:
                snapshot.conversation_turns = list(conversation_turns)
                snapshot.has_conversation_turns = True
                snapshot.conversation_turn_count = len(conversation_turns)

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

        # Merge extra negative patterns (e.g. previous suggestions on regenerate)
        if extra_negative_patterns:
            negative_patterns = (negative_patterns or []) + extra_negative_patterns

        self._latency_tracker.mark("context_ready")
        self._latency_tracker.mark("llm_start")
        fallback_used = False

        def _on_stream_event(event_name: str, payload: dict[str, object] | None = None) -> None:
            nonlocal fallback_used
            if event_name == "fallback_started":
                fallback_used = True
            if snapshot is not None and payload is not None:
                if event_name == "request_built":
                    snapshot.request = dict(payload)
                else:
                    snapshot.request.setdefault("events", []).append({
                        "name": event_name,
                        "payload": dict(payload),
                    })

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
                shell_mode=use_shell,
                temperature_boost=temperature_boost,
                event_callback=_on_stream_event,
                prompt_placeholder_aware=True,
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

                # Mark first suggestion arrival (TTFT)
                if len(suggestions) == 1:
                    self._latency_tracker.mark("first_suggestion")

                logger.info(
                    f"Stream [{generation_id}]: suggestion {suggestion.index} arrived "
                    f"({time.time() - t0:.2f}s): {suggestion.text[:80]!r}"
                )

                # Capture a snapshot of suggestions for the closure
                suggestions_snapshot = list(suggestions)

                def _update(snp=suggestions_snapshot):
                    if generation_id == self._generation_id:
                        if len(snp) == 1:
                            self._latency_tracker.mark("overlay_first_show")
                        self.overlay.show(snp, x, y, caret_height=caret_height)

                self._run_on_main(_update)

        except Exception:
            logger.exception(f"Error during streaming generation {generation_id}")

        elapsed = time.time() - t0
        self._latency_tracker.mark("llm_done")

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
            record = self._latency_tracker.finish(
                app_name=app_name,
                mode=mode.value,
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                suggestion_count=0,
                trigger_type=trigger_type,
                visible_source=str((visible_meta or {}).get("visible_source", "")),
                use_shell=use_shell,
                use_tui=use_tui,
                has_conversation_turns=bool(conversation_turns),
                used_subtree_context=bool(subtree_context),
                used_semantic_context=used_semantic_context,
                used_memory_context=bool(memory_context),
                fallback_used=fallback_used,
                visible_cache_age_ms=(
                    float((visible_meta or {}).get("visible_cache_age_ms"))
                    if (visible_meta or {}).get("visible_cache_age_ms") is not None
                    else None
                ),
                visible_content_changed=(
                    bool((visible_meta or {}).get("visible_content_changed"))
                    if (visible_meta or {}).get("visible_content_changed") is not None
                    else None
                ),
            )
            self._latency_store.save(record)
            # Save snapshot even with no suggestions (useful for debugging)
            if snapshot and self._dumper:
                snapshot.latency = {
                    k: v for k, v in vars(record).items()
                }
                snapshot.suggestion_latency_ms = elapsed * 1000
                self._dumper.save(snapshot)
            return

        self._latency_tracker.mark("displayed")

        logger.info(
            f"--- STREAM COMPLETE (gen={generation_id}, {elapsed:.2f}s, "
            f"{len(suggestions)} suggestions) ---"
        )
        for i, s in enumerate(suggestions):
            logger.info(f"  [{i}]: {s.text[:120]}")

        self._current_suggestions = suggestions

        # Save trigger dump with suggestions
        if snapshot and self._dumper:
            snapshot.suggestions = [s.text for s in suggestions]
            snapshot.suggestion_latency_ms = elapsed * 1000

        # Record latency metrics
        record = self._latency_tracker.finish(
            app_name=app_name,
            mode=mode.value,
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            suggestion_count=len(suggestions),
            trigger_type=trigger_type,
            visible_source=str((visible_meta or {}).get("visible_source", "")),
            use_shell=use_shell,
            use_tui=use_tui,
            has_conversation_turns=bool(conversation_turns),
            used_subtree_context=bool(subtree_context),
            used_semantic_context=used_semantic_context,
            used_memory_context=bool(memory_context),
            fallback_used=fallback_used,
            visible_cache_age_ms=(
                float((visible_meta or {}).get("visible_cache_age_ms"))
                if (visible_meta or {}).get("visible_cache_age_ms") is not None
                else None
            ),
            visible_content_changed=(
                bool((visible_meta or {}).get("visible_content_changed"))
                if (visible_meta or {}).get("visible_content_changed") is not None
                else None
            ),
        )
        self._latency_store.save(record)
        if snapshot and self._dumper:
            snapshot.latency = {
                k: v for k, v in vars(record).items()
            }
            self._dumper.save(snapshot)

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
        self._latency_tracker.start(generation_id=self._generation_id + 1)
        self._latency_tracker.mark("trigger")
        focused = self.observer.get_focused_element()
        self._latency_tracker.mark("focused_ready")
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
        self._trigger_before_cursor = focused.before_cursor
        self._trigger_after_cursor = focused.after_cursor
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
        self._latency_tracker.mark("caret_ready")

        # Skip loading indicator for auto-trigger — suggestions just appear
        # silently when ready, without the disruptive "Generating..." flash.

        visible, visible_meta = self._resolve_visible_content(focused)
        window_title = visible.window_title if visible else ""
        source_url = visible.url if visible else ""
        visible_text_elements: list[str] = visible.text_elements if visible else []
        conversation_turns = None
        if visible and visible.conversation_turns:
            conversation_turns = [
                {"speaker": t.speaker, "text": t.text, "timestamp": t.timestamp}
                for t in visible.conversation_turns
            ]

        cross_app_snapshots = self.context_trail.get_recent_cross_app_context(
            current_app=focused.app_name,
            max_age_seconds=60.0,
            max_entries=3,
        )
        cross_app_context = ContextTrail.format_cross_app_context(cross_app_snapshots)

        # Fetch subtree context (XML from walking up from focused element)
        self._latency_tracker.mark("subtree_start")
        subtree_context = self.observer.get_subtree_context(token_budget=500)
        self._latency_tracker.mark("subtree_ready")

        # Save trigger state so _on_regenerate can replay (same as manual trigger)
        self._last_trigger_args = dict(
            focused=focused,
            x=x, y=y, caret_height=caret_height,
            mode=mode,
            window_title=window_title,
            source_url=source_url,
            conversation_turns=conversation_turns,
            visible_text_elements=visible_text_elements,
            cross_app_context=cross_app_context,
            subtree_context=subtree_context,
            visible_meta=visible_meta,
            trigger_type="auto",
        )

        self._generation_id += 1
        gen_id = self._generation_id

        threading.Thread(
            target=self._generate_and_show_streaming,
            args=(
                focused, x, y, caret_height, mode,
                window_title, source_url, conversation_turns,
                visible_text_elements, gen_id, cross_app_context,
                None, subtree_context, 0.0, None, visible_meta, "auto",
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

    def _on_regenerate(self) -> bool:
        """Regenerate suggestions using the same context but fresh sampling.

        Only active when the overlay is visible and we have saved trigger args.
        Validates that the user is still in the same app/field before replaying.
        Re-captures the caret position so the overlay tracks window movement.
        """
        if not self.overlay.is_visible:
            return False
        if self._last_trigger_args is None:
            return False

        # Validate: user must still be in the same app — otherwise stale
        # context would produce wrong suggestions and overlay would appear
        # at the old position.
        focused = self.observer.get_focused_element()
        self._latency_tracker.mark("focused_ready")
        if focused is None:
            logger.debug("Regenerate: no focused element, aborting")
            return False
        saved_focused = self._last_trigger_args["focused"]
        if focused.app_pid != saved_focused.app_pid:
            logger.info(
                f"Regenerate: app changed ({saved_focused.app_name!r} → "
                f"{focused.app_name!r}), aborting"
            )
            return False

        logger.info("--- REGENERATE ---")
        self._auto_trigger_debouncer.cancel()
        self._trigger_time = time.time()
        self._latency_tracker.start(generation_id=self._generation_id + 1)
        self._latency_tracker.mark("trigger")

        args = self._last_trigger_args

        # Re-capture caret position so overlay tracks window scroll/resize
        x, y, caret_height = args["x"], args["y"], args["caret_height"]
        caret_pos = _get_caret_screen_position()
        if caret_pos:
            x, y, caret_height = caret_pos
            # Apply same element-bottom clamp as _on_trigger
            if focused.position and focused.size:
                elem_bottom = focused.position[1] + focused.size[1]
                if y < elem_bottom:
                    y = elem_bottom
        self._latency_tracker.mark("caret_ready")

        # Bump generation to discard any in-flight stream
        self._generation_id += 1
        gen_id = self._generation_id

        # Show loading indicator at the (updated) position
        self._run_on_main(lambda: self.overlay.show(
            [Suggestion(text="Regenerating...", index=0)],
            x, y, caret_height=caret_height,
        ))

        # Feed current suggestions as negative examples so the LLM avoids
        # repeating them.  This is more effective than temperature alone
        # because inference providers may cache prefixes aggressively.
        prev_texts = [s.text for s in self._current_suggestions if s.text.strip()]

        # Re-dispatch the streaming LLM call with boosted temperature for diversity
        threading.Thread(
            target=self._generate_and_show_streaming,
            args=(
                args["focused"], x, y, caret_height,
                args["mode"], args["window_title"], args["source_url"],
                args["conversation_turns"], args["visible_text_elements"],
                gen_id, args["cross_app_context"],
                None,  # snapshot — skip trigger dump on regenerate
                args["subtree_context"],
            ),
            kwargs={
                "temperature_boost": 0.3,
                "extra_negative_patterns": prev_texts,
                "visible_meta": args.get("visible_meta"),
                "trigger_type": "regenerate",
            },
            daemon=True,
        ).start()

        return True

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
                text = suggestion.text
                # Strip trailing newlines/carriage returns
                text = text.rstrip("\n\r")
                # Strip exactly one leading space if user already typed a trailing space
                if self._trigger_before_cursor.endswith(" ") and text.startswith(" "):
                    text = text[1:]
                # Strip exactly one trailing space if text after cursor starts with space
                if self._trigger_after_cursor.startswith(" ") and text.endswith(" "):
                    text = text[:-1]
                success = self.injector.inject(
                    text,
                    replace=self._replace_on_inject,
                    insertion_point=cursor_pos,
                    app_name=app_name,
                    app_pid=app_pid,
                )
                if success:
                    logger.info(f"Injected: {text[:60]}")
                    self.context_store.add_entry(
                        source_app=app_name,
                        content=suggestion.text,
                        entry_type="accepted_suggestion",
                    )
                    # Store in long-term memory (async, non-blocking).
                    # Use fresh before_cursor from the focused element captured
                    # at accept time (line 1339) — not the stale _trigger_before_cursor
                    # which was set at trigger time and may be outdated.
                    if self.memory.enabled:
                        before_text = (
                            focused.before_cursor if focused
                            else self._trigger_before_cursor
                        )
                        logger.debug(
                            f"[MEM] Storing accepted suggestion: {suggestion.text[:60]}"
                        )
                        self.memory.add_async(
                            [
                                {"role": "user", "content": before_text[-300:]},
                                {"role": "assistant", "content": suggestion.text},
                            ],
                            metadata={"app": app_name, "mode": self._trigger_mode},
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

    def _on_nav_accept_index(self, index: int) -> bool:
        """Handle number key — accept the suggestion at the given index.

        Only intercepts when the overlay is visible and the index is valid.
        Returns False (pass-through) so the key types normally when the
        overlay isn't showing.
        """
        if not self.overlay.is_visible:
            return False
        if index < 0 or index >= len(self._current_suggestions):
            return False

        # Select the target suggestion and accept it
        def _accept():
            focused = self.observer.get_focused_element()
            cursor_pos = focused.insertion_point if focused else None
            app_name = focused.app_name if focused else "Unknown"
            app_pid = focused.app_pid if focused else 0

            # Set the selection to the target index
            self.overlay._selected_index = index
            suggestion = self.overlay.accept_selection()
            if suggestion:
                text = suggestion.text
                text = text.rstrip("\n\r")
                if self._trigger_before_cursor.endswith(" ") and text.startswith(" "):
                    text = text[1:]
                if self._trigger_after_cursor.startswith(" ") and text.endswith(" "):
                    text = text[:-1]
                success = self.injector.inject(
                    text,
                    replace=self._replace_on_inject,
                    insertion_point=cursor_pos,
                    app_name=app_name,
                    app_pid=app_pid,
                )
                if success:
                    logger.info(f"Injected [{index}]: {text[:60]}")
                    self.context_store.add_entry(
                        source_app=app_name,
                        content=suggestion.text,
                        entry_type="accepted_suggestion",
                    )
                    if self.memory.enabled:
                        before_text = (
                            focused.before_cursor if focused
                            else self._trigger_before_cursor
                        )
                        self.memory.add_async(
                            [
                                {"role": "user", "content": before_text[-300:]},
                                {"role": "assistant", "content": suggestion.text},
                            ],
                            metadata={"app": app_name, "mode": self._trigger_mode},
                        )
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
                partial_text = partial_text.rstrip("\n\r")
                if self._trigger_before_cursor.endswith(" ") and partial_text.startswith(" "):
                    partial_text = partial_text[1:]
                if self._trigger_after_cursor.startswith(" ") and partial_text.endswith(" "):
                    partial_text = partial_text[:-1]
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

    def _on_nav_dismiss(self, *, set_cooldown: bool = True) -> bool:
        """Handle escape / click-outside / typing — only intercept when overlay is visible.

        Args:
            set_cooldown: When True (default), record dismiss time so auto-trigger
                respects the cooldown period. Pass False for "typing through"
                dismissals where the user didn't intentionally reject suggestions.
        """
        if not self.overlay.is_visible:
            return False

        # Cancel any pending auto-trigger
        self._auto_trigger_debouncer.cancel()

        # Record dismiss time for auto-trigger cooldown (skip for typing-through)
        if set_cooldown:
            self._last_dismiss_time = time.time()

        # Bump generation_id so any in-flight LLM stream is discarded
        self._generation_id += 1

        # Free saved trigger state (avoids holding stale FocusedElement/context)
        self._last_trigger_args = None

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
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all console output (logs still written to --log-file).",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Dump AX tree, context, and suggestions for each trigger to this directory.",
    )
    parser.add_argument(
        "--stats",
        nargs="?",
        const=50,
        type=int,
        metavar="N",
        help="Print latency stats for the last N triggers (default: 50) and exit.",
    )
    args = parser.parse_args()

    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Allow everything through; handlers filter

    # Stderr: show INFO+ by default (clean output), or nothing if --quiet
    if not args.quiet:
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

    if args.log_file and not args.quiet:
        print(f"Logging to {args.log_file} (file=DEBUG, console={args.log_level})")

    config = load_config()

    # --stats: print latency report and exit
    if args.stats is not None:
        from .latency_tracker import print_stats
        print_stats(config.data_dir / "context.db", last_n=args.stats)
        return

    app = Autocompleter(config, dump_dir=args.dump_dir)

    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
