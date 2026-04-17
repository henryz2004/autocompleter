"""Dump trigger data (AX tree, context, suggestions) for diagnostics.

Each hotkey trigger produces a JSON file containing everything needed to
reproduce and debug the autocomplete pipeline:

- The AX tree of the focused window (same format as ``dump_ax_tree_json.py``)
- Focused element metadata (cursor position, before/after cursor text)
- Assembled context string sent to the LLM
- Mode (continuation / reply)
- Generated suggestions
- Detection flags (TUI, shell, messaging)

Files can be loaded with ``tests/ax_fixture_loader.py`` to build regression
tests for context extractors.

Enable via ``--dump-dir <path>`` CLI flag.  Disabled by default.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Turn an app name into a filename-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _json_safe(value):
    """Recursively coerce artifact values into JSON-safe shapes.

    Accessibility / AppKit APIs occasionally surface wrapper objects like
    ``AXValueRef`` inside captured payloads. We never want artifact writing to
    fail because one field contains an opaque object, so unknown values are
    downgraded to a small descriptive string.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    return f"<nonserializable:{type(value).__name__}>"


@dataclass
class TriggerSnapshot:
    """Captures all data from a single trigger event."""

    # Identity
    generation_id: int = 0
    timestamp: str = ""
    app_name: str = ""
    window_title: str = ""
    source_url: str = ""
    trigger_type: str = "manual"

    # Focused element
    role: str = ""
    before_cursor: str = ""
    after_cursor: str = ""
    insertion_point: int | None = None
    selection_length: int = 0
    value_length: int = 0
    placeholder_detected: bool = False
    raw_value: str = ""
    raw_placeholder_value: str = ""
    raw_number_of_characters: int | None = None

    # AX tree (serialized)
    ax_tree: dict | None = None

    # Detection
    mode: str = ""
    use_shell: bool = False
    use_tui: bool = False
    tui_name: str = ""
    tui_user_input: str = ""
    has_conversation_turns: bool = False
    conversation_turn_count: int = 0

    # Context
    context: str = ""
    context_inputs: dict[str, Any] = field(default_factory=dict)
    conversation_turns: list[dict[str, str]] = field(default_factory=list)
    request: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)

    # Suggestions (filled after LLM responds)
    suggestions: list[str] = field(default_factory=list)
    suggestion_latency_ms: float | None = None


class TriggerDumper:
    """Writes trigger snapshots to JSON files in a directory.

    Usage::

        dumper = TriggerDumper("dumps/")

        # At trigger time — capture AX tree + focused element
        snap = dumper.start_snapshot(generation_id, focused, visible)

        # After context assembly
        snap.context = context
        snap.mode = mode.value
        ...

        # After suggestions arrive
        snap.suggestions = [s.text for s in suggestions]
        dumper.save(snap)
    """

    def __init__(self, dump_dir: str):
        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        logger.info(f"Trigger dumper enabled: {dump_dir}")

    def new_snapshot(self, generation_id: int) -> TriggerSnapshot:
        """Create a new empty snapshot for a trigger."""
        return TriggerSnapshot(
            generation_id=generation_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def capture_ax_tree(self, snapshot: TriggerSnapshot) -> None:
        """Capture the AX tree of the focused window.

        This is intentionally a separate step because AX tree serialization
        can be slow (~50-200ms) and should only run when dumping is enabled.
        """
        try:
            snapshot.ax_tree = self.capture_current_ax_tree(max_depth=20)
        except Exception:
            logger.debug("Failed to capture AX tree for dump", exc_info=True)

    @staticmethod
    def capture_current_ax_tree(
        *,
        max_depth: int = 20,
    ) -> dict[str, Any] | None:
        import AppKit
        from ApplicationServices import AXUIElementCreateApplication

        from .ax_utils import ax_get_attribute, serialize_ax_tree

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if not front_app:
            return None

        pid = front_app.processIdentifier()
        app_el = AXUIElementCreateApplication(pid)
        window = ax_get_attribute(app_el, "AXFocusedWindow")
        if window is None:
            windows = ax_get_attribute(app_el, "AXWindows")
            window = windows[0] if windows else None
        if window is None:
            return None

        focused_el = ax_get_attribute(app_el, "AXFocusedUIElement")
        return serialize_ax_tree(
            window,
            max_depth=max_depth,
            focused_element=focused_el,
        )

    @staticmethod
    def build_envelope(snapshot: TriggerSnapshot) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "artifactType": "manual_invocation_v1",
            "app": snapshot.app_name,
            "windowTitle": snapshot.window_title,
            "capturedAt": snapshot.timestamp,
            "macosVersion": platform.mac_ver()[0],
            "generationId": snapshot.generation_id,
            "triggerType": snapshot.trigger_type,
            "focused": {
                "role": snapshot.role,
                "insertionPoint": snapshot.insertion_point,
                "selectionLength": snapshot.selection_length,
                "valueLength": snapshot.value_length,
                "placeholderDetected": snapshot.placeholder_detected,
                "beforeCursor": snapshot.before_cursor,
                "afterCursor": snapshot.after_cursor,
                "sourceUrl": snapshot.source_url,
                "rawValue": snapshot.raw_value,
                "rawValueLength": len(snapshot.raw_value or ""),
                "rawPlaceholderValue": snapshot.raw_placeholder_value,
                "rawNumberOfCharacters": snapshot.raw_number_of_characters,
            },
            "detection": {
                "mode": snapshot.mode,
                "useShell": snapshot.use_shell,
                "useTui": snapshot.use_tui,
                "tuiName": snapshot.tui_name,
                "tuiUserInput": snapshot.tui_user_input,
                "hasConversationTurns": snapshot.has_conversation_turns,
                "conversationTurnCount": snapshot.conversation_turn_count,
            },
            "context": snapshot.context,
            "contextInputs": snapshot.context_inputs,
            "conversationTurns": snapshot.conversation_turns,
            "request": snapshot.request,
            "latency": snapshot.latency,
            "suggestions": snapshot.suggestions,
            "suggestionLatencyMs": snapshot.suggestion_latency_ms,
            "tree": snapshot.ax_tree,
        }
        return _json_safe(envelope)

    def save(self, snapshot: TriggerSnapshot) -> str | None:
        """Write the snapshot to a JSON file.  Returns the file path."""
        slug = _slugify(snapshot.app_name) if snapshot.app_name else "unknown"
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{ts}-gen{snapshot.generation_id}-{slug}.json"
        path = os.path.join(self.dump_dir, filename)

        envelope = self.build_envelope(snapshot)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=2, ensure_ascii=False)
            logger.info(f"Trigger dump saved: {path}")
            return path
        except Exception:
            logger.warning("Failed to save trigger dump", exc_info=True)
            return None
