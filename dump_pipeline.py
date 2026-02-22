#!/usr/bin/env python3
"""Pipeline diagnostic tool — press Ctrl+Space to dump the full context pipeline.

Shows every stage of the autocompleter pipeline for the frontmost app:
  A. Focused element (cursor state, placeholder detection)
  B. Visible text elements
  C. Conversation extraction (per-app extractor output)
  D. Mode detection (CONTINUATION vs REPLY)
  E. Assembled context string (what the LLM sees)
  F. LLM prompts (system + user)
  G. Raw AX tree

Usage:
    source venv/bin/activate
    python dump_pipeline.py                        # dump to stdout
    python dump_pipeline.py -o /tmp/pipeline.log   # dump to file (append)
    python dump_pipeline.py --depth 20             # increase AX tree depth
    python dump_pipeline.py --both-modes           # show both CONTINUATION and REPLY context
    python dump_pipeline.py --num-suggestions 5    # change suggestion count in prompts

Press Ctrl+Space while focused on the target app.
Press Ctrl+C to quit.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")

import AppKit
from autocompleter.ax_utils import dump_ax_tree
from autocompleter.context_store import ContextStore
from autocompleter.conversation_extractors import get_extractor
from autocompleter.hotkey import HotkeyListener
from autocompleter.input_observer import InputObserver
from autocompleter.suggestion_engine import (
    SYSTEM_PROMPT_COMPLETION,
    SYSTEM_PROMPT_REPLY,
    USER_PROMPT_TEMPLATE_COMPLETION,
    USER_PROMPT_TEMPLATE_REPLY,
    AutocompleteMode,
    MODE_THRESHOLD_CHARS,
    detect_mode,
)


def _indent(text: str, prefix: str = "  | ") -> str:
    """Indent each line of text with a prefix."""
    return "\n".join(prefix + line for line in text.splitlines()) if text else prefix + "(empty)"


def _section(f, title: str) -> None:
    """Write a section header."""
    f.write(f"\n=== {title} ===\n")


def _dump_focused_element(f, focused) -> None:
    """Dump section A: focused element details."""
    _section(f, "A. FOCUSED ELEMENT")
    f.write(f"  Role: {focused.role}\n")

    val_display = focused.value.replace("\n", "\\n")
    if len(val_display) > 200:
        val_display = val_display[:200] + "..."
    f.write(f"  Value ({len(focused.value)} chars): \"{val_display}\"\n")

    before = focused.before_cursor
    after = focused.after_cursor
    before_display = before.replace("\n", "\\n")
    after_display = after.replace("\n", "\\n")
    if len(before_display) > 200:
        before_display = before_display[:200] + "..."
    if len(after_display) > 200:
        after_display = after_display[:200] + "..."
    f.write(f"  Before cursor ({len(before)} chars): \"{before_display}\"\n")
    f.write(f"  After cursor ({len(after)} chars): \"{after_display}\"\n")

    ip = focused.insertion_point
    sl = focused.selection_length
    f.write(f"  Insertion point: {ip}  |  Selection length: {sl}\n")
    f.write(f"  Placeholder detected: {focused.placeholder_detected}\n")

    pos = focused.position
    size = focused.size
    pos_str = f"({pos[0]}, {pos[1]})" if pos else "None"
    size_str = f"({size[0]}, {size[1]})" if size else "None"
    f.write(f"  Position: {pos_str}  |  Size: {size_str}\n")


def _dump_visible_content(f, visible) -> None:
    """Dump section B: visible text elements."""
    _section(f, "B. VISIBLE TEXT ELEMENTS")
    f.write(f"  Window: \"{visible.window_title}\"  |  URL: \"{visible.url}\"\n")
    f.write(f"  Count: {len(visible.text_elements)} elements\n")
    for i, elem in enumerate(visible.text_elements):
        display = elem.replace("\n", "\\n")
        if len(display) > 120:
            display = display[:120] + "..."
        f.write(f"  [{i}]: \"{display}\"\n")
        if i >= 29:
            f.write(f"  ... ({len(visible.text_elements) - 30} more elements)\n")
            break


def _dump_conversation(f, visible) -> None:
    """Dump section C: conversation extraction."""
    _section(f, "C. CONVERSATION EXTRACTION")
    extractor = get_extractor(visible.app_name)
    f.write(f"  Extractor used: {type(extractor).__name__}\n")

    turns = visible.conversation_turns
    if turns is None:
        f.write("  No conversation turns extracted (returned None)\n")
    elif len(turns) == 0:
        f.write("  Conversation turns: 0 (empty list)\n")
    else:
        f.write(f"  Turns extracted: {len(turns)}\n")
        for i, turn in enumerate(turns):
            text_display = turn.text.replace("\n", "\\n")
            if len(text_display) > 120:
                text_display = text_display[:120] + "..."
            f.write(f"  [{i}] {turn.speaker}: \"{text_display}\"\n")
            if i >= 19:
                f.write(f"  ... ({len(turns) - 20} more turns)\n")
                break


def _dump_mode(f, mode: AutocompleteMode, before_cursor: str) -> None:
    """Dump section D: mode detection."""
    _section(f, "D. MODE DETECTION")
    stripped_len = len(before_cursor.strip())
    f.write(f"  Mode: {mode.name}\n")
    if mode == AutocompleteMode.CONTINUATION:
        f.write(f"  Reason: before_cursor has {stripped_len} chars (stripped), threshold is {MODE_THRESHOLD_CHARS}\n")
    else:
        f.write(f"  Reason: before_cursor has {stripped_len} chars (stripped), below threshold of {MODE_THRESHOLD_CHARS}\n")


def _dump_context(f, context: str, mode: AutocompleteMode) -> None:
    """Dump section E: assembled context."""
    _section(f, f"E. ASSEMBLED CONTEXT — {mode.name} (what the LLM sees)")
    f.write(_indent(context) + "\n")


def _dump_prompts(f, context: str, mode: AutocompleteMode, num_suggestions: int) -> None:
    """Dump section F: LLM prompts."""
    _section(f, "F. LLM PROMPTS")

    if mode == AutocompleteMode.CONTINUATION:
        system = SYSTEM_PROMPT_COMPLETION.format(num_suggestions=num_suggestions)
        user_msg = USER_PROMPT_TEMPLATE_COMPLETION.format(
            context=context or "(no context yet)",
            num_suggestions=num_suggestions,
        )
    else:
        system = SYSTEM_PROMPT_REPLY.format(
            num_suggestions=num_suggestions,
            max_suggestion_lines=10,
        )
        user_msg = USER_PROMPT_TEMPLATE_REPLY.format(
            context=context or "(no context yet)",
            num_suggestions=num_suggestions,
        )

    f.write(f"  --- SYSTEM PROMPT ({mode.name.lower()}, {num_suggestions} suggestions) ---\n")
    f.write(_indent(system) + "\n")
    f.write(f"  --- USER PROMPT ---\n")
    f.write(_indent(user_msg) + "\n")


def _dump_ax_tree_section(f, max_depth: int) -> None:
    """Dump section G: raw AX tree."""
    _section(f, f"G. RAW AX TREE (max_depth={max_depth})")

    workspace = AppKit.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    if not front_app:
        f.write("  (no frontmost app)\n")
        return

    from ApplicationServices import AXUIElementCreateApplication
    from autocompleter.ax_utils import ax_get_attribute

    pid = front_app.processIdentifier()
    app_el = AXUIElementCreateApplication(pid)
    window = ax_get_attribute(app_el, "AXFocusedWindow")
    if window is None:
        windows = ax_get_attribute(app_el, "AXWindows")
        window = windows[0] if windows else None
    if window is None:
        f.write("  (no window found)\n")
        return

    dump_ax_tree(window, f, max_depth=max_depth)


def on_trigger(out_path: str | None, max_depth: int, num_suggestions: int, both_modes: bool):
    """Called when hotkey is pressed. Dumps the full pipeline."""
    timestamp = time.strftime("%H:%M:%S")

    # Get frontmost app info
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    if not front_app:
        print("No frontmost app")
        return True

    app_name = front_app.localizedName() or "Unknown"
    pid = front_app.processIdentifier()

    # Open output
    if out_path:
        f = open(out_path, "a")
    else:
        f = sys.stdout

    try:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"[{timestamp}] Pipeline Diagnostic — App: {app_name} (pid={pid})\n")
        f.write(f"{'=' * 70}\n")

        observer = InputObserver()

        # --- A. Focused element ---
        focused = None
        try:
            focused = observer.get_focused_element()
            if focused:
                _dump_focused_element(f, focused)
            else:
                _section(f, "A. FOCUSED ELEMENT")
                f.write("  (no focused text element — role may not be a text input)\n")
        except Exception as e:
            _section(f, "A. FOCUSED ELEMENT")
            f.write(f"  ERROR: {e}\n")

        # --- B. Visible content ---
        visible = None
        try:
            visible = observer.get_visible_content()
            if visible:
                _dump_visible_content(f, visible)
            else:
                _section(f, "B. VISIBLE TEXT ELEMENTS")
                f.write("  (no visible content returned)\n")
        except Exception as e:
            _section(f, "B. VISIBLE TEXT ELEMENTS")
            f.write(f"  ERROR: {e}\n")

        # --- C. Conversation extraction ---
        try:
            if visible:
                _dump_conversation(f, visible)
            else:
                _section(f, "C. CONVERSATION EXTRACTION")
                f.write("  (skipped — no visible content)\n")
        except Exception as e:
            _section(f, "C. CONVERSATION EXTRACTION")
            f.write(f"  ERROR: {e}\n")

        # --- D/E/F require cursor state ---
        if focused:
            before_cursor = focused.before_cursor
            after_cursor = focused.after_cursor
            detected_mode = detect_mode(before_cursor=before_cursor)

            # --- D. Mode detection ---
            try:
                _dump_mode(f, detected_mode, before_cursor)
            except Exception as e:
                _section(f, "D. MODE DETECTION")
                f.write(f"  ERROR: {e}\n")

            # Set up ephemeral context store
            tmp_dir = tempfile.mkdtemp(prefix="dump_pipeline_")
            db_path = Path(tmp_dir) / "diagnostic.db"
            ctx_store = ContextStore(db_path)
            ctx_store.open()

            # Prepare conversation turns as dicts for reply context
            turn_dicts = []
            if visible and visible.conversation_turns:
                turn_dicts = [
                    {"speaker": t.speaker, "text": t.text}
                    for t in visible.conversation_turns
                ]

            visible_texts = visible.text_elements if visible else None
            source_url = visible.url if visible else ""
            window_title = visible.window_title if visible else ""

            modes_to_show = [detected_mode]
            if both_modes:
                other = (AutocompleteMode.REPLY if detected_mode == AutocompleteMode.CONTINUATION
                         else AutocompleteMode.CONTINUATION)
                modes_to_show.append(other)

            for mode in modes_to_show:
                # --- E. Assembled context ---
                try:
                    if mode == AutocompleteMode.CONTINUATION:
                        context = ctx_store.get_continuation_context(
                            before_cursor=before_cursor,
                            after_cursor=after_cursor,
                            source_app=app_name,
                            window_title=window_title,
                            source_url=source_url,
                            visible_text=visible_texts,
                            cross_app_context="",
                        )
                    else:
                        context = ctx_store.get_reply_context(
                            conversation_turns=turn_dicts,
                            source_app=app_name,
                            window_title=window_title,
                            source_url=source_url,
                            draft_text=before_cursor,
                            visible_text=visible_texts,
                            cross_app_context="",
                        )
                    _dump_context(f, context, mode)
                except Exception as e:
                    _section(f, f"E. ASSEMBLED CONTEXT — {mode.name}")
                    f.write(f"  ERROR: {e}\n")
                    context = ""

                # --- F. LLM prompts ---
                try:
                    _dump_prompts(f, context, mode, num_suggestions)
                except Exception as e:
                    _section(f, "F. LLM PROMPTS")
                    f.write(f"  ERROR: {e}\n")

            ctx_store.close()
        else:
            _section(f, "D. MODE DETECTION")
            f.write("  (skipped — no focused element)\n")
            _section(f, "E. ASSEMBLED CONTEXT")
            f.write("  (skipped — no focused element)\n")
            _section(f, "F. LLM PROMPTS")
            f.write("  (skipped — no focused element)\n")

        # --- G. Raw AX tree ---
        try:
            _dump_ax_tree_section(f, max_depth)
        except Exception as e:
            _section(f, f"G. RAW AX TREE (max_depth={max_depth})")
            f.write(f"  ERROR: {e}\n")

        f.write("\n")
        f.flush()
    finally:
        if out_path:
            f.close()

    if out_path:
        print(f"[{timestamp}] Dumped {app_name} -> {out_path}")
    else:
        print(f"\n[{timestamp}] Done dumping {app_name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Dump the full autocompleter pipeline on Ctrl+Space"
    )
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (append mode). Default: stdout")
    parser.add_argument("--depth", type=int, default=12,
                        help="Max AX tree depth (default: 12)")
    parser.add_argument("--num-suggestions", type=int, default=3,
                        help="Number of suggestions for prompt templates (default: 3)")
    parser.add_argument("--both-modes", action="store_true",
                        help="Show context for both CONTINUATION and REPLY modes")
    args = parser.parse_args()

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Pipeline Diagnostic — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Output: {args.output}")

    listener = HotkeyListener()
    listener.register(
        "ctrl+space",
        lambda: on_trigger(args.output, args.depth, args.num_suggestions, args.both_modes),
    )
    listener.start()

    print("Press Ctrl+Space to dump the pipeline for the frontmost app.")
    print("Press Ctrl+C to quit.")

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    try:
        while True:
            AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )
    except KeyboardInterrupt:
        listener.stop()
        print("\nDone.")


if __name__ == "__main__":
    main()
