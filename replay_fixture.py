#!/usr/bin/env python3
"""Replay a saved AX tree fixture through the full autocompleter pipeline.

Runs every stage of the autocomplete pipeline offline against a saved JSON
fixture, so you can iterate on context extraction / prompts / LLM behaviour
without manually triggering in live apps.

Usage:
    source venv/bin/activate

    # Show pipeline stages (no LLM call)
    python replay_fixture.py tests/fixtures/ax_trees/claude-current-chat.json

    # Show both continuation and reply context
    python replay_fixture.py claude.json --both-modes

    # Actually call the LLM and print suggestions
    python replay_fixture.py claude.json --call-llm

    # Override mode (ignore auto-detection)
    python replay_fixture.py claude.json --mode reply

    # Write output to file
    python replay_fixture.py claude.json -o /tmp/replay.log
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# DictNode: make serialized AX tree dicts usable with ax_get_attribute
# ---------------------------------------------------------------------------

# Map from AX attribute names to fixture JSON keys.
_AX_ATTR_MAP: dict[str, str | None] = {
    "AXRole": "role",
    "AXSubrole": "subrole",
    "AXRoleDescription": "roleDescription",
    "AXValue": "value",
    "AXTitle": "title",
    "AXDescription": "description",
    "AXPlaceholderValue": "placeholderValue",
    "AXNumberOfCharacters": "numberOfCharacters",
    "AXChildren": "children",
    # Not available in fixtures:
    "AXSelectedTextRange": None,
    "AXDocument": None,
    "AXPosition": None,
    "AXSize": None,
    "AXFocusedWindow": None,
    "AXFocusedUIElement": None,
    "AXWindows": None,
}


class DictNode:
    """Wraps a fixture dict so it can be passed to code that calls ax_get_attribute."""

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        self._data = data

    def __repr__(self) -> str:
        role = self._data.get("role", "?")
        val = self._data.get("value")
        val_str = f' val="{val[:40]}..."' if isinstance(val, str) and len(val) > 40 else (f' val="{val}"' if val else "")
        return f"DictNode({role}{val_str})"


def _ax_get_attribute_dict(element: Any, attribute: str) -> Any:
    """Drop-in replacement for ax_get_attribute that handles DictNode/dict."""
    if not isinstance(element, (dict, DictNode)):
        # Not a dict — return None (we're in replay mode, no live AX)
        return None

    data = element._data if isinstance(element, DictNode) else element
    key = _AX_ATTR_MAP.get(attribute)

    if key is None:
        return None

    val = data.get(key)

    # Wrap children as DictNode so downstream code works
    if attribute == "AXChildren" and val is not None:
        return [DictNode(c) if isinstance(c, dict) else c for c in val]

    return val


@contextmanager
def patch_ax_for_dicts():
    """Context manager that monkey-patches ax_get_attribute everywhere.

    Patches the function in ax_utils and in every module that imported it,
    so all extractors / observers transparently work with DictNode trees.
    """
    import autocompleter.ax_utils as _ax_mod
    import autocompleter.conversation_extractors as _ext_mod
    import autocompleter.input_observer as _obs_mod

    originals = {
        "ax_utils": _ax_mod.ax_get_attribute,
        "extractors": _ext_mod.ax_get_attribute,
        "observer": _obs_mod.ax_get_attribute,
    }

    _ax_mod.ax_get_attribute = _ax_get_attribute_dict
    _ext_mod.ax_get_attribute = _ax_get_attribute_dict
    _obs_mod.ax_get_attribute = _ax_get_attribute_dict

    try:
        yield
    finally:
        _ax_mod.ax_get_attribute = originals["ax_utils"]
        _ext_mod.ax_get_attribute = originals["extractors"]
        _obs_mod.ax_get_attribute = originals["observer"]


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(path: str) -> dict:
    """Load a JSON fixture file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FocusedElement construction from fixture data
# ---------------------------------------------------------------------------

def build_focused_element(fixture: dict):
    """Build a FocusedElement dataclass from the fixture envelope."""
    from autocompleter.input_observer import FocusedElement

    fe = fixture.get("focusedElement")
    if fe is None:
        # Legacy fixture without focusedElement — try to find the focused
        # node in the tree and construct from that.
        fe = _focused_element_from_tree(fixture.get("tree", {}))
        if fe is None:
            return None

    value = fe.get("value") or ""
    cursor_pos = fe.get("cursorPosition")
    sel_len = fe.get("selectionLength", 0)

    # If no cursor position recorded, place cursor at end of value
    if cursor_pos is None:
        cursor_pos = len(value)

    return FocusedElement(
        app_name=fixture.get("app", "Unknown"),
        app_pid=0,
        role=fe.get("role", "AXTextArea"),
        value=value,
        selected_text=value[cursor_pos:cursor_pos + sel_len] if sel_len else "",
        position=None,
        size=None,
        insertion_point=cursor_pos,
        selection_length=sel_len,
        placeholder_detected=_detect_placeholder(fe, value),
    )


def _focused_element_from_tree(tree: dict) -> dict | None:
    """Walk the tree to find the focused node and extract element info."""
    if tree.get("focused"):
        return {
            "role": tree.get("role", ""),
            "roleDescription": tree.get("roleDescription", ""),
            "description": tree.get("description", ""),
            "value": tree.get("value"),
            "placeholderValue": tree.get("placeholderValue"),
            "numberOfCharacters": tree.get("numberOfCharacters"),
            "cursorPosition": tree.get("cursorPosition"),
            "selectionLength": tree.get("selectionLength", 0),
        }

    for child in tree.get("children", []):
        if child.get("focused") or child.get("ancestorOfFocused"):
            result = _focused_element_from_tree(child)
            if result:
                return result
    return None


def _detect_placeholder(fe: dict, value: str) -> bool:
    """Simple placeholder detection from fixture data."""
    placeholder = fe.get("placeholderValue")
    if placeholder and value == placeholder:
        return True
    num_chars = fe.get("numberOfCharacters")
    if num_chars is not None and num_chars == 0:
        return True
    # Common placeholder values
    if value.strip().lower() in ("reply...", "message", "type a message", "type a message..."):
        return True
    return False


# ---------------------------------------------------------------------------
# Visible text extraction (dict-based, mirrors _collect_text logic)
# ---------------------------------------------------------------------------

# Roles that are considered UI chrome (skip for text extraction).
_CHROME_ROLES = frozenset({
    "AXButton", "AXCheckBox", "AXImage", "AXMenu", "AXMenuBar",
    "AXMenuItem", "AXPopUpButton", "AXProgressIndicator", "AXScrollBar",
    "AXSlider", "AXTabGroup", "AXToolbar", "AXValueIndicator",
    "AXBusyIndicator",
})

# Roles whose value attribute should be captured.
_TEXT_ROLES = frozenset({
    "AXStaticText", "AXTextArea", "AXTextField", "AXHeading",
    "AXLink", "AXParagraph",
})


def collect_text_from_dict(
    node: dict,
    max_depth: int = 15,
    max_items: int = 50,
    depth: int = 0,
) -> list[str]:
    """Extract visible text elements from a serialized AX tree dict.

    Mimics InputObserver._collect_text but works on dict nodes.
    Returns a list of text strings found in the tree.
    """
    results: list[str] = []
    _collect_text_recurse(node, results, max_depth, max_items, depth)
    return results


def _collect_text_recurse(
    node: dict,
    results: list[str],
    max_depth: int,
    max_items: int,
    depth: int,
) -> None:
    if depth > max_depth or len(results) >= max_items:
        return

    role = node.get("role", "")

    # Skip chrome
    if role in _CHROME_ROLES:
        return

    # Extract text from content roles
    if role in _TEXT_ROLES:
        value = node.get("value")
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if len(text) >= 2:  # Skip very short fragments
                results.append(text[:4000])  # Cap individual elements

    # Also capture AXDescription text if it has content
    desc = node.get("description", "")
    if isinstance(desc, str) and len(desc) > 10 and role not in _TEXT_ROLES:
        results.append(desc[:4000])

    # Recurse into children
    for child in node.get("children", []):
        if len(results) >= max_items:
            break
        _collect_text_recurse(child, results, max_depth, max_items, depth + 1)


# ---------------------------------------------------------------------------
# Conversation extraction via patched ax_get_attribute
# ---------------------------------------------------------------------------

def extract_conversations(tree: dict, app_name: str, window_title: str = ""):
    """Run conversation extractors on a fixture tree dict.

    Uses the DictNode + monkey-patch approach so existing extractors
    work transparently on dict data.
    """
    from autocompleter.conversation_extractors import get_extractor

    with patch_ax_for_dicts():
        extractor = get_extractor(app_name, window_title=window_title)
        extractor_name = type(extractor).__name__
        try:
            turns = extractor.extract(DictNode(tree), max_turns=15)
            return turns, extractor_name
        except Exception as e:
            return None, f"{extractor_name} (ERROR: {e})"


# ---------------------------------------------------------------------------
# Subtree context extraction (already works on dicts natively)
# ---------------------------------------------------------------------------

def extract_subtree(tree: dict, token_budget: int = 500) -> str | None:
    """Run the subtree context walker on a fixture tree."""
    from autocompleter.subtree_context import extract_context_from_tree
    return extract_context_from_tree(tree, token_budget=token_budget)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(
    fixture: dict,
    focused,
    visible_text: list[str],
    conversation_turns,
    subtree_context: str | None,
    mode,
    num_suggestions: int = 3,
) -> tuple[str, str, str]:
    """Assemble the LLM context and prompts.

    Uses the same ``build_messages()`` function as the live pipeline so
    prompts stay in sync (including STREAMING_JSON_INSTRUCTION).

    Returns (context, system_prompt, user_prompt).
    """
    from autocompleter.context_store import ContextStore
    from autocompleter.suggestion_engine import AutocompleteMode, build_messages

    app_name = fixture.get("app", "Unknown")
    window_title = fixture.get("windowTitle", "")
    source_url = fixture.get("sourceUrl", "")

    # Ephemeral context store for assembly
    tmp_dir = tempfile.mkdtemp(prefix="replay_fixture_")
    db_path = Path(tmp_dir) / "replay.db"
    ctx_store = ContextStore(db_path)
    ctx_store.open()

    try:
        # Convert conversation turns to dicts
        turn_dicts = []
        if conversation_turns:
            turn_dicts = [
                {"speaker": t.speaker, "text": t.text}
                for t in conversation_turns
            ]

        before_cursor = focused.before_cursor if focused else ""
        after_cursor = focused.after_cursor if focused else ""

        if mode == AutocompleteMode.CONTINUATION:
            context = ctx_store.get_continuation_context(
                before_cursor=before_cursor,
                after_cursor=after_cursor,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                visible_text=visible_text,
                cross_app_context="",
                subtree_context=subtree_context,
                memory_context="",
            )
        else:
            context = ctx_store.get_reply_context(
                conversation_turns=turn_dicts,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=before_cursor,
                visible_text=visible_text,
                cross_app_context="",
                subtree_context=subtree_context,
                memory_context="",
            )

        # Use the same prompt builder as the live streaming pipeline.
        system_prompt, user_prompt = build_messages(
            mode=mode,
            context=context,
            num_suggestions=num_suggestions,
            streaming=True,  # include STREAMING_JSON_INSTRUCTION
            source_app=app_name,
        )
    finally:
        ctx_store.close()

    return context, system_prompt, user_prompt


# ---------------------------------------------------------------------------
# LLM call (optional)
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, mode) -> list[str]:
    """Call the LLM and return suggestions."""
    from autocompleter.config import Config
    from autocompleter.suggestion_engine import AutocompleteMode

    config = Config()

    temp = 0.3 if mode == AutocompleteMode.CONTINUATION else 0.7
    max_tokens = 80 if mode == AutocompleteMode.CONTINUATION else 200

    try:
        import openai
        client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.llm_base_url or None,
        )
        t0 = time.time()
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content or ""

        # Parse JSON suggestions
        suggestions = []
        try:
            data = json.loads(content)
            for s in data.get("suggestions", []):
                text = s.get("text", "") if isinstance(s, dict) else str(s)
                if text.strip():
                    suggestions.append(text.strip())
        except json.JSONDecodeError:
            suggestions = [content.strip()]

        return suggestions, elapsed
    except Exception as e:
        return [f"(LLM error: {e})"], 0.0


# ---------------------------------------------------------------------------
# Pretty-print sections (modeled after dump_pipeline.py)
# ---------------------------------------------------------------------------

def _indent(text: str, prefix: str = "  | ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) if text else prefix + "(empty)"


def _section(f, title: str) -> None:
    f.write(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


def replay(
    fixture_path: str,
    mode_override: str | None = None,
    both_modes: bool = False,
    do_call_llm: bool = False,
    num_suggestions: int = 3,
    out_file=None,
):
    """Run the full replay pipeline and print results."""
    from autocompleter.suggestion_engine import (
        AutocompleteMode,
        MODE_THRESHOLD_CHARS,
        detect_mode,
    )

    f = out_file or sys.stdout
    fixture = load_fixture(fixture_path)

    app_name = fixture.get("app", "Unknown")
    window_title = fixture.get("windowTitle", "")
    source_url = fixture.get("sourceUrl", "")
    tree = fixture.get("tree", {})

    f.write(f"\nReplay: {fixture_path}\n")
    f.write(f"App: {app_name} | Window: {window_title!r}\n")
    if source_url:
        f.write(f"URL: {source_url}\n")
    f.write(f"Captured: {fixture.get('capturedAt', '?')}\n")
    if fixture.get("notes"):
        f.write(f"Notes: {fixture['notes']}\n")

    # --- A. Focused Element ---
    _section(f, "A. FOCUSED ELEMENT")
    focused = build_focused_element(fixture)
    if focused:
        val_display = focused.value.replace("\n", "\\n")
        if len(val_display) > 200:
            val_display = val_display[:200] + "..."
        f.write(f"  Role: {focused.role}\n")
        f.write(f"  Value ({len(focused.value)} chars): \"{val_display}\"\n")

        before = focused.before_cursor
        after = focused.after_cursor
        before_display = before.replace("\n", "\\n")[:200]
        after_display = after.replace("\n", "\\n")[:200]
        f.write(f"  Before cursor ({len(before)} chars): \"{before_display}\"\n")
        f.write(f"  After cursor ({len(after)} chars): \"{after_display}\"\n")
        f.write(f"  Insertion point: {focused.insertion_point} | Selection: {focused.selection_length}\n")
        f.write(f"  Placeholder detected: {focused.placeholder_detected}\n")
    else:
        f.write("  (no focused element found in fixture)\n")

    # --- B. Visible Text Elements ---
    _section(f, "B. VISIBLE TEXT ELEMENTS")
    visible_text = collect_text_from_dict(tree)
    f.write(f"  Count: {len(visible_text)} elements\n")
    for i, elem in enumerate(visible_text):
        display = elem.replace("\n", "\\n")
        if len(display) > 120:
            display = display[:120] + "..."
        f.write(f"  [{i}]: \"{display}\"\n")
        if i >= 29:
            f.write(f"  ... ({len(visible_text) - 30} more)\n")
            break

    # --- C. Conversation Extraction ---
    _section(f, "C. CONVERSATION EXTRACTION")
    turns, extractor_name = extract_conversations(tree, app_name, window_title)
    f.write(f"  Extractor: {extractor_name}\n")
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

    # --- D. Subtree Context ---
    _section(f, "D. SUBTREE CONTEXT")
    subtree = extract_subtree(tree)
    if subtree:
        f.write(f"  Length: {len(subtree)} chars\n")
        f.write(_indent(subtree) + "\n")
    else:
        f.write("  (no focused path found — subtree walker returned None)\n")

    # --- E/F/G: Mode, Context, Prompts ---
    # Even without a focused element, we can still assemble context in
    # reply mode (conversation turns don't require cursor position).
    can_proceed = focused or turns or mode_override
    if can_proceed:
        before_cursor = focused.before_cursor if focused else ""

        # Mode detection
        if mode_override:
            detected_mode = AutocompleteMode[mode_override.upper()]
        elif focused:
            detected_mode = detect_mode(before_cursor=before_cursor)
        elif turns:
            # No focused element but we have conversation turns → reply mode
            detected_mode = AutocompleteMode.REPLY
        else:
            detected_mode = AutocompleteMode.REPLY

        modes_to_show = [detected_mode]
        if both_modes:
            other = (
                AutocompleteMode.REPLY
                if detected_mode == AutocompleteMode.CONTINUATION
                else AutocompleteMode.CONTINUATION
            )
            modes_to_show.append(other)

        for mode in modes_to_show:
            # --- E. Mode Detection ---
            _section(f, f"E. MODE: {mode.name}")
            stripped_len = len(before_cursor.strip())
            f.write(f"  Before cursor stripped length: {stripped_len} chars\n")
            f.write(f"  Threshold: {MODE_THRESHOLD_CHARS} chars\n")
            if mode_override:
                f.write(f"  (overridden via --mode {mode_override})\n")
            if not focused:
                f.write("  (no focused element — using conversation turns for context)\n")

            # --- F. Assembled Context ---
            _section(f, f"F. ASSEMBLED CONTEXT ({mode.name})")
            context, system_prompt, user_prompt = assemble_context(
                fixture, focused, visible_text, turns, subtree,
                mode, num_suggestions,
            )
            f.write(_indent(context) + "\n")

            # --- G. LLM Prompts ---
            _section(f, f"G. SYSTEM PROMPT ({mode.name})")
            f.write(_indent(system_prompt) + "\n")

            _section(f, f"H. USER PROMPT ({mode.name})")
            f.write(_indent(user_prompt) + "\n")

            # --- I. LLM Call (optional) ---
            if do_call_llm:
                _section(f, f"I. LLM SUGGESTIONS ({mode.name})")
                f.write("  Calling LLM...\n")
                f.flush()
                suggestions, elapsed = call_llm(system_prompt, user_prompt, mode)
                f.write(f"  Latency: {elapsed:.3f}s\n")
                f.write(f"  Suggestions ({len(suggestions)}):\n")
                for i, s in enumerate(suggestions):
                    f.write(f"    [{i+1}] {s}\n")
    else:
        _section(f, "E-I. SKIPPED")
        f.write("  (no focused element or conversation turns — cannot assemble context)\n")

    f.write(f"\n{'=' * 60}\n")
    f.write("Done.\n")
    f.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay a saved AX tree fixture through the autocompleter pipeline",
    )
    parser.add_argument(
        "fixture",
        help="Path to the JSON fixture file",
    )
    parser.add_argument(
        "--mode", choices=["continuation", "reply"],
        default=None,
        help="Override auto-detected mode",
    )
    parser.add_argument(
        "--both-modes", action="store_true",
        help="Show context for both CONTINUATION and REPLY modes",
    )
    parser.add_argument(
        "--call-llm", action="store_true",
        help="Actually call the LLM and print suggestions",
    )
    parser.add_argument(
        "--num-suggestions", type=int, default=3,
        help="Number of suggestions (default: 3)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    out_file = None
    if args.output:
        out_file = open(args.output, "w")

    try:
        replay(
            fixture_path=args.fixture,
            mode_override=args.mode,
            both_modes=args.both_modes,
            do_call_llm=args.call_llm,
            num_suggestions=args.num_suggestions,
            out_file=out_file,
        )
    finally:
        if out_file:
            out_file.close()
            print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
