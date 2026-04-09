#!/usr/bin/env python3
"""AX tree JSON dumper — press Ctrl+Space to capture the frontmost app's
accessibility tree as a JSON fixture.

Usage:
    source venv/bin/activate

    # Multi-capture mode (default): auto-names files by app, keeps running
    python dump_ax_tree_json.py
    python dump_ax_tree_json.py -d tests/fixtures/ax_trees --depth 15

    # Single-file mode: saves to a specific path
    python dump_ax_tree_json.py -o tests/fixtures/ax_trees/claude_3turn.json

Press Ctrl+Space while focused on the target app.
Press Ctrl+C to quit.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that coerces non-serializable AX values to strings."""

    def default(self, o):
        # AXValueRef, NSArray, and other pyobjc types → str representation
        try:
            return str(o)
        except Exception:
            return f"<unserializable: {type(o).__name__}>"

sys.path.insert(0, ".")

import AppKit
from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
)
from autocompleter.ax_utils import ax_get_attribute, serialize_ax_tree
from autocompleter.hotkey import HotkeyListener


def _slugify(name: str) -> str:
    """Turn an app name into a filename-safe slug: 'Google Chrome' -> 'google-chrome'."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _next_path(directory: str, slug: str) -> str:
    """Return the next available path: slug.json, slug-2.json, slug-3.json, ..."""
    base = os.path.join(directory, f"{slug}.json")
    if not os.path.exists(base):
        return base
    n = 2
    while True:
        candidate = os.path.join(directory, f"{slug}-{n}.json")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _get_source_url(app_el, app_name: str, window) -> str:
    """Try to extract the page URL for browser / Electron apps."""
    # Safari: AXDocument on window
    if app_name == "Safari":
        doc = ax_get_attribute(window, "AXDocument")
        if isinstance(doc, str):
            return doc

    # Chrome-based: look for address bar in toolbar
    toolbar = _find_by_role(window, "AXToolbar", max_depth=3)
    if toolbar:
        text_field = _find_by_role(toolbar, "AXTextField", max_depth=3)
        if text_field:
            val = ax_get_attribute(text_field, "AXValue")
            if isinstance(val, str) and ("." in val or val.startswith("http")):
                return val
    return ""


def _find_by_role(element, target_role: str, max_depth: int = 3, depth: int = 0):
    """Simple DFS search for an element with a given role."""
    if depth > max_depth:
        return None
    role = ax_get_attribute(element, "AXRole") or ""
    if role == target_role:
        return element
    for child in ax_get_attribute(element, "AXChildren") or []:
        found = _find_by_role(child, target_role, max_depth, depth + 1)
        if found:
            return found
    return None


def on_trigger(out_path: str | None, out_dir: str | None, max_depth: int, notes: str):
    """Called when hotkey is pressed."""
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    if not front_app:
        print("No frontmost app")
        return True

    pid = front_app.processIdentifier()
    name = front_app.localizedName()

    app_el = AXUIElementCreateApplication(pid)
    window = ax_get_attribute(app_el, "AXFocusedWindow")
    if window is None:
        windows = ax_get_attribute(app_el, "AXWindows")
        window = windows[0] if windows else None

    if window is None:
        print(f"No window for {name}")
        return True

    window_title = ax_get_attribute(window, "AXTitle") or ""

    # Capture the focused (keyboard-active) element for subtree walking.
    focused_el = ax_get_attribute(app_el, "AXFocusedUIElement")

    focus_info = ""
    if focused_el is not None:
        fr = ax_get_attribute(focused_el, "AXRole") or "?"
        focus_info = f" focused={fr}"
    print(f"Capturing {name} (pid={pid}) window={window_title!r}{focus_info} ...")
    tree = serialize_ax_tree(
        window, max_depth=max_depth, focused_element=focused_el
    )

    # Build a quick summary of the focused element for the envelope.
    focused_summary = None
    if focused_el is not None:
        focused_summary = {
            "role": ax_get_attribute(focused_el, "AXRole") or "",
            "roleDescription": ax_get_attribute(focused_el, "AXRoleDescription") or "",
            "description": ax_get_attribute(focused_el, "AXDescription") or "",
            "value": v if isinstance((v := ax_get_attribute(focused_el, "AXValue")), str) else None,
            "placeholderValue": (
                pv if isinstance((pv := ax_get_attribute(focused_el, "AXPlaceholderValue")), str) else None
            ),
            "numberOfCharacters": (
                nc if isinstance((nc := ax_get_attribute(focused_el, "AXNumberOfCharacters")), (int, float)) else None
            ),
        }
        # Cursor position from AXSelectedTextRange
        try:
            from ApplicationServices import AXValueGetValue
            sel_range = ax_get_attribute(focused_el, "AXSelectedTextRange")
            if sel_range is not None:
                ok, cf_range = AXValueGetValue(sel_range, 4, None)  # kAXValueTypeCFRange
                if ok:
                    focused_summary["cursorPosition"] = cf_range.location
                    focused_summary["selectionLength"] = cf_range.length
        except Exception:
            pass

    # Try to extract page URL for browser / Electron apps.
    source_url = _get_source_url(app_el, name, window)

    envelope = {
        "app": name,
        "windowTitle": window_title,
        "sourceUrl": source_url,
        "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "macosVersion": platform.mac_ver()[0],
        "notes": notes,
        "focusedElement": focused_summary,
        "tree": tree,
    }

    # Determine output path
    if out_path:
        path = out_path
    else:
        slug = _slugify(name)
        path = _next_path(out_dir, slug)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False, cls=_SafeEncoder)

    print(f"Saved -> {path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Capture AX tree as JSON fixture on Ctrl+Space"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON file path (single-file mode)",
    )
    parser.add_argument(
        "-d", "--dir", type=str, default="tests/fixtures/ax_trees",
        help="Output directory for auto-named files (default: tests/fixtures/ax_trees)",
    )
    parser.add_argument(
        "--depth", type=int, default=30,
        help="Max tree depth (default: 30)",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Free-text notes to embed in the fixture",
    )
    args = parser.parse_args()

    listener = HotkeyListener()
    listener.register(
        "ctrl+space",
        lambda: on_trigger(args.output, args.dir, args.depth, args.notes),
    )
    listener.start()

    if args.output:
        print(f"Output: {args.output}")
    else:
        print(f"Output dir: {args.dir}/")
        print("Files auto-named by app (e.g. google-chrome.json, claude.json)")
    print("Press Ctrl+Space to capture the AX tree of the frontmost app.")
    print("Press Ctrl+C when done.")

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
