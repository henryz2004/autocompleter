#!/usr/bin/env python3
"""AX tree dumper — press Ctrl+Space to dump the frontmost app's accessibility tree.

Usage:
    source venv/bin/activate
    python dump_ax_tree.py                      # dump to stdout
    python dump_ax_tree.py -o /tmp/ax_dump.log  # dump to file
    python dump_ax_tree.py --depth 20           # increase max depth

Press Ctrl+Space while focused on the target app (ChatGPT, Claude, etc.).
Press Ctrl+C to quit.
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

import AppKit
from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
)
from autocompleter.ax_utils import ax_get_attribute, dump_ax_tree
from autocompleter.hotkey import HotkeyListener


def dump_focused_element(out):
    """Dump details about the currently focused text element."""
    system_wide = AXUIElementCreateSystemWide()
    focused = ax_get_attribute(system_wide, "AXFocusedUIElement")
    if focused is None:
        out.write("  (no focused element)\n")
        return

    attrs_to_check = [
        "AXRole", "AXSubrole", "AXRoleDescription",
        "AXValue", "AXTitle", "AXDescription",
        "AXPlaceholderValue", "AXNumberOfCharacters",
        "AXSelectedText", "AXSelectedTextRange",
    ]
    out.write("  FOCUSED ELEMENT ATTRIBUTES:\n")
    for attr in attrs_to_check:
        val = ax_get_attribute(focused, attr)
        if val is None:
            out.write(f"    {attr}: None (not set)\n")
        elif isinstance(val, str):
            display = val.replace("\n", "\\n")
            if len(display) > 200:
                display = display[:200] + "..."
            out.write(f'    {attr}: "{display}"\n')
        else:
            out.write(f"    {attr}: {val} ({type(val).__name__})\n")

    # Also dump a few levels of children for the focused element
    out.write("  FOCUSED ELEMENT CHILDREN (depth 4):\n")
    children = ax_get_attribute(focused, "AXChildren") or []
    if children:
        for child in children[:20]:
            dump_ax_tree(child, out, max_depth=4, max_children=10, depth=2)
    else:
        out.write("    (no children)\n")


def on_trigger(out_path: str | None, max_depth: int):
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
    timestamp = time.strftime("%H:%M:%S")

    if out_path:
        f = open(out_path, "a")
    else:
        f = sys.stdout

    try:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"[{timestamp}] App: {name} (pid={pid})  Window: {window_title}\n")
        f.write(f"{'=' * 70}\n\n")

        f.write("--- FOCUSED ELEMENT ---\n")
        dump_focused_element(f)
        f.write("\n")

        f.write(f"--- FULL WINDOW TREE (max_depth={max_depth}) ---\n")
        dump_ax_tree(window, f, max_depth=max_depth)
        f.write("\n")

        f.flush()
    finally:
        if out_path:
            f.close()

    if out_path:
        print(f"[{timestamp}] Dumped {name} -> {out_path}")
    else:
        print(f"\n[{timestamp}] Done dumping {name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Dump AX tree on Ctrl+Space")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (append mode). Default: stdout")
    parser.add_argument("--depth", type=int, default=12,
                        help="Max tree depth (default: 12)")
    args = parser.parse_args()

    if args.output:
        # Clear the file
        with open(args.output, "w") as f:
            f.write(f"AX Tree Dump — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Output: {args.output}")

    listener = HotkeyListener()
    listener.register("ctrl+space", lambda: on_trigger(args.output, args.depth))
    listener.start()

    print("Press Ctrl+Space to dump the AX tree of the frontmost app.")
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
