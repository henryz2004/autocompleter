#!/usr/bin/env python3
"""Diagnostic: compare AXChildren vs AXVisibleChildren for the frontmost app.

Usage: Focus on Gemini first, then run:
    python diag_gemini_ax.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

import AppKit
from ApplicationServices import AXUIElementCreateApplication
from autocompleter.ax_utils import ax_get_attribute


def _count_nodes(element, use_visible: bool, max_depth: int = 25, depth: int = 0):
    """Count nodes and collect interesting elements."""
    if depth > max_depth:
        return 0, []

    count = 1
    interesting = []

    role = ax_get_attribute(element, "AXRole") or ""
    title = ax_get_attribute(element, "AXTitle") or ""
    value = ax_get_attribute(element, "AXValue")
    value_str = value if isinstance(value, str) else ""
    desc = ax_get_attribute(element, "AXDescription") or ""
    subrole = ax_get_attribute(element, "AXSubrole") or ""

    # Collect interesting nodes
    if role == "AXHeading":
        interesting.append(f"  [d={depth}] HEADING title={title!r}")
    if role == "AXStaticText" and len(value_str) > 30:
        snippet = value_str[:100].replace("\n", "\\n")
        interesting.append(f"  [d={depth}] TEXT({len(value_str)}ch): {snippet!r}")
    if subrole == "AXContentList":
        kids_all = ax_get_attribute(element, "AXChildren")
        n_all = len(kids_all) if kids_all and hasattr(kids_all, "__len__") and not isinstance(kids_all, (str, bytes)) else 0
        interesting.append(f"  [d={depth}] CONTENT_LIST: children={n_all}")
    if subrole == "AXLandmarkMain":
        interesting.append(f"  [d={depth}] LANDMARK_MAIN")

    # Get children
    attr_name = "AXVisibleChildren" if use_visible else "AXChildren"
    children = ax_get_attribute(element, attr_name)
    if not children or not hasattr(children, "__len__") or isinstance(children, (str, bytes)):
        if use_visible:
            # fallback to AXChildren if AXVisibleChildren empty
            children = ax_get_attribute(element, "AXChildren")
            if not children or not hasattr(children, "__len__") or isinstance(children, (str, bytes)):
                children = []
            else:
                children = list(children)
        else:
            children = []
    else:
        children = list(children)

    for child in children[:100]:
        child_count, child_interesting = _count_nodes(child, use_visible, max_depth, depth + 1)
        count += child_count
        interesting.extend(child_interesting)

    return count, interesting


def main():
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    if not front_app:
        print("No frontmost app")
        return

    name = front_app.localizedName()
    pid = front_app.processIdentifier()
    app_el = AXUIElementCreateApplication(pid)
    window = ax_get_attribute(app_el, "AXFocusedWindow")
    if window is None:
        windows = ax_get_attribute(app_el, "AXWindows")
        window = windows[0] if windows else None
    if window is None:
        print(f"No window for {name}")
        return

    title = ax_get_attribute(window, "AXTitle") or ""
    print(f"App: {name} | Window: {title!r}")
    print(f"{'='*70}")

    # AXVisibleChildren first (fast)
    print(f"\n--- AXVisibleChildren (current approach) ---")
    vis_count, vis_interesting = _count_nodes(window, use_visible=True)
    print(f"Total nodes: {vis_count}")
    for item in vis_interesting[:30]:
        print(item)
    if not vis_interesting:
        print("  (none found)")

    # AXChildren (potentially much larger)
    print(f"\n--- AXChildren (old approach) ---")
    all_count, all_interesting = _count_nodes(window, use_visible=False)
    print(f"Total nodes: {all_count}")
    for item in all_interesting[:30]:
        print(item)
    if not all_interesting:
        print("  (none found)")

    print(f"\n{'='*70}")
    print(f"AXVisibleChildren: {vis_count} nodes, {len(vis_interesting)} interesting")
    print(f"AXChildren:        {all_count} nodes, {len(all_interesting)} interesting")
    if all_count > vis_count:
        print(f"*** AXChildren has {all_count - vis_count} MORE nodes ({(all_count/max(vis_count,1) - 1)*100:.0f}% more)")


if __name__ == "__main__":
    main()
