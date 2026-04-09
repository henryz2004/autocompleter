#!/usr/bin/env python3
"""Walk UP from AXFocusedUIElement, examining siblings at each level.

Usage: Focus on Gemini input box, then within 5 seconds run:
    python diag_walk_up.py
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")

import AppKit
from ApplicationServices import AXUIElementCreateApplication
from autocompleter.ax_utils import ax_get_attribute


def _node_summary(el, max_value=200) -> str:
    role = ax_get_attribute(el, "AXRole") or "?"
    subrole = ax_get_attribute(el, "AXSubrole") or ""
    title = ax_get_attribute(el, "AXTitle") or ""
    value = ax_get_attribute(el, "AXValue")
    desc = ax_get_attribute(el, "AXDescription") or ""
    n_chars = ax_get_attribute(el, "AXNumberOfCharacters")

    parts = [f"role={role}"]
    if subrole:
        parts.append(f"subrole={subrole}")
    if title:
        parts.append(f"title={title[:80]!r}")
    if isinstance(value, str) and value.strip():
        parts.append(f"value={value[:max_value]!r}")
    if desc:
        parts.append(f"desc={desc[:80]!r}")
    if isinstance(n_chars, (int, float)):
        parts.append(f"nchars={n_chars}")

    children = ax_get_attribute(el, "AXChildren")
    n_children = len(children) if children and hasattr(children, "__len__") and not isinstance(children, (str, bytes)) else 0
    parts.append(f"children={n_children}")

    return " ".join(parts)


def _collect_text_from_subtree(el, max_depth=6, depth=0, budget=[0]):
    """Collect all text from a subtree (DFS), return list of (depth, text)."""
    if depth > max_depth or budget[0] > 500:
        return []
    budget[0] += 1
    results = []

    role = ax_get_attribute(el, "AXRole") or ""
    value = ax_get_attribute(el, "AXValue")
    title = ax_get_attribute(el, "AXTitle") or ""
    desc = ax_get_attribute(el, "AXDescription") or ""

    if isinstance(value, str) and len(value.strip()) > 20:
        results.append((depth, f"[{role}] value: {value[:300]!r}"))
    if len(title) > 20:
        results.append((depth, f"[{role}] title: {title[:300]!r}"))
    if len(desc) > 20:
        results.append((depth, f"[{role}] desc: {desc[:300]!r}"))

    children = ax_get_attribute(el, "AXChildren")
    if children and hasattr(children, "__len__") and not isinstance(children, (str, bytes)):
        for child in list(children)[:50]:
            results.extend(_collect_text_from_subtree(child, max_depth, depth + 1, budget))

    return results


def main():
    print("Waiting 5 seconds — switch to Gemini and click in the input box...")
    time.sleep(5)

    workspace = AppKit.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    if not front_app:
        print("No frontmost app")
        return

    name = front_app.localizedName()
    pid = front_app.processIdentifier()
    print(f"App: {name} (pid={pid})")

    app_el = AXUIElementCreateApplication(pid)
    focused = ax_get_attribute(app_el, "AXFocusedUIElement")
    if focused is None:
        print("No focused element!")
        return

    print(f"\nFocused element: {_node_summary(focused)}")
    print(f"{'=' * 70}")

    # Walk UP from focused element via AXParent
    current = focused
    level = 0
    while current is not None:
        print(f"\n--- Level {level} (walking UP) ---")
        print(f"  THIS: {_node_summary(current)}")

        # Get siblings (parent's children)
        parent = ax_get_attribute(current, "AXParent")
        if parent is not None:
            siblings = ax_get_attribute(parent, "AXChildren")
            if siblings and hasattr(siblings, "__len__") and not isinstance(siblings, (str, bytes)):
                siblings = list(siblings)
                print(f"  Parent has {len(siblings)} children (siblings):")
                for i, sib in enumerate(siblings[:30]):
                    summary = _node_summary(sib, max_value=100)
                    print(f"    [{i}] {summary}")

                    # For each sibling, peek into its text content
                    text_items = _collect_text_from_subtree(sib, max_depth=4, budget=[0])
                    for d, txt in text_items[:5]:
                        print(f"      {'  ' * d}TEXT: {txt[:200]}")
                    if len(text_items) > 5:
                        print(f"      ... and {len(text_items) - 5} more text items")

        current = parent
        level += 1
        if level > 30:
            print("... stopping at level 30")
            break

    print(f"\n{'=' * 70}")
    print(f"Walked {level} levels up from focused element to root.")


if __name__ == "__main__":
    main()
