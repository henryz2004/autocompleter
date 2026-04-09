#!/usr/bin/env python3
"""
Analyze AX tree fixtures to understand content structure for conversation extraction.

For each fixture, produces a "minimal parse" that:
- Walks the tree depth-first
- Skips nodes with no text content (empty/null title, description, value)
- Shows text-bearing nodes with: depth, role, and text (truncated to 120 chars)
- Preserves structural relationships
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def has_text_content(node: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if node has text content. Returns (has_content, text_string)."""
    texts = []

    # Check value
    if node.get("value") and str(node["value"]).strip():
        texts.append(("value", str(node["value"]).strip()))

    # Check title
    if node.get("title") and str(node["title"]).strip():
        texts.append(("title", str(node["title"]).strip()))

    # Check description
    if node.get("description") and str(node["description"]).strip():
        texts.append(("description", str(node["description"]).strip()))

    if not texts:
        return False, ""

    # Format as "field: content" if single field, otherwise combine
    if len(texts) == 1:
        field, content = texts[0]
        return True, content
    else:
        combined = " | ".join(f"{field}={content[:40]}" for field, content in texts)
        return True, combined


def minimal_parse(node: Dict[str, Any], depth: int = 0, lines: List[str] = None) -> List[str]:
    """
    Recursively parse tree, collecting text-bearing nodes.
    Returns list of formatted lines.
    """
    if lines is None:
        lines = []

    role = node.get("role", "?")
    has_text, text = has_text_content(node)

    if has_text:
        # Truncate text to 120 chars
        if len(text) > 120:
            text = text[:117] + "..."

        # Format: indent + role + text
        indent = "  " * depth
        lines.append(f"{indent}{role}: {text}")

    # Always recurse into children (even if parent has no text)
    children = node.get("children", [])
    for child in children:
        minimal_parse(child, depth + 1, lines)

    return lines


def analyze_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a single AX tree fixture file."""
    with open(filepath) as f:
        data = json.load(f)

    tree = data.get("tree", {})
    app = data.get("app", "?")
    window_title = data.get("windowTitle", "?")

    # Get minimal parse
    lines = minimal_parse(tree)

    # Count total nodes (for comparison)
    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node.get("children", []))

    total_nodes = count_nodes(tree)

    # Estimate tree depth
    def max_depth(node, d=0):
        children = node.get("children", [])
        if not children:
            return d
        return max(max_depth(child, d + 1) for child in children)

    depth = max_depth(tree)

    return {
        "app": app,
        "window_title": window_title,
        "total_nodes": total_nodes,
        "max_depth": depth,
        "text_bearing_lines": len(lines),
        "minimal_parse": lines
    }


def main():
    base_path = Path(__file__).parent / "tests/fixtures/ax_trees"

    files = [
        # Priority 1 - New apps
        ("messages.json", "iMessage - Jinpaaaa conversation"),
        ("messages-2.json", "iMessage - Emma Zhang"),
        ("messages-3.json", "iMessage - Group chat"),
        ("microsoft-edge.json", "Edge/Gmail (MASSIVE - 3033 nodes)"),
        # Priority 2 - ChatGPT with content
        ("chatgpt-6.json", "ChatGPT - 30KB"),
        ("chatgpt-7.json", "ChatGPT - 13KB with text"),
    ]

    print("=" * 80)
    print("AX TREE FIXTURE ANALYSIS")
    print("=" * 80)
    print()

    for filename, description in files:
        filepath = base_path / filename
        if not filepath.exists():
            print(f"SKIPPED: {filename} (not found)")
            print()
            continue

        print(f"FILE: {filename}")
        print(f"DESC: {description}")
        print("-" * 80)

        result = analyze_file(filepath)

        print(f"App: {result['app']}")
        print(f"Window: {result['window_title'][:80]}")
        print(f"Total nodes: {result['total_nodes']}")
        print(f"Max depth: {result['max_depth']}")
        print(f"Text-bearing lines: {result['text_bearing_lines']}")
        print()

        lines = result['minimal_parse']

        # Special handling for microsoft-edge.json (massive file)
        if filename == "microsoft-edge.json":
            print("MINIMAL PARSE (first 30 lines):")
            for line in lines[:30]:
                print(line)
            print()
            print(f"... ({len(lines) - 60} lines omitted) ...")
            print()
            print("MINIMAL PARSE (depths 25+, first 30 lines):")
            deep_lines = [l for l in lines if l.startswith("  " * 25)]
            for line in deep_lines[:30]:
                print(line)
            print()
            if len(deep_lines) > 30:
                print(f"... ({len(deep_lines) - 30} more deep lines omitted) ...")
        else:
            print("MINIMAL PARSE:")
            for line in lines:
                print(line)
            print()

        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
