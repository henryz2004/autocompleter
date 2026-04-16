#!/usr/bin/env python3
"""Analyze captured AX fixtures for extractor design and context benchmarking.

Usage:
    source venv/bin/activate

    python analyze_fixture.py tests/fixtures/ax_trees/claude-current-chat.json
    python analyze_fixture.py tests/fixtures/replay_sessions/20260316-143623-gen8-claude.json
    python analyze_fixture.py tests/fixtures/ax_trees --limit 5
    python analyze_fixture.py tests/fixtures/ax_trees/discord.json --json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from autocompleter.fixture_tools import (
    extract_conversation_turns,
    iter_text_nodes,
    load_normalized_fixture,
    summarize_text_containers,
)
from autocompleter.subtree_context import build_context_bundle_from_tree


def count_nodes(node: dict[str, Any]) -> int:
    """Recursively count nodes in a serialized AX tree."""
    return 1 + sum(count_nodes(child) for child in node.get("children", []) or [])


def max_depth(node: dict[str, Any], depth: int = 0) -> int:
    """Return the maximum depth of the serialized AX tree."""
    children = node.get("children", []) or []
    if not children:
        return depth
    return max(max_depth(child, depth + 1) for child in children)


def role_histogram(tree: dict[str, Any], *, limit: int = 12) -> list[tuple[str, int]]:
    """Count the most common roles in the fixture tree."""
    counts: Counter[str] = Counter()

    def _walk(node: dict[str, Any]) -> None:
        counts[str(node.get("role") or "?")] += 1
        for child in node.get("children", []) or []:
            _walk(child)

    _walk(tree)
    return counts.most_common(limit)


def build_analysis(path: Path) -> dict[str, Any]:
    """Build a structured analysis payload for one fixture."""
    fixture = load_normalized_fixture(path)
    tree = fixture.tree
    bundle = build_context_bundle_from_tree(
        tree,
        token_budget=500,
        overview_token_budget=120,
    ) if tree else None
    extracted_turns, extractor_name = extract_conversation_turns(
        tree,
        app_name=fixture.app,
        window_title=fixture.window_title,
    ) if tree else (None, "None")
    text_nodes = iter_text_nodes(tree, max_depth=40) if tree else []
    top_containers = summarize_text_containers(tree, limit=10) if tree else []

    return {
        "path": str(path),
        "artifactType": fixture.artifact_type,
        "app": fixture.app,
        "windowTitle": fixture.window_title,
        "sourceUrl": fixture.source_url,
        "capturedAt": fixture.captured_at,
        "notes": fixture.notes,
        "focused": (
            {
                "role": fixture.focused.role,
                "roleDescription": fixture.focused.role_description,
                "description": fixture.focused.description,
                "cursorPosition": fixture.focused.cursor_position,
                "selectionLength": fixture.focused.selection_length,
                "valueLength": fixture.focused.value_length,
                "placeholderDetected": fixture.focused.placeholder_detected,
                "beforeCursorPreview": fixture.focused.before_cursor[-120:],
                "afterCursorPreview": fixture.focused.after_cursor[:120],
            }
            if fixture.focused is not None
            else None
        ),
        "stats": {
            "totalNodes": count_nodes(tree) if tree else 0,
            "maxDepth": max_depth(tree) if tree else 0,
            "textNodeCount": len(text_nodes),
            "topRoles": role_histogram(tree),
        },
        "extractor": {
            "name": extractor_name,
            "storedTurnCount": len(fixture.conversation_turns),
            "storedTurns": fixture.conversation_turns[:8],
            "extractedTurnCount": len(extracted_turns or []),
            "extractedTurns": [
                {
                    "speaker": turn.speaker,
                    "text": turn.text,
                    **({"timestamp": turn.timestamp} if turn.timestamp else {}),
                }
                for turn in (extracted_turns or [])[:8]
            ],
        },
        "subtree": {
            "overview": bundle.top_down_context if bundle else None,
            "bottomUp": bundle.bottom_up_context if bundle else None,
            "selectionDebug": bundle.selection_debug if bundle else None,
        },
        "topContainers": top_containers,
        "sampleTextNodes": text_nodes[:20],
    }


def format_analysis(analysis: dict[str, Any]) -> str:
    """Pretty-print a structured analysis payload."""
    lines: list[str] = []
    lines.append(f"Fixture: {analysis['path']}")
    lines.append(
        f"App: {analysis['app']} | Window: {analysis['windowTitle']!r} | "
        f"Artifact: {analysis['artifactType']}"
    )
    if analysis.get("sourceUrl"):
        lines.append(f"URL: {analysis['sourceUrl']}")
    if analysis.get("capturedAt"):
        lines.append(f"Captured: {analysis['capturedAt']}")
    if analysis.get("notes"):
        lines.append(f"Notes: {analysis['notes']}")

    focused = analysis.get("focused")
    lines.append("")
    lines.append("Focused element")
    if focused is None:
        lines.append("  (none recorded)")
    else:
        lines.append(
            "  "
            f"role={focused['role']} | cursor={focused['cursorPosition']} | "
            f"selection={focused['selectionLength']} | "
            f"value_length={focused['valueLength']} | "
            f"placeholder_detected={focused['placeholderDetected']}"
        )
        if focused.get("description"):
            lines.append(f"  desc={focused['description']}")
        if focused.get("beforeCursorPreview"):
            lines.append(f"  before={focused['beforeCursorPreview']!r}")
        if focused.get("afterCursorPreview"):
            lines.append(f"  after={focused['afterCursorPreview']!r}")

    stats = analysis["stats"]
    lines.append("")
    lines.append("Tree stats")
    lines.append(
        f"  total_nodes={stats['totalNodes']} | max_depth={stats['maxDepth']} | "
        f"text_nodes={stats['textNodeCount']}"
    )
    lines.append("  top_roles=" + ", ".join(f"{role}:{count}" for role, count in stats["topRoles"]))

    extractor = analysis["extractor"]
    lines.append("")
    lines.append("Extractor")
    lines.append(
        f"  selected={extractor['name']} | stored_turns={extractor['storedTurnCount']} | "
        f"extracted_turns={extractor['extractedTurnCount']}"
    )
    for index, turn in enumerate(extractor["extractedTurns"]):
        preview = turn["text"].replace("\n", "\\n")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        lines.append(f"  [{index}] {turn['speaker']}: {preview}")

    subtree = analysis["subtree"]
    lines.append("")
    lines.append("Subtree context")
    if subtree["selectionDebug"] is not None:
        strategy = subtree["selectionDebug"].get("strategy")
        lines.append(f"  strategy={strategy}")
    if subtree["overview"]:
        lines.append("  overview:")
        lines.extend(f"    {line}" for line in subtree["overview"].splitlines())
    else:
        lines.append("  overview: (none)")
    if subtree["bottomUp"]:
        lines.append("  bottom_up:")
        lines.extend(f"    {line}" for line in subtree["bottomUp"].splitlines())
    else:
        lines.append("  bottom_up: (none)")

    lines.append("")
    lines.append("Top text-rich containers")
    for item in analysis["topContainers"]:
        preview = item["preview"] or item["description"] or item["title"] or ""
        if len(preview) > 120:
            preview = preview[:117] + "..."
        lines.append(
            "  "
            f"{item['path']} | role={item['role']} | text_nodes={item['text_nodes']} | "
            f"text_chars={item['text_chars']} | preview={preview!r}"
        )

    lines.append("")
    lines.append("Sample text nodes")
    for item in analysis["sampleTextNodes"]:
        lines.append(
            "  "
            f"{item['path']} | role={item['role']} | {item['text_preview']}"
        )

    return "\n".join(lines)


def discover_paths(target: Path, *, limit: int | None = None) -> list[Path]:
    """Expand a file or directory into a list of fixture paths."""
    if target.is_file():
        return [target]
    paths = sorted(target.glob("*.json"))
    if limit is not None:
        return paths[:limit]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze AX fixtures for custom extractor design and context benchmarking",
    )
    parser.add_argument("target", help="Fixture file or directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of fixtures when TARGET is a directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary",
    )
    args = parser.parse_args()

    target = Path(args.target)
    paths = discover_paths(target, limit=args.limit)
    analyses = [build_analysis(path) for path in paths]

    if args.json:
        print(json.dumps(analyses if len(analyses) > 1 else analyses[0], indent=2, ensure_ascii=False))
        return

    for index, analysis in enumerate(analyses):
        if index:
            print("\n" + "=" * 100 + "\n")
        print(format_analysis(analysis))


if __name__ == "__main__":
    main()
