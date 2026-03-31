"""Generate golden expected output files from captured AX tree fixtures.

Usage:
    python -m tests.generate_expected tests/fixtures/ax_trees/claude_3turn.json
    python -m tests.generate_expected --all   # regenerate all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autocompleter.conversation_extractors import get_extractor

from tests.ax_fixture_loader import load_fixture, load_fixture_metadata

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AX_TREES_DIR = FIXTURES_DIR / "ax_trees"
EXPECTED_DIR = FIXTURES_DIR / "expected"


def _ax_get_attribute_dispatcher(element, attribute):
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get(attribute)
    return None


def _mock_collect_child_text(element, max_depth=5, max_chars=2000, depth=0):
    if depth > max_depth:
        return ""

    role = _ax_get_attribute_dispatcher(element, "AXRole") or ""

    if role == "AXStaticText":
        value = _ax_get_attribute_dispatcher(element, "AXValue")
        if isinstance(value, str) and value.strip():
            return value.strip()
        desc = _ax_get_attribute_dispatcher(element, "AXDescription")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        return ""

    value = _ax_get_attribute_dispatcher(element, "AXValue")
    if isinstance(value, str) and value.strip():
        return value.strip()

    children = _ax_get_attribute_dispatcher(element, "AXChildren")
    if not children:
        return ""

    parts = []
    for child in children:
        child_text = _mock_collect_child_text(child, max_depth, max_chars, depth + 1)
        if child_text:
            parts.append(child_text)
    return "\n".join(parts)


def generate_expected(ax_tree_path: Path) -> Path:
    """Generate a golden expected file from a captured AX tree fixture.

    Returns the path to the written expected file.
    """
    root = load_fixture(ax_tree_path)
    metadata = load_fixture_metadata(ax_tree_path)

    app_name = metadata.get("app", "")
    extractor = get_extractor(app_name)
    extractor_class = type(extractor).__name__

    with (
        patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_get_attribute_dispatcher,
        ),
        patch(
            "autocompleter.conversation_extractors._collect_child_text",
            side_effect=_mock_collect_child_text,
        ),
    ):
        turns = extractor.extract(root) or []

    result = {
        "extractor": extractor_class,
        "app": app_name,
        "turns": [
            {
                "speaker": t.speaker, "text": t.text,
                **({"timestamp": t.timestamp} if t.timestamp else {}),
            }
            for t in turns
        ],
    }

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPECTED_DIR / f"{ax_tree_path.stem}.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Generated: {out_path}")
    print(f"  Extractor: {extractor_class}")
    print(f"  App: {app_name}")
    print(f"  Turns: {len(turns)}")
    for i, t in enumerate(turns):
        text_preview = t.text[:80].replace("\n", "\\n")
        print(f"    [{i}] {t.speaker}: {text_preview}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden expected files from AX tree fixtures"
    )
    parser.add_argument(
        "fixture",
        nargs="?",
        help="Path to a specific AX tree fixture JSON file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate expected files for all fixtures in ax_trees/",
    )
    args = parser.parse_args()

    if args.all:
        if not AX_TREES_DIR.is_dir():
            print(f"No fixtures directory: {AX_TREES_DIR}")
            sys.exit(1)
        fixtures = sorted(AX_TREES_DIR.glob("*.json"))
        if not fixtures:
            print(f"No fixtures found in {AX_TREES_DIR}")
            sys.exit(1)
        for f in fixtures:
            print(f"\n{'=' * 60}")
            generate_expected(f)
    elif args.fixture:
        path = Path(args.fixture)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        generate_expected(path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
