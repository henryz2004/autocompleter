"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .ax_fixture_loader import load_fixture, load_fixture_metadata

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AX_TREES_DIR = FIXTURES_DIR / "ax_trees"
EXPECTED_DIR = FIXTURES_DIR / "expected"


@pytest.fixture
def ax_tree_fixture():
    """Factory fixture: load an AX tree fixture by name.

    Usage::

        def test_something(ax_tree_fixture):
            root, metadata, expected = ax_tree_fixture("claude_3turn")

    Returns ``(root_element, metadata_dict, expected_dict)``.
    ``expected`` is ``None`` if no expected file exists.
    """
    def _load(name: str):
        ax_path = AX_TREES_DIR / f"{name}.json"
        if not ax_path.exists():
            pytest.skip(f"Fixture not found: {ax_path}")

        root = load_fixture(ax_path)
        metadata = load_fixture_metadata(ax_path)

        expected_path = EXPECTED_DIR / f"{name}.json"
        expected = None
        if expected_path.exists():
            expected = json.loads(expected_path.read_text(encoding="utf-8"))

        return root, metadata, expected

    return _load
