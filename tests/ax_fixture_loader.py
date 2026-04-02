"""Load JSON AX tree fixtures into mock AX elements for testing.

Converts the JSON format produced by ``dump_ax_tree_json.py`` into
``MagicMock`` objects that carry ``_ax_attrs`` dicts, matching the
convention established in ``test_conversation_extractors.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

# Maps JSON keys to AX attribute names
_KEY_TO_AX = {
    "role": "AXRole",
    "subrole": "AXSubrole",
    "roleDescription": "AXRoleDescription",
    "value": "AXValue",
    "title": "AXTitle",
    "description": "AXDescription",
    "placeholderValue": "AXPlaceholderValue",
    "numberOfCharacters": "AXNumberOfCharacters",
}


def _node_to_mock(node: dict) -> MagicMock:
    """Recursively convert a JSON node dict to a MagicMock AX element."""
    elem = MagicMock()

    attrs: dict = {}
    for json_key, ax_key in _KEY_TO_AX.items():
        attrs[ax_key] = node.get(json_key)

    # Convert children: empty list → None (matches make_ax_element convention)
    json_children = node.get("children", [])
    if json_children:
        attrs["AXChildren"] = [_node_to_mock(c) for c in json_children]
    else:
        attrs["AXChildren"] = None

    elem._ax_attrs = attrs
    return elem


def load_fixture(path: str | Path) -> MagicMock:
    """Load a JSON AX tree fixture and return the root mock element.

    Parameters
    ----------
    path : str or Path
        Path to a JSON fixture file (the format written by
        ``dump_ax_tree_json.py``).

    Returns
    -------
    MagicMock
        Root element with ``_ax_attrs`` dict, suitable for use with
        ``_ax_get_attribute_dispatcher``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _node_to_mock(data["tree"])


def load_fixture_metadata(path: str | Path) -> dict:
    """Load just the metadata envelope from a JSON fixture.

    Returns a dict with keys: ``app``, ``windowTitle``, ``capturedAt``,
    ``macosVersion``, ``notes``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "tree"}
