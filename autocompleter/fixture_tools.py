"""Shared helpers for captured AX fixtures and replay artifacts.

Normalizes the two fixture families we use today:

- Raw AX tree captures from ``dump_ax_tree_json.py``
- Trigger/replay artifacts from ``TriggerDumper``

The goal is to let analysis tools, replay scripts, and regression tests all
work from one consistent in-memory shape even if the on-disk envelope differs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizedFocusedData:
    """Focused element metadata normalized across fixture formats."""

    role: str = ""
    role_description: str = ""
    description: str = ""
    value: str = ""
    placeholder_value: str = ""
    number_of_characters: int | None = None
    cursor_position: int | None = None
    selection_length: int = 0
    before_cursor: str = ""
    after_cursor: str = ""
    value_length: int = 0
    placeholder_detected: bool = False


@dataclass(frozen=True)
class NormalizedFixture:
    """Canonical in-memory representation of a captured fixture/dump."""

    path: Path | None
    artifact_type: str
    app: str
    window_title: str
    source_url: str
    captured_at: str
    notes: str
    tree: dict[str, Any]
    focused: NormalizedFocusedData | None
    detection: dict[str, Any]
    conversation_turns: list[dict[str, str]]
    context: str
    context_inputs: dict[str, Any]
    suggestions: list[str]
    raw: dict[str, Any]


def load_normalized_fixture(path: str | Path) -> NormalizedFixture:
    """Load and normalize a fixture file from disk."""
    fixture_path = Path(path)
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    return normalize_fixture(data, path=fixture_path)


def normalize_fixture(
    data: dict[str, Any],
    *,
    path: Path | None = None,
) -> NormalizedFixture:
    """Normalize a raw fixture envelope into a stable shape."""
    focused = _normalize_focused_payload(data)
    artifact_type = (
        data.get("artifactType")
        or _infer_artifact_type(data)
    )
    source_url = (
        data.get("sourceUrl")
        or (data.get("focused") or {}).get("sourceUrl")
        or ""
    )
    conversation_turns = [
        _normalize_turn(turn)
        for turn in data.get("conversationTurns", []) or []
        if isinstance(turn, dict)
    ]
    suggestions = [
        str(s)
        for s in data.get("suggestions", []) or []
    ]

    return NormalizedFixture(
        path=path,
        artifact_type=artifact_type,
        app=str(data.get("app") or "Unknown"),
        window_title=str(data.get("windowTitle") or ""),
        source_url=str(source_url),
        captured_at=str(data.get("capturedAt") or ""),
        notes=str(data.get("notes") or ""),
        tree=data.get("tree") or {},
        focused=focused,
        detection=dict(data.get("detection") or {}),
        conversation_turns=conversation_turns,
        context=str(data.get("context") or ""),
        context_inputs=dict(data.get("contextInputs") or {}),
        suggestions=suggestions,
        raw=data,
    )


def build_focused_element(normalized: NormalizedFixture):
    """Build a ``FocusedElement`` dataclass from a normalized fixture."""
    if normalized.focused is None:
        return None

    from autocompleter.input_observer import FocusedElement

    focused = normalized.focused
    value = focused.value
    cursor_position = focused.cursor_position
    selection_length = focused.selection_length

    if cursor_position is None:
        cursor_position = len(value)

    return FocusedElement(
        app_name=normalized.app,
        app_pid=0,
        role=focused.role or "AXTextArea",
        value=value,
        selected_text=(
            value[cursor_position: cursor_position + selection_length]
            if selection_length
            else ""
        ),
        position=None,
        size=None,
        insertion_point=cursor_position,
        selection_length=selection_length,
        placeholder_detected=focused.placeholder_detected,
        raw_value=value,
        raw_placeholder_value=focused.placeholder_value,
        raw_number_of_characters=focused.number_of_characters,
    )


def iter_text_nodes(
    tree: dict[str, Any],
    *,
    max_depth: int = 40,
) -> list[dict[str, Any]]:
    """Return a flat list of text-bearing nodes with depth/path metadata."""
    results: list[dict[str, Any]] = []

    def _walk(node: dict[str, Any], depth: int, path_bits: list[str]) -> None:
        if depth > max_depth:
            return

        text_fields = _text_fields(node)
        if text_fields:
            results.append(
                {
                    "depth": depth,
                    "path": " > ".join(path_bits),
                    "role": node.get("role", ""),
                    "subrole": node.get("subrole", ""),
                    "title": node.get("title", ""),
                    "description": node.get("description", ""),
                    "value": node.get("value", ""),
                    "text_fields": text_fields,
                    "text_preview": _text_preview(text_fields),
                }
            )

        for index, child in enumerate(node.get("children", []) or []):
            child_role = child.get("role", "?")
            _walk(child, depth + 1, [*path_bits, f"{child_role}[{index}]"])

    root_role = tree.get("role", "?")
    _walk(tree, 0, [root_role])
    return results


def summarize_text_containers(
    tree: dict[str, Any],
    *,
    limit: int = 12,
    max_depth: int = 40,
) -> list[dict[str, Any]]:
    """Rank nodes by how much descendant text they contain.

    This is intended for extractor design work: it highlights likely transcript
    containers or message groups without requiring app-specific logic.
    """
    candidates: list[dict[str, Any]] = []

    def _walk(node: dict[str, Any], depth: int, path_bits: list[str]) -> tuple[int, int]:
        if depth > max_depth:
            return 0, 0

        node_text = _joined_node_text(node)
        text_chars = len(node_text)
        text_nodes = 1 if node_text else 0

        for index, child in enumerate(node.get("children", []) or []):
            child_role = child.get("role", "?")
            child_nodes, child_chars = _walk(
                child,
                depth + 1,
                [*path_bits, f"{child_role}[{index}]"],
            )
            text_nodes += child_nodes
            text_chars += child_chars

        if text_nodes:
            candidates.append(
                {
                    "depth": depth,
                    "path": " > ".join(path_bits),
                    "role": node.get("role", ""),
                    "subrole": node.get("subrole", ""),
                    "role_description": node.get("roleDescription", ""),
                    "title": node.get("title", ""),
                    "description": node.get("description", ""),
                    "text_nodes": text_nodes,
                    "text_chars": text_chars,
                    "preview": _preview_from_node(node),
                }
            )

        return text_nodes, text_chars

    root_role = tree.get("role", "?")
    _walk(tree, 0, [root_role])
    candidates.sort(
        key=lambda item: (
            -item["text_chars"],
            -item["text_nodes"],
            -item["depth"],
        )
    )
    return candidates[:limit]


class DictNode:
    """Wrap a fixture dict so AX helpers can treat it like a live element."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __repr__(self) -> str:
        role = self._data.get("role", "?")
        val = self._data.get("value")
        if isinstance(val, str) and val:
            preview = val[:40] + ("..." if len(val) > 40 else "")
            return f'DictNode({role} val="{preview}")'
        return f"DictNode({role})"


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
    "AXVisibleChildren": "children",
    "AXSelectedTextRange": None,
    "AXDocument": None,
    "AXPosition": None,
    "AXSize": None,
    "AXFocusedWindow": None,
    "AXFocusedUIElement": None,
    "AXWindows": None,
}


def ax_get_attribute_from_fixture(element: Any, attribute: str) -> Any:
    """AX attribute shim for serialized fixture trees."""
    if not isinstance(element, (dict, DictNode)):
        return None

    data = element._data if isinstance(element, DictNode) else element
    key = _AX_ATTR_MAP.get(attribute)
    if key is None:
        return None

    value = data.get(key)
    if attribute in {"AXChildren", "AXVisibleChildren"} and value is not None:
        return [
            DictNode(child) if isinstance(child, dict) else child
            for child in value
        ]
    return value


@contextmanager
def patch_ax_for_fixture_dicts():
    """Patch AX accessors so extractors can run against serialized fixtures."""
    import autocompleter.ax_utils as ax_utils_mod
    import autocompleter.conversation_extractors as extractors_mod
    import autocompleter.input_observer as observer_mod

    originals = {
        "ax_utils": ax_utils_mod.ax_get_attribute,
        "extractors": extractors_mod.ax_get_attribute,
        "observer": observer_mod.ax_get_attribute,
    }

    ax_utils_mod.ax_get_attribute = ax_get_attribute_from_fixture
    extractors_mod.ax_get_attribute = ax_get_attribute_from_fixture
    observer_mod.ax_get_attribute = ax_get_attribute_from_fixture
    try:
        yield
    finally:
        ax_utils_mod.ax_get_attribute = originals["ax_utils"]
        extractors_mod.ax_get_attribute = originals["extractors"]
        observer_mod.ax_get_attribute = originals["observer"]


def extract_conversation_turns(
    tree: dict[str, Any],
    *,
    app_name: str,
    window_title: str = "",
    max_turns: int = 15,
):
    """Run the current extractor selection + extraction logic on a fixture tree."""
    from autocompleter.conversation_extractors import get_extractor

    with patch_ax_for_fixture_dicts():
        extractor = get_extractor(app_name, window_title=window_title)
        extractor_name = type(extractor).__name__
        turns = extractor.extract(DictNode(tree), max_turns=max_turns)
    return turns, extractor_name


def _infer_artifact_type(data: dict[str, Any]) -> str:
    if "focusedElement" in data:
        return "ax_tree_fixture_v0"
    if "tree" in data and "context" not in data:
        return "ax_tree_fixture_v0"
    if "focused" in data and "context" in data:
        return "manual_invocation_v0"
    return "unknown_fixture"


def _normalize_focused_payload(data: dict[str, Any]) -> NormalizedFocusedData | None:
    focused_data = data.get("focusedElement")
    if focused_data is None:
        focused_data = _focused_from_tree(data.get("tree") or {})
    replay_focused = data.get("focused") or {}

    if focused_data is None and not replay_focused:
        return None

    raw_role = (
        focused_data.get("role")
        if focused_data
        else replay_focused.get("role", "")
    ) or replay_focused.get("role", "")
    role_description = (
        focused_data.get("roleDescription")
        if focused_data
        else ""
    ) or ""
    description = (
        focused_data.get("description")
        if focused_data
        else ""
    ) or ""
    value = (
        focused_data.get("value")
        if focused_data and focused_data.get("value") is not None
        else ""
    )
    if not isinstance(value, str):
        value = ""

    placeholder_value = (
        focused_data.get("placeholderValue")
        if focused_data
        else ""
    ) or ""
    number_of_characters = (
        focused_data.get("numberOfCharacters")
        if focused_data is not None
        else replay_focused.get("rawNumberOfCharacters")
    )
    cursor_position = (
        focused_data.get("cursorPosition")
        if focused_data is not None
        else replay_focused.get("insertionPoint")
    )
    selection_length = (
        focused_data.get("selectionLength")
        if focused_data is not None
        else replay_focused.get("selectionLength", 0)
    ) or 0
    value_length = int(
        replay_focused.get("valueLength")
        or len(value)
    )
    before_cursor = replay_focused.get("beforeCursor", "")
    after_cursor = replay_focused.get("afterCursor", "")

    if not value and (before_cursor or after_cursor):
        value = str(before_cursor) + str(after_cursor)

    if cursor_position is None:
        cursor_position = len(str(before_cursor)) if before_cursor else len(value)
    if not before_cursor and isinstance(cursor_position, int):
        before_cursor = value[:cursor_position]
    if not after_cursor and isinstance(cursor_position, int):
        after_cursor = value[cursor_position + selection_length:]

    placeholder_detected = bool(
        replay_focused.get("placeholderDetected")
        or _detect_placeholder(value, placeholder_value, number_of_characters)
    )

    return NormalizedFocusedData(
        role=str(raw_role or ""),
        role_description=str(role_description),
        description=str(description),
        value=value,
        placeholder_value=str(placeholder_value),
        number_of_characters=(
            int(number_of_characters)
            if isinstance(number_of_characters, (int, float))
            else None
        ),
        cursor_position=(
            int(cursor_position)
            if isinstance(cursor_position, (int, float))
            else None
        ),
        selection_length=int(selection_length),
        before_cursor=str(before_cursor),
        after_cursor=str(after_cursor),
        value_length=value_length,
        placeholder_detected=placeholder_detected,
    )


def _focused_from_tree(tree: dict[str, Any]) -> dict[str, Any] | None:
    if not tree:
        return None
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
    for child in tree.get("children", []) or []:
        if child.get("focused") or child.get("ancestorOfFocused"):
            found = _focused_from_tree(child)
            if found is not None:
                return found
    return None


def _normalize_turn(turn: dict[str, Any]) -> dict[str, str]:
    normalized = {
        "speaker": str(turn.get("speaker") or "Unknown"),
        "text": str(turn.get("text") or ""),
    }
    if turn.get("timestamp"):
        normalized["timestamp"] = str(turn["timestamp"])
    return normalized


def _detect_placeholder(
    value: str,
    placeholder_value: str,
    number_of_characters: int | None,
) -> bool:
    if placeholder_value and value == placeholder_value:
        return True
    if number_of_characters == 0:
        return True
    stripped = value.strip().lower()
    return stripped in {
        "reply...",
        "message",
        "type a message",
        "type a message...",
    }


def _text_fields(node: dict[str, Any]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for key in ("value", "title", "description", "placeholderValue"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            fields.append((key, value.strip()))
    return fields


def _joined_node_text(node: dict[str, Any]) -> str:
    return " | ".join(text for _, text in _text_fields(node))


def _text_preview(text_fields: list[tuple[str, str]], max_chars: int = 160) -> str:
    joined = " | ".join(f"{key}={text}" for key, text in text_fields)
    if len(joined) > max_chars:
        return joined[: max_chars - 3] + "..."
    return joined


def _preview_from_node(node: dict[str, Any], max_chars: int = 120) -> str:
    preview = _joined_node_text(node)
    if not preview:
        return ""
    if len(preview) > max_chars:
        return preview[: max_chars - 3] + "..."
    return preview
