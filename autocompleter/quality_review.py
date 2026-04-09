"""Helpers for fixture-driven autocomplete quality review.

This module keeps offline review harness logic and live context/prompt
cleanup aligned so we can iterate on captured manual invocations and
promote the winning generic variant into the real app path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class QualityVariant:
    name: str
    prompt_placeholder_aware: bool = False
    strip_cross_app: bool = False
    strip_semantic: bool = False
    reduce_nearby: bool = False
    prefer_cursor_only_when_strong_prefix: bool = False
    continuation_temperature: float | None = None


BASELINE_VARIANT = QualityVariant(name="baseline")
PROMPT_ONLY_VARIANT = QualityVariant(
    name="prompt_placeholder_aware",
    prompt_placeholder_aware=True,
)
NO_CROSS_APP_VARIANT = QualityVariant(
    name="no_cross_app",
    prompt_placeholder_aware=True,
    strip_cross_app=True,
)
NO_SEMANTIC_VARIANT = QualityVariant(
    name="no_semantic",
    prompt_placeholder_aware=True,
    strip_semantic=True,
)
REDUCED_NEARBY_VARIANT = QualityVariant(
    name="reduced_nearby",
    prompt_placeholder_aware=True,
    reduce_nearby=True,
)
COMBINED_VARIANT = QualityVariant(
    name="combined_candidate",
    prompt_placeholder_aware=True,
    strip_cross_app=True,
    strip_semantic=True,
    reduce_nearby=True,
    prefer_cursor_only_when_strong_prefix=True,
    continuation_temperature=0.35,
)

REVIEW_VARIANTS: tuple[QualityVariant, ...] = (
    BASELINE_VARIANT,
    PROMPT_ONLY_VARIANT,
    NO_CROSS_APP_VARIANT,
    NO_SEMANTIC_VARIANT,
    REDUCED_NEARBY_VARIANT,
    COMBINED_VARIANT,
)

# Live defaults chosen after fixture review.
# Continuation quality was best with placeholder-aware prompting, the new
# visible-text fallback, and cross-app pollution removed, without the more
# aggressive context stripping used by the combined ablation.
LIVE_CONTINUATION_VARIANT = QualityVariant(
    name="continuation_live_reviewed",
    prompt_placeholder_aware=True,
    strip_cross_app=True,
)
LIVE_REPLY_VARIANT = QualityVariant(
    name="reply_live_reviewed",
    prompt_placeholder_aware=True,
    strip_cross_app=True,
    strip_semantic=True,
)


def load_valid_invocation_artifacts(paths: list[str | Path]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Load valid manual invocation artifacts, skipping malformed files."""
    valid: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for path_like in paths:
        path = Path(path_like)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append((path.name, type(exc).__name__))
            continue
        if not isinstance(data, dict):
            skipped.append((path.name, "InvalidShape"))
            continue
        data["_artifact_path"] = str(path)
        valid.append(data)
    return valid, skipped


def split_context_blocks(context: str) -> list[str]:
    """Split an assembled context string into top-level blocks."""
    return [block for block in context.split("\n\n") if block.strip()]


def summarize_context(context: str, max_blocks: int = 4, max_chars: int = 120) -> str:
    """Create a compact human-review summary of context blocks."""
    blocks = split_context_blocks(context)
    parts: list[str] = []
    for block in blocks[:max_blocks]:
        line = block.replace("\n", " | ").strip()
        if len(line) > max_chars:
            line = line[:max_chars].rstrip() + "..."
        parts.append(line)
    return "\n".join(parts)


def build_prompt_extra_rules(
    mode,
    prompt_placeholder_aware: bool,
) -> str:
    """Return extra prompt rules for a quality variant."""
    if not prompt_placeholder_aware:
        return ""
    mode_value = getattr(mode, "value", mode)
    if mode_value == "continuation":
        return (
            "\n\nAdditional context handling rules:\n"
            "- Treat 'Text before cursor' as the strongest signal.\n"
            "- Nearby UI labels, navigation text, or context from other apps may be irrelevant; ignore them unless they clearly continue the same thought.\n"
            "- If the focused field text looks like placeholder text, a suggested prompt, or non-user-authored UI copy, do not continue it literally.\n"
            "- Prefer the most literal continuation of the exact words already typed over a creative guess about what the user might mean.\n"
            "- Reuse the syntax and direction already present in the draft instead of pivoting to advice, troubleshooting, or a new task.\n"
            "- If the draft ends with an unfinished lead-in such as 'also,' or 'do you think', complete that same clause naturally rather than starting a different idea.\n"
            "- If the local context is weak, prefer abstract references like 'it', 'this', 'that', or 'we' over inventing specific nouns.\n"
            "- Do not invent debugging steps, logs, servers, APIs, env vars, deployments, or configuration issues unless the user already mentioned them explicitly.\n"
            "- When context is weak, stay close to the literal wording and syntax of the text before cursor instead of guessing a new topic.\n"
            "- Prefer relevant continuation over generic troubleshooting, task switching, or invented follow-up actions.\n"
        )
    return (
        "\n\nAdditional context handling rules:\n"
        "- Prefer the user's draft and the most relevant recent content over nearby UI labels and navigation text.\n"
        "- If the focused field text looks like placeholder text or non-user-authored UI copy, do not treat it as the user's intended message.\n"
        "- Ignore unrelated context from other apps unless it is clearly relevant to what the user is replying to.\n"
        "- Do not invent operational debugging or implementation details unless they are clearly present in the visible context.\n"
    )


def _is_cross_app_block(block: str) -> bool:
    return block.startswith("[Recent activity from other apps]")


def _is_semantic_block(block: str) -> bool:
    return block.startswith("Background context (lower priority):")


def _is_nearby_block(block: str) -> bool:
    return block.startswith("Nearby content:")


def _is_visible_context_block(block: str) -> bool:
    return block.startswith("Visible context:")


def _extract_text_from_xmlish(line: str) -> str:
    text = re.sub(r"<[^>]+>", " ", line)
    return " ".join(text.split())


def _reduce_nearby_block(block: str, max_lines: int = 4, max_chars: int = 240) -> str:
    header, _, body = block.partition("\n")
    kept: list[str] = []
    total = 0
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # The focused input already appears separately as before/after cursor.
        if 'focused="true"' in stripped or "<input" in stripped or "</input" in stripped:
            continue
        text = _extract_text_from_xmlish(stripped)
        if not text:
            continue
        if not _filter_ui_chrome_text(text).strip():
            continue
        if len(text) < 20:
            continue
        candidate = stripped
        if total + len(candidate) > max_chars:
            break
        kept.append(candidate)
        total += len(candidate)
        if len(kept) >= max_lines:
            break
    if not kept:
        return ""
    return header + "\n" + "\n".join(kept)


def _extract_block_value(context: str, header: str) -> str:
    for block in split_context_blocks(context):
        if block.startswith(header):
            _, _, body = block.partition("\n")
            return body.strip()
    return ""


def _strong_prefix_present(context: str) -> bool:
    before_cursor = _extract_block_value(context, "Text before cursor:")
    if len(before_cursor.strip()) >= 24:
        return True
    words = [word for word in before_cursor.strip().split() if word]
    return len(words) >= 5


def apply_quality_variant_to_context(
    context: str,
    variant: QualityVariant,
) -> str:
    """Transform an assembled context string according to a review variant."""
    blocks = split_context_blocks(context)
    result: list[str] = []
    for block in blocks:
        if variant.strip_cross_app and _is_cross_app_block(block):
            continue
        if variant.strip_semantic and _is_semantic_block(block):
            continue
        if (
            variant.prefer_cursor_only_when_strong_prefix
            and _strong_prefix_present(context)
            and (_is_nearby_block(block) or _is_visible_context_block(block))
        ):
            continue
        if variant.reduce_nearby and _is_nearby_block(block):
            reduced = _reduce_nearby_block(block)
            if reduced:
                result.append(reduced)
            continue
        result.append(block)
    return "\n\n".join(result)
# Lightweight copy of the UI-chrome filter so this module stays dependency-light
# and avoids import cycles with input observation / suggestion generation.
_UI_CHROME_PATTERNS = frozenset({
    "send gif", "send a gif", "send sticker", "unmute", "mute",
    "do not disturb", "set status", "user status and settings",
    "manage profile and status", "deafen", "undeafen",
    "open user settings", "direct messages", "find or start a conversation",
    "create dm", "add friends to dm", "hide user profile",
    "show user profile", "start voice call", "start video call",
    "pinned messages", "add friends", "notification settings",
    "thread panel", "member list", "search", "inbox",
    "new message", "details", "send", "attach", "emoji",
})
_UI_CHROME_SUBSTRINGS = (
    "icon, button",
    "status and settings",
    "unread message",
    "new unreads",
)


def _filter_ui_chrome_text(text: str) -> str:
    lines = text.split("\n")
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in _UI_CHROME_PATTERNS:
            continue
        if any(sub in lower for sub in _UI_CHROME_SUBSTRINGS):
            continue
        if len(stripped) <= 3:
            continue
        filtered.append(line)
    return "\n".join(filtered)
