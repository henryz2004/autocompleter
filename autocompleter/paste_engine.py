"""Intelligent paste: given recent OCR captures + current field context,
produce ranked paste suggestions via LLM.

Separate from the autocomplete suggestion engine — paste is retrieval-
focused ("what did the user recently see that belongs here?") rather
than generation-focused ("what will the user type next?").
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You help users fill form fields by extracting relevant information "
    "from content they recently viewed in other windows.\n\n"
    "The captured content includes OCR text with bounding box coordinates "
    "[x:X y:Y w:W h:H] per line. Use x-coordinates to distinguish main "
    "content (usually one column) from sidebar UI chrome (different column). "
    "Ignore UI elements like buttons, navigation, and labels.\n\n"
    "Rules:\n"
    "- Each suggestion must be ready to paste — no quotes, no explanation.\n"
    "- Be specific: use exact values from the captured content.\n"
    "- PRESERVE original formatting: keep line breaks, bullet points, and "
    "paragraph structure from the source content.\n"
    "- Pay attention to the field context: a 'Title' field needs a short "
    "title, a 'Price' field needs a price, a 'Description' field needs "
    "the full description with formatting preserved.\n"
    "- Use the field label/placeholder to determine what content belongs.\n"
    "- If nothing relevant was captured, return an empty suggestions array.\n"
    "- Return valid JSON only. Use \\n for newlines inside JSON strings."
)

# Qwen3 / reasoning model <think> block stripping — reuse from trajectory_intent.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


@dataclass
class PasteSuggestion:
    text: str
    rank: int
    source: str | None = None


def _build_user_message(
    *,
    app_name: str,
    window_title: str | None,
    focused_role: str | None,
    focused_value: str | None,
    focused_label: str | None = None,
    caret_context_ocr: str | None = None,
    captures: list[dict[str, Any]],
) -> str:
    parts = []
    parts.append(
        f"The user is currently in: [{app_name}"
        + (f' - "{window_title}"' if window_title else "")
        + "]"
    )
    field_desc = focused_role or "unknown field"
    if focused_label:
        field_desc += f' (label: "{focused_label}")'
    parts.append(f"Field: {field_desc}")
    if focused_value:
        parts.append(f"Field currently contains: {focused_value[:200]}")
    else:
        parts.append("Field is currently: empty")

    if caret_context_ocr:
        parts.append(
            "\nOCR of the area around the user's cursor (shows the field "
            "and nearby labels/context — use this to identify WHICH field "
            "they are in):"
        )
        parts.append(f"```\n{caret_context_ocr[:2000]}\n```")

    parts.append(
        "\nManually captured content (ordered by relevance — prefer "
        "earlier entries. Entries marked [USED Nx] have already been "
        "pasted and should be deprioritized):"
    )
    for cap in captures:
        cap_app = cap.get("app_name", "?")
        cap_title = cap.get("window_title", "")
        cap_url = cap.get("url", "")
        ocr = cap.get("ocr_text", "")
        times_used = cap.get("times_used", 0)
        used_tag = f" [USED {times_used}x]" if times_used > 0 else ""
        header = f"[{cap_app}"
        if cap_title:
            header += f' - "{cap_title}"'
        if cap_url:
            header += f" | {cap_url}"
        header += f"]{used_tag}"
        parts.append(f"---\n{header}\n{ocr[:3000]}\n---")

    parts.append(
        '\nProduce 3 suggestions for what to paste in this field, ranked by '
        'relevance. Return as JSON: [{"text": "...", "rank": 1}, ...]'
    )
    return "\n".join(parts)


def generate_paste_suggestions(
    *,
    app_name: str,
    window_title: str | None,
    focused_role: str | None,
    focused_value: str | None,
    focused_label: str | None = None,
    caret_context_ocr: str | None = None,
    captures: list[dict[str, Any]],
    client,
    model: str,
    request_timeout_s: float = 5.0,
) -> list[PasteSuggestion]:
    """Call the LLM and return ranked paste suggestions."""
    user_msg = _build_user_message(
        app_name=app_name,
        window_title=window_title,
        focused_role=focused_role,
        focused_value=focused_value,
        focused_label=focused_label,
        caret_context_ocr=caret_context_ocr,
        captures=captures,
    )

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.2,
            timeout=request_timeout_s,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info("paste LLM call failed in %.0fms: %s", latency_ms, exc)
        return []

    latency_ms = (time.perf_counter() - start) * 1000.0
    raw = resp.choices[0].message.content or ""
    stripped = _strip_reasoning(raw)

    # Parse JSON array from the response.
    try:
        # Find the JSON array in the response — model may wrap it in markdown.
        match = re.search(r"\[.*\]", stripped, re.DOTALL)
        if not match:
            logger.warning("paste: no JSON array in response: %s", stripped[:200])
            return []
        arr = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("paste: JSON parse failed: %s — raw: %s", exc, stripped[:200])
        return []

    suggestions = []
    for item in arr:
        if isinstance(item, dict) and "text" in item:
            suggestions.append(PasteSuggestion(
                text=item["text"],
                rank=item.get("rank", len(suggestions) + 1),
                source=item.get("source"),
            ))

    logger.info(
        "paste: %d suggestions in %.0fms (model=%s)",
        len(suggestions), latency_ms, model,
    )
    return suggestions
