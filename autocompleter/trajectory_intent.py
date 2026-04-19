"""Experimental: turn a recent-window-activity ring buffer into a one-sentence
guess at what the user is currently doing.

Lives on the `trajectory-intent-probe` branch. Not wired into the main
suggestion pipeline. The question this probes: given only what AX + NSWorkspace
can see (app name, window title, URL, focused role + short value preview) over
the last ~10 minutes, can a fast LLM produce a useful summary of intent?

Design choices (see plan for the why):
    * Fast LLM first. The user wants semi-real-time output, not batch summary
      at end of day. Use the fallback provider (Cerebras/Groq) so first token
      is ~200-500ms.
    * Log raw input + output + latency to JSONL so we can re-eval the same
      trajectory later with heavier models without re-capturing.
    * No structured output. One English sentence, max 30 words. Parsing a
      rigid schema here slows first token and we're not consuming this
      programmatically — a human (Henry) reads it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Qwen3 and other reasoning models wrap chain-of-thought in <think>...</think>
# before the final answer. Strip that so we show only the one-liner the model
# actually wants to surface. Also handles the case where the closing tag is
# missing (truncated by max_tokens) by falling back to content after the last
# </think> or the whole string if neither tag is present.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"^<think>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    # Remove complete <think>...</think> blocks.
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # If an unclosed <think> remains (truncation), drop everything from it
    # onward — we don't want the reasoning bleeding into the overlay.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()

logger = logging.getLogger(__name__)


EVAL_LOG_FILENAME = "trajectory_eval.jsonl"


SYSTEM_PROMPT = (
    "From a log of recent macOS window/app activity, name the user's GOAL — "
    "the objective they are working toward, or predict what they will do next. "
    "ONE sentence, at most 25 words, second person ('You're trying to…'). "
    "Be specific: name files, URLs, repos, people, or topics you can see in "
    "the data. Do NOT summarize which apps are open or describe the screen. "
    "Do NOT use the pattern 'doing X while doing Y'. "
    "If the trail is too sparse, say 'Not enough activity to infer a goal yet.'"
)


@dataclass
class TrajectoryGuess:
    intent_sentence: str
    confidence: float | None
    model_used: str
    latency_ms: float


def _format_snapshots(snapshots: list[dict[str, Any]]) -> str:
    """Compact the snapshot list into a single prompt-friendly block.

    Each line: `[+{dt}s] {trigger}  {app}  |  {title}  |  {role}={preview}`.
    Relative timestamps keep the prompt short and remove an absolute-time
    distractor for the LLM.
    """
    if not snapshots:
        return "(no activity recorded)"
    base_ts = snapshots[0].get("timestamp") or time.time()
    lines: list[str] = []
    for s in snapshots:
        ts = s.get("timestamp") or base_ts
        dt = max(0.0, ts - base_ts)
        last_seen = s.get("last_seen") or ts
        dwell = int(max(0.0, last_seen - ts))
        trigger = s.get("trigger") or "?"
        app = s.get("app_name") or "?"
        title = s.get("window_title") or ""
        url = s.get("url") or ""
        role = s.get("focused_role") or ""
        preview = s.get("focused_value_preview") or ""
        sim = s.get("content_similarity")
        parts = [f"[+{int(dt):>4}s]", f"{trigger:<13}", app]
        if dwell > 0:
            parts.append(f"(dwell {dwell}s)")
        if sim is not None:
            parts.append(f"(sim {sim:.2f})")
        if title:
            parts.append(f"| {title}")
        if url:
            parts.append(f"| url={url}")
        if role or preview:
            parts.append(f"| {role}={preview}" if preview else f"| role={role}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _truncate_for_log(text: str | None, limit: int = 200) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _log_eval(
    eval_log_path: Path,
    *,
    snapshots: list[dict[str, Any]],
    prompt_block: str,
    guess: TrajectoryGuess | None,
    error: str | None,
) -> None:
    """Append a single JSONL record for offline re-eval.

    We log the *raw* snapshots (not the compacted block) so a future
    experiment can try a different formatting. We also log the compacted
    block so reproducing the exact prompt is trivial.
    """
    try:
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "snapshots": snapshots,
            "prompt_block": prompt_block,
            "guess": None
            if guess is None
            else {
                "intent_sentence": guess.intent_sentence,
                "confidence": guess.confidence,
                "model_used": guess.model_used,
                "latency_ms": guess.latency_ms,
            },
            "error": error,
        }
        with eval_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — eval log must never kill the worker
        logger.warning("trajectory eval log write failed: %s", exc)


def infer_intent(
    snapshots: list[dict[str, Any]],
    *,
    client,
    model: str,
    eval_log_path: Path,
    request_timeout_s: float = 2.0,
) -> TrajectoryGuess | None:
    """Call the LLM with a one-shot completion and return the guess.

    `client` is any OpenAI-compatible client (the fallback provider's client
    from SuggestionEngine fits). Returns None on any error after logging it
    to the eval log — the caller can decide how to surface that.
    """
    prompt_block = _format_snapshots(snapshots)

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Recent activity (most recent last):\n\n"
                        f"{prompt_block}\n\n"
                        "What is the user doing right now?"
                    ),
                },
            ],
            # Generous budget because qwen3 and other reasoning models burn
            # tokens inside <think> before emitting the final one-liner.
            # Without this the think block gets truncated and the answer
            # never arrives.
            max_tokens=512,
            temperature=0.3,
            timeout=request_timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000.0
        err = f"{type(exc).__name__}: {_truncate_for_log(str(exc))}"
        logger.info("trajectory_intent LLM call failed in %.0fms: %s", latency_ms, err)
        _log_eval(
            eval_log_path,
            snapshots=snapshots,
            prompt_block=prompt_block,
            guess=None,
            error=err,
        )
        return None

    latency_ms = (time.perf_counter() - start) * 1000.0
    try:
        text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 — defensive, unusual provider
        err = f"response parse failed: {exc}"
        _log_eval(
            eval_log_path,
            snapshots=snapshots,
            prompt_block=prompt_block,
            guess=None,
            error=err,
        )
        return None

    stripped = _strip_reasoning(text)
    # Take the first non-empty line — the overlay is one-line sized.
    sentence = next(
        (line.strip() for line in stripped.splitlines() if line.strip()),
        "",
    )
    guess = TrajectoryGuess(
        intent_sentence=sentence,
        confidence=None,  # no calibrated confidence for this probe
        model_used=model,
        latency_ms=latency_ms,
    )
    _log_eval(
        eval_log_path,
        snapshots=snapshots,
        prompt_block=prompt_block,
        guess=guess,
        error=None,
    )
    return guess
