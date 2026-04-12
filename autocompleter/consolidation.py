"""Daily memory consolidation: FAISS → markdown.

Reads all short-term memories from the FAISS vector store, asks an LLM to
classify them as durable vs ephemeral, merges the durable ones into
``~/.autocompleter/memory.md``, and purges consolidated entries from FAISS.

The consolidation is triggered:
- On startup if >24 h since last consolidation.
- Manually via ``python -m autocompleter --consolidate-memory``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config
    from .memory import MemoryStore

logger = logging.getLogger(__name__)

_TIMESTAMP_FILE = ".last_consolidation"
_CONSOLIDATION_INTERVAL_H = 24

# ---------------------------------------------------------------------------
# Timestamp tracking
# ---------------------------------------------------------------------------

def _should_consolidate(config: Config, *, force: bool = False) -> bool:
    if force:
        return True
    ts_path = config.data_dir / _TIMESTAMP_FILE
    if not ts_path.exists():
        return True
    try:
        text = ts_path.read_text(encoding="utf-8").strip()
        last = datetime.fromisoformat(text)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return hours >= _CONSOLIDATION_INTERVAL_H
    except (ValueError, OSError):
        return True


def _update_timestamp(config: Config) -> None:
    ts_path = config.data_dir / _TIMESTAMP_FILE
    tmp = ts_path.with_suffix(".tmp")
    try:
        tmp.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        os.replace(str(tmp), str(ts_path))
    except OSError:
        logger.warning("Failed to update consolidation timestamp", exc_info=True)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory curator for a personal autocomplete assistant. Your job is to \
analyze raw memory facts extracted from the user's conversations and organize \
lasting, durable ones into a structured markdown document.

Classify each memory as:
- DURABLE: identity, preferences, projects, contacts, recurring patterns, \
tool/tech choices, writing style — facts that remain true for weeks or longer.
- EPHEMERAL: one-time events, transient plans, stale dates, temporary context \
— facts that are no longer relevant or will expire soon.

All ephemeral memories should still be listed in consolidated_ids so they get \
purged from short-term storage."""

_USER_TEMPLATE = """\
## Current long-term memory file
{existing_md}

## New raw memories from short-term store
{memories_list}

## Instructions
1. Classify each numbered memory as DURABLE or EPHEMERAL.
2. Deduplicate against the existing memory file — do not add facts already captured.
3. Merge DURABLE facts into the appropriate section, updating or refining existing \
entries as needed.
4. Keep sections even if empty (with no bullets).
5. Output a JSON object with exactly two keys:
   - "markdown": the full updated memory.md content (with sections ## Identity, \
## Preferences, ## Projects, ## Contacts)
   - "consolidated_ids": list of ALL memory IDs (strings) from the numbered list \
— both durable (now in markdown) and ephemeral (to be purged)

Respond ONLY with the JSON object, no other text."""


def _build_consolidation_messages(
    existing_md: str,
    memories: list[dict],
) -> list[dict[str, str]]:
    if not existing_md.strip():
        existing_md = "(empty — first consolidation)"

    lines: list[str] = []
    for i, mem in enumerate(memories, 1):
        mid = mem.get("id", "?")
        text = mem.get("memory", "")
        ts = mem.get("updated_at") or mem.get("created_at", "")
        lines.append(f"{i}. [{mid}] {text}  (updated: {ts})")

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                existing_md=existing_md,
                memories_list="\n".join(lines),
            ),
        },
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(config: Config, messages: list[dict[str, str]]) -> dict | None:
    """Call the memory LLM and parse the JSON response.

    Returns ``{"markdown": "...", "consolidated_ids": [...]}`` or None on failure.
    """
    try:
        from openai import OpenAI

        provider = config.memory_llm_provider
        model = config.memory_llm_model

        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            base_url = "https://api.groq.com/openai/v1"
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "") or config.openai_api_key
            base_url = None

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=4000,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        logger.warning("Consolidation LLM call failed", exc_info=True)
        return None

    # Parse JSON — try direct, strip thinking tags, strip code fences.
    content = _strip_thinking(content)
    for attempt in (content, _strip_code_fences(content)):
        try:
            result = json.loads(attempt)
            if isinstance(result, dict) and "markdown" in result:
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    logger.warning(f"Consolidation LLM returned unparseable response: {content[:200]}")
    return None


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 chain-of-thought)."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# FAISS purge
# ---------------------------------------------------------------------------

def _purge_memories(memory_store: MemoryStore, ids: list[str]) -> int:
    """Delete memories by ID. Returns count of successful deletions."""
    deleted = 0
    for mid in ids:
        if memory_store.delete(mid):
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_consolidation(
    memory_store: MemoryStore,
    config: Config,
    *,
    force: bool = False,
) -> bool:
    """Run memory consolidation if due (or forced).

    Returns True if consolidation ran successfully, False otherwise.
    """
    if not memory_store.enabled:
        return False

    if not _should_consolidate(config, force=force):
        logger.debug("[CONSOLIDATE] Not due yet, skipping.")
        return False

    logger.info("[CONSOLIDATE] Starting memory consolidation...")

    # 1. Read all FAISS memories.
    memories = memory_store.get_all_with_ids()
    if not memories:
        logger.info("[CONSOLIDATE] No memories in FAISS, nothing to consolidate.")
        _update_timestamp(config)
        return True

    # 2. Read existing memory.md.
    memory_md_path = config.data_dir / "memory.md"
    try:
        existing_md = memory_md_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        existing_md = ""

    # 3. Build prompt and call LLM.
    messages = _build_consolidation_messages(existing_md, memories)
    result = _call_llm(config, messages)
    if result is None:
        logger.warning("[CONSOLIDATE] LLM call failed, aborting. Will retry next time.")
        return False

    new_markdown = result.get("markdown", "")
    consolidated_ids = result.get("consolidated_ids", [])

    if not new_markdown.strip():
        logger.warning("[CONSOLIDATE] LLM returned empty markdown, aborting.")
        return False

    # 4. Write markdown FIRST (atomic), then purge FAISS.
    tmp_path = memory_md_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(new_markdown, encoding="utf-8")
        os.replace(str(tmp_path), str(memory_md_path))
        logger.info(
            f"[CONSOLIDATE] Wrote {len(new_markdown)} chars to {memory_md_path}"
        )
    except OSError:
        logger.warning("[CONSOLIDATE] Failed to write memory.md, aborting.", exc_info=True)
        return False

    # 5. Purge consolidated memories from FAISS.
    if consolidated_ids:
        deleted = _purge_memories(memory_store, consolidated_ids)
        logger.info(
            f"[CONSOLIDATE] Purged {deleted}/{len(consolidated_ids)} memories from FAISS"
        )

    _update_timestamp(config)
    logger.info("[CONSOLIDATE] Consolidation complete.")
    return True
