#!/usr/bin/env python3
"""Evaluate memory consolidation quality.

Seeds a mock MemoryStore with realistic raw memories, runs the consolidation
pipeline with a real LLM call (Groq), and scores the output markdown against
expected classifications and content.

Requires GROQ_API_KEY.

Usage:
    source venv/bin/activate
    python evaluate_consolidation.py [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from autocompleter.config import Config
from autocompleter.consolidation import (
    _build_consolidation_messages,
    _call_llm,
    run_consolidation,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic raw memories with ground-truth labels
# ---------------------------------------------------------------------------

MEMORIES = [
    # DURABLE — should end up in markdown
    {"id": "d1", "memory": "User's name is Henry", "updated_at": "2026-04-01T10:00:00Z",
     "label": "durable", "expect_keyword": "Henry"},
    {"id": "d2", "memory": "User prefers PostgreSQL over MySQL for all projects", "updated_at": "2026-04-02T08:00:00Z",
     "label": "durable", "expect_keyword": "PostgreSQL"},
    {"id": "d3", "memory": "User works at Acme Corp as a senior backend engineer", "updated_at": "2026-04-03T09:00:00Z",
     "label": "durable", "expect_keyword": "Acme"},
    {"id": "d4", "memory": "User prefers dark mode in all editors and tools", "updated_at": "2026-04-04T11:00:00Z",
     "label": "durable", "expect_keyword": "dark mode"},
    {"id": "d5", "memory": "User's main project is called Orchestrator, a macOS autocomplete tool", "updated_at": "2026-04-05T14:00:00Z",
     "label": "durable", "expect_keyword": "Orchestrator"},
    {"id": "d6", "memory": "User writes Python and Go primarily", "updated_at": "2026-04-05T15:00:00Z",
     "label": "durable", "expect_keyword": "Python"},
    {"id": "d7", "memory": "User's teammate Alex handles the iOS frontend", "updated_at": "2026-04-06T10:00:00Z",
     "label": "durable", "expect_keyword": "Alex"},
    {"id": "d8", "memory": "User prefers concise commit messages with imperative mood", "updated_at": "2026-04-07T16:00:00Z",
     "label": "durable", "expect_keyword": "commit"},

    # EPHEMERAL — should be in consolidated_ids but NOT in markdown
    {"id": "e1", "memory": "User is out of office April 14-18 2026", "updated_at": "2026-04-10T09:00:00Z",
     "label": "ephemeral", "expect_keyword": "April 14"},
    {"id": "e2", "memory": "User has a dentist appointment tomorrow at 3pm", "updated_at": "2026-04-11T20:00:00Z",
     "label": "ephemeral", "expect_keyword": "dentist"},
    {"id": "e3", "memory": "User is debugging a CORS issue in the staging environment", "updated_at": "2026-04-11T22:00:00Z",
     "label": "ephemeral", "expect_keyword": "CORS"},
    {"id": "e4", "memory": "User needs to respond to Sarah's Slack message about the Q2 budget", "updated_at": "2026-04-12T01:00:00Z",
     "label": "ephemeral", "expect_keyword": "Sarah"},

    # BORDERLINE — reasonable either way
    {"id": "b1", "memory": "User is currently migrating the auth service from REST to gRPC", "updated_at": "2026-04-08T13:00:00Z",
     "label": "borderline", "expect_keyword": "gRPC"},
]


def _score_result(result: dict, verbose: bool = False) -> dict:
    """Score the LLM consolidation result against ground truth."""
    md = result.get("markdown", "")
    consolidated = set(result.get("consolidated_ids", []))
    md_lower = md.lower()

    all_ids = {m["id"] for m in MEMORIES}
    durable = [m for m in MEMORIES if m["label"] == "durable"]
    ephemeral = [m for m in MEMORIES if m["label"] == "ephemeral"]
    borderline = [m for m in MEMORIES if m["label"] == "borderline"]

    scores = {
        "durable_in_markdown": 0,
        "durable_total": len(durable),
        "ephemeral_excluded": 0,
        "ephemeral_total": len(ephemeral),
        "all_ids_consolidated": 0,
        "all_ids_total": len(all_ids),
        "has_sections": False,
        "details": [],
    }

    # Check durable memories appear in markdown
    for m in durable:
        kw = m["expect_keyword"].lower()
        found = kw in md_lower
        scores["durable_in_markdown"] += int(found)
        if verbose or not found:
            scores["details"].append(
                f"  {'✓' if found else '✗'} DURABLE [{m['id']}] keyword '{m['expect_keyword']}': "
                f"{'found' if found else 'MISSING'} in markdown"
            )

    # Check ephemeral memories are NOT in markdown
    for m in ephemeral:
        kw = m["expect_keyword"].lower()
        excluded = kw not in md_lower
        scores["ephemeral_excluded"] += int(excluded)
        if verbose or not excluded:
            scores["details"].append(
                f"  {'✓' if excluded else '✗'} EPHEMERAL [{m['id']}] keyword '{m['expect_keyword']}': "
                f"{'excluded' if excluded else 'LEAKED into markdown'}"
            )

    # Check borderline (just report, don't score)
    for m in borderline:
        kw = m["expect_keyword"].lower()
        in_md = kw in md_lower
        scores["details"].append(
            f"  ? BORDERLINE [{m['id']}] keyword '{m['expect_keyword']}': "
            f"{'in markdown' if in_md else 'excluded'}"
        )

    # Check all IDs are in consolidated_ids
    for mid in all_ids:
        if mid in consolidated:
            scores["all_ids_consolidated"] += 1
        elif verbose:
            scores["details"].append(f"  ✗ ID {mid} missing from consolidated_ids")

    # Check section structure
    expected_sections = ["## Identity", "## Preferences", "## Projects", "## Contacts"]
    scores["has_sections"] = all(s in md for s in expected_sections)

    return scores


def _print_report(scores: dict, md: str):
    d_pct = (scores["durable_in_markdown"] / scores["durable_total"] * 100
             if scores["durable_total"] else 0)
    e_pct = (scores["ephemeral_excluded"] / scores["ephemeral_total"] * 100
             if scores["ephemeral_total"] else 0)
    id_pct = (scores["all_ids_consolidated"] / scores["all_ids_total"] * 100
              if scores["all_ids_total"] else 0)

    print("\n" + "=" * 60)
    print("CONSOLIDATION QUALITY REPORT")
    print("=" * 60)
    print(f"\nDurable recall:      {scores['durable_in_markdown']}/{scores['durable_total']} ({d_pct:.0f}%)")
    print(f"Ephemeral exclusion: {scores['ephemeral_excluded']}/{scores['ephemeral_total']} ({e_pct:.0f}%)")
    print(f"IDs consolidated:    {scores['all_ids_consolidated']}/{scores['all_ids_total']} ({id_pct:.0f}%)")
    print(f"Section structure:   {'✓' if scores['has_sections'] else '✗'}")

    if scores["details"]:
        print("\nDetails:")
        for line in scores["details"]:
            print(line)

    print(f"\n--- Generated memory.md ---")
    print(md)
    print("--- end ---\n")

    # Overall pass/fail
    pass_threshold = (
        d_pct >= 75
        and e_pct >= 75
        and id_pct >= 80
        and scores["has_sections"]
    )
    print(f"Overall: {'PASS' if pass_threshold else 'FAIL'}")
    return pass_threshold


def run_eval(verbose: bool = False) -> bool:
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Cannot run consolidation eval.")
        sys.exit(1)

    # Build prompt with empty existing markdown (first consolidation)
    memories_for_prompt = [
        {"id": m["id"], "memory": m["memory"], "updated_at": m["updated_at"]}
        for m in MEMORIES
    ]
    messages = _build_consolidation_messages("", memories_for_prompt)

    print(f"Sending {len(MEMORIES)} memories to LLM for consolidation...")
    print(f"  ({len([m for m in MEMORIES if m['label'] == 'durable'])} durable, "
          f"{len([m for m in MEMORIES if m['label'] == 'ephemeral'])} ephemeral, "
          f"{len([m for m in MEMORIES if m['label'] == 'borderline'])} borderline)")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            data_dir=Path(tmp),
            memory_enabled=True,
            memory_llm_provider="groq",
            memory_llm_model="qwen/qwen3-32b",
        )
        result = _call_llm(cfg, messages)

    if result is None:
        print("ERROR: LLM returned None (call failed or unparseable response)")
        return False

    scores = _score_result(result, verbose=verbose)
    return _print_report(scores, result.get("markdown", ""))


def run_update_eval(verbose: bool = False) -> bool:
    """Test consolidation with pre-existing memory.md (update scenario)."""
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    existing_md = """\
## Identity
- Name: Henry
- Senior backend engineer at Acme Corp

## Preferences
- PostgreSQL over MySQL
- Dark mode in all editors

## Projects
- Orchestrator: macOS autocomplete tool (main project)

## Contacts
- Alex: handles iOS frontend
"""

    new_memories = [
        {"id": "n1", "memory": "User started learning Rust last week", "updated_at": "2026-04-11T10:00:00Z",
         "label": "durable", "expect_keyword": "Rust"},
        {"id": "n2", "memory": "User's manager is named Jordan", "updated_at": "2026-04-11T12:00:00Z",
         "label": "durable", "expect_keyword": "Jordan"},
        {"id": "n3", "memory": "User has a PR review due by end of day", "updated_at": "2026-04-12T08:00:00Z",
         "label": "ephemeral", "expect_keyword": "PR review"},
        {"id": "n4", "memory": "User prefers vim keybindings", "updated_at": "2026-04-12T09:00:00Z",
         "label": "durable", "expect_keyword": "vim"},
    ]

    memories_for_prompt = [
        {"id": m["id"], "memory": m["memory"], "updated_at": m["updated_at"]}
        for m in new_memories
    ]
    messages = _build_consolidation_messages(existing_md, memories_for_prompt)

    print(f"\n{'=' * 60}")
    print("UPDATE SCENARIO (existing memory.md + 4 new memories)")
    print("=" * 60)
    print(f"Sending {len(new_memories)} new memories to LLM...")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            data_dir=Path(tmp),
            memory_enabled=True,
            memory_llm_provider="groq",
            memory_llm_model="qwen/qwen3-32b",
        )
        result = _call_llm(cfg, messages)

    if result is None:
        print("ERROR: LLM returned None")
        return False

    md = result.get("markdown", "")
    md_lower = md.lower()
    consolidated = set(result.get("consolidated_ids", []))

    # Check existing content preserved
    preserved = ["henry", "acme", "postgresql", "dark mode", "orchestrator", "alex"]
    preserved_count = sum(1 for kw in preserved if kw in md_lower)

    # Check new durable added
    new_durable = [m for m in new_memories if m["label"] == "durable"]
    new_found = sum(1 for m in new_durable if m["expect_keyword"].lower() in md_lower)

    # Check ephemeral excluded
    new_ephemeral = [m for m in new_memories if m["label"] == "ephemeral"]
    eph_excluded = sum(1 for m in new_ephemeral if m["expect_keyword"].lower() not in md_lower)

    print(f"\nExisting preserved: {preserved_count}/{len(preserved)}")
    print(f"New durable added:  {new_found}/{len(new_durable)}")
    print(f"Ephemeral excluded: {eph_excluded}/{len(new_ephemeral)}")
    print(f"IDs consolidated:   {len(consolidated)}/{len(new_memories)}")

    details = []
    for kw in preserved:
        found = kw in md_lower
        if not found:
            details.append(f"  ✗ EXISTING '{kw}' LOST from markdown")
    for m in new_durable:
        kw = m["expect_keyword"].lower()
        found = kw in md_lower
        details.append(f"  {'✓' if found else '✗'} NEW DURABLE [{m['id']}] '{m['expect_keyword']}': {'found' if found else 'MISSING'}")
    for m in new_ephemeral:
        kw = m["expect_keyword"].lower()
        excluded = kw not in md_lower
        details.append(f"  {'✓' if excluded else '✗'} EPHEMERAL [{m['id']}] '{m['expect_keyword']}': {'excluded' if excluded else 'LEAKED'}")

    if details:
        print("\nDetails:")
        for line in details:
            print(line)

    print(f"\n--- Updated memory.md ---")
    print(md)
    print("--- end ---\n")

    ok = preserved_count >= 5 and new_found >= 2 and eph_excluded >= 1
    print(f"Overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate consolidation quality")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    ok1 = run_eval(args.verbose)
    ok2 = run_update_eval(args.verbose)

    print(f"\n{'=' * 60}")
    print(f"FINAL: First consolidation {'PASS' if ok1 else 'FAIL'}, Update {'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)
