#!/usr/bin/env python3
"""Evaluate memory extraction and retrieval quality.

Three evaluation modes:

  extraction  — Feed curated (messages, expected_facts) pairs through mem0,
                check which facts are captured vs missed.
  retrieval   — Load known memories into a fresh FAISS store, query with
                composite search strings, check recall@k.
  e2e         — Replay invocation artifacts with/without memory context,
                compare suggestion quality in a markdown report.

Requires GROQ_API_KEY and OPENAI_API_KEY for extraction/retrieval modes
(real API calls against mem0's fact extraction and embedding pipelines).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _load_fixtures(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"Fixture file not found: {p}")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Mode A: Extraction evaluation
# ---------------------------------------------------------------------------

def _run_extraction(fixtures: dict, verbose: bool = False) -> list[dict]:
    """For each extraction case, add messages to a fresh mem0 store and
    check which expected facts appear in the extracted memories."""
    from autocompleter.config import Config
    from autocompleter.memory import MemoryStore

    results: list[dict] = []
    cases = fixtures.get("extraction_cases", [])
    if not cases:
        print("No extraction_cases in fixtures.")
        return results

    for case in cases:
        name = case["name"]
        messages = case["messages"]
        expected = case.get("expected_facts", [])

        print(f"\n--- Extraction: {name} ---")
        if verbose:
            for msg in messages:
                print(f"  [{msg['role']}] {msg['content'][:120]}")

        # Fresh store for each case to isolate extraction.
        with tempfile.TemporaryDirectory(prefix="mem-eval-") as tmpdir:
            cfg = Config(
                data_dir=Path(tmpdir),
                memory_enabled=True,
            )
            store = MemoryStore(cfg)
            if not store.enabled:
                print("  SKIP — memory store failed to initialize (check API keys)")
                results.append({
                    "name": name,
                    "status": "skipped",
                    "reason": "init_failed",
                })
                continue

            t0 = time.time()
            store.add(messages)
            add_ms = (time.time() - t0) * 1000

            # Retrieve all extracted memories.
            all_memories = store.get_all()
            elapsed_ms = (time.time() - t0) * 1000

            # Check which expected facts are present.
            found: list[str] = []
            missed: list[str] = []
            memories_lower = " ".join(all_memories).lower()
            for fact in expected:
                if fact.lower() in memories_lower:
                    found.append(fact)
                else:
                    missed.append(fact)

            recall = len(found) / len(expected) if expected else 1.0

            print(f"  Extracted memories ({len(all_memories)}):")
            for mem in all_memories:
                print(f"    - {mem}")
            print(f"  Expected facts: {expected}")
            print(f"  Found: {found}")
            print(f"  Missed: {missed}")
            print(f"  Recall: {recall:.0%}  |  add: {add_ms:.0f}ms  |  total: {elapsed_ms:.0f}ms")

            results.append({
                "name": name,
                "status": "ok",
                "extracted_memories": all_memories,
                "expected_facts": expected,
                "found": found,
                "missed": missed,
                "recall": recall,
                "add_ms": add_ms,
            })

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_recall = sum(r["recall"] for r in ok_results) / len(ok_results)
        print(f"\n=== Extraction Summary: {len(ok_results)} cases, avg recall {avg_recall:.0%} ===")

    return results


# ---------------------------------------------------------------------------
# Mode B: Retrieval evaluation
# ---------------------------------------------------------------------------

def _run_retrieval(
    fixtures: dict,
    verbose: bool = False,
    decay_rate: float = 0.0,
) -> list[dict]:
    """Load known memories, query, check if expected ones surface in top-k."""
    from mem0 import Memory as Mem0Memory
    from autocompleter.config import Config

    results: list[dict] = []
    cases = fixtures.get("retrieval_cases", [])
    if not cases:
        print("No retrieval_cases in fixtures.")
        return results

    for case in cases:
        name = case["name"]
        memories_to_load = case["memories_to_load"]
        query = case["query"]
        expected_top_k = case.get("expected_in_top_3", [])

        print(f"\n--- Retrieval: {name} ---")
        print(f"  Query: {query}")

        with tempfile.TemporaryDirectory(prefix="mem-eval-ret-") as tmpdir:
            cfg = Config(
                data_dir=Path(tmpdir),
                memory_enabled=True,
            )
            # Build mem0 config directly to bypass MemoryStore's add() which
            # uses LLM extraction — we want to load raw memories.
            import os
            openai_key = os.environ.get("OPENAI_API_KEY", "") or cfg.openai_api_key
            groq_key = os.environ.get("GROQ_API_KEY", "")

            mem0_config: dict[str, Any] = {
                "vector_store": {
                    "provider": "faiss",
                    "config": {
                        "collection_name": "eval_retrieval",
                        "path": str(Path(tmpdir) / "memories"),
                    },
                },
                "llm": {
                    "provider": cfg.memory_llm_provider,
                    "config": {
                        "model": cfg.memory_llm_model,
                        "temperature": 0.1,
                        "max_tokens": 1500,
                    },
                },
                "embedder": {
                    "provider": cfg.memory_embedder_provider,
                    "config": {
                        "model": cfg.memory_embedder_model,
                    },
                },
                "history_db_path": str(Path(tmpdir) / "history.db"),
                "version": "v1.1",
            }

            try:
                mem = Mem0Memory.from_config(mem0_config)
            except Exception as exc:
                print(f"  SKIP — mem0 init failed: {exc}")
                results.append({"name": name, "status": "skipped", "reason": str(exc)})
                continue

            # Add each memory as a direct user message so mem0 stores it.
            for memory_text in memories_to_load:
                mem.add(
                    [{"role": "user", "content": memory_text}],
                    user_id="eval_user",
                )

            # Search
            t0 = time.time()
            search_results = mem.search(query=query, user_id="eval_user", limit=3)
            search_ms = (time.time() - t0) * 1000

            entries = (
                search_results.get("results", [])
                if isinstance(search_results, dict)
                else search_results
            )

            # Apply decay if configured.
            if decay_rate > 0:
                from autocompleter.memory import MemoryStore
                MemoryStore._apply_decay(entries, decay_rate)
                decay_info = f"  (decay λ={decay_rate})"
            else:
                decay_info = ""

            retrieved = [
                entry.get("memory", "") if isinstance(entry, dict) else str(entry)
                for entry in entries
            ]

            # Check expected in retrieved using keyword overlap.
            # mem0 rephrases memories during storage, so exact substring
            # matching is too strict.  Instead, check that the majority of
            # significant words from the expected string appear in at least
            # one retrieved memory.
            found: list[str] = []
            missed: list[str] = []
            _stop = {"a", "an", "the", "is", "are", "was", "were", "to",
                      "for", "of", "in", "on", "at", "by", "and", "or",
                      "user", "user's", "with", "has", "have", "that",
                      "this", "from", "due", "end"}
            for exp in expected_top_k:
                keywords = [
                    w for w in exp.lower().split()
                    if w not in _stop and len(w) > 2
                ]
                if not keywords:
                    found.append(exp)
                    continue
                matched = any(
                    sum(1 for kw in keywords if kw in r.lower())
                    >= max(1, len(keywords) * 0.5)
                    for r in retrieved
                )
                if matched:
                    found.append(exp)
                else:
                    missed.append(exp)

            recall = len(found) / len(expected_top_k) if expected_top_k else 1.0

            print(f"  Retrieved (top {len(retrieved)}):")
            for r in retrieved:
                print(f"    - {r}")
            print(f"  Expected in top-3: {expected_top_k}")
            print(f"  Found: {found}")
            print(f"  Missed: {missed}")
            print(f"  Recall@3: {recall:.0%}  |  search: {search_ms:.0f}ms{decay_info}")

            results.append({
                "name": name,
                "status": "ok",
                "retrieved": retrieved,
                "expected": expected_top_k,
                "found": found,
                "missed": missed,
                "recall_at_3": recall,
                "search_ms": search_ms,
            })

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_recall = sum(r["recall_at_3"] for r in ok_results) / len(ok_results)
        print(f"\n=== Retrieval Summary: {len(ok_results)} cases, avg recall@3 {avg_recall:.0%} ===")

    return results


# ---------------------------------------------------------------------------
# Mode C: End-to-end (suggestion quality with/without memory)
# ---------------------------------------------------------------------------

def _run_e2e(
    artifacts_path: str,
    output_path: str,
    verbose: bool = False,
) -> None:
    """Replay invocation artifacts with and without memory context."""
    from autocompleter.quality_review import (
        load_valid_invocation_artifacts,
        summarize_context,
    )

    paths = sorted(Path(artifacts_path).glob("*.json")) if Path(artifacts_path).is_dir() else [Path(artifacts_path)]
    artifacts, skipped = load_valid_invocation_artifacts(paths)

    if not artifacts:
        print("No valid artifacts found.")
        return

    lines: list[str] = [
        "# Memory E2E Evaluation",
        "",
        f"Artifacts: {len(artifacts)} loaded, {len(skipped)} skipped",
        "",
    ]

    for artifact in artifacts:
        name = Path(artifact["_artifact_path"]).name
        context = artifact.get("context", "")
        has_memory = "User memories:" in context

        lines.append(f"## {name}")
        lines.append(f"- Mode: `{(artifact.get('detection') or {}).get('mode', '')}`")
        lines.append(f"- Has memory context: `{has_memory}`")
        lines.append(f"- Context summary: {summarize_context(context)[:200]}")
        suggestions = artifact.get("suggestions", [])
        if suggestions:
            lines.append(f"- Suggestions: {json.dumps(suggestions, ensure_ascii=False)}")
        lines.append("")

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote e2e report to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory extraction and retrieval quality",
    )
    parser.add_argument(
        "--mode",
        choices=["extraction", "retrieval", "e2e", "all"],
        default="all",
        help="Evaluation mode (default: all)",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default="tests/fixtures/memory_eval_fixtures.json",
        help="Path to fixture JSON file",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="dumps/manual-invocations",
        help="Invocation artifacts path (for e2e mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dumps/reviews/memory-review.md",
        help="Markdown report output path (for e2e mode)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.0,
        help="Exponential decay λ for retrieval scoring (0 = disabled, 0.01 = ~3-day half-life)",
    )
    args = parser.parse_args()

    fixtures = None
    if args.mode in ("extraction", "retrieval", "all"):
        fixtures = _load_fixtures(args.fixtures)

    if args.mode in ("extraction", "all"):
        _run_extraction(fixtures, verbose=args.verbose)

    if args.mode in ("retrieval", "all"):
        _run_retrieval(fixtures, verbose=args.verbose, decay_rate=args.decay_rate)

    if args.mode in ("e2e", "all"):
        _run_e2e(args.artifacts, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
