#!/usr/bin/env python3
"""Benchmark regeneration diversity strategies.

Replays saved fixture requests with different temperature/prompt strategies
and measures how different the regenerated suggestions are from the originals.

Usage:
    source venv/bin/activate
    python benchmark_regeneration.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from autocompleter.suggestion_engine import (
    AutocompleteMode,
    _extract_complete_suggestions,
    _normalize_similarity_text,
    postprocess_suggestion_texts,
)


FIXTURE_DIR = Path("/tmp/autocompleter_fixtures")

# Strategies to test: (name, temperature, system_prompt_modifier)
# system_prompt_modifier is a callable(system, avoided) -> new_system
STRATEGIES: list[tuple[str, float, object]] = [
    (
        "baseline (current: append, temp=0.75)",
        0.75,
        lambda system, avoided: system + (
            "\n\nIMPORTANT: The user is regenerating because they want "
            "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
            "any of these previous suggestions — take a completely "
            "different angle, tone, or approach:\n" + avoided
        ),
    ),
    (
        "prepend avoid list, temp=0.75",
        0.75,
        lambda system, avoided: (
            "IMPORTANT: The user is regenerating because they want "
            "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
            "any of these previous suggestions — take a completely "
            "different angle, tone, or approach:\n" + avoided
            + "\n\n" + system
        ),
    ),
    (
        "append, temp=1.0",
        1.0,
        lambda system, avoided: system + (
            "\n\nIMPORTANT: The user is regenerating because they want "
            "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
            "any of these previous suggestions — take a completely "
            "different angle, tone, or approach:\n" + avoided
        ),
    ),
    (
        "prepend, temp=1.0",
        1.0,
        lambda system, avoided: (
            "IMPORTANT: The user is regenerating because they want "
            "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
            "any of these previous suggestions — take a completely "
            "different angle, tone, or approach:\n" + avoided
            + "\n\n" + system
        ),
    ),
    (
        "append, temp=1.2",
        1.2,
        lambda system, avoided: system + (
            "\n\nIMPORTANT: The user is regenerating because they want "
            "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
            "any of these previous suggestions — take a completely "
            "different angle, tone, or approach:\n" + avoided
        ),
    ),
    (
        "prepend, temp=1.2",
        1.2,
        lambda system, avoided: (
            "IMPORTANT: The user is regenerating because they want "
            "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
            "any of these previous suggestions — take a completely "
            "different angle, tone, or approach:\n" + avoided
            + "\n\n" + system
        ),
    ),
]


def _call_llm(system: str, user_msg: str, model: str, base_url: str,
              temperature: float, max_tokens: int) -> list[str]:
    """Call the LLM and return raw suggestion texts."""
    import openai
    from autocompleter.config import load_config
    cfg = load_config()
    client = openai.OpenAI(api_key=cfg.openai_api_key, base_url=base_url or None)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens or 1024,
    )
    if "groq.com" in (base_url or ""):
        kwargs["extra_body"] = {"reasoning_effort": "none"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    # Strip Qwen3 thinking tags
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    suggestions = _extract_complete_suggestions(content)
    if not suggestions:
        suggestions = [content.strip()] if content.strip() else []
    return suggestions


def _diversity_score(new_suggestions: list[str], original_suggestions: list[str],
                     negative_patterns: list[str]) -> dict:
    """Score how different new suggestions are from originals + negative patterns."""
    originals_norm = {_normalize_similarity_text(s) for s in original_suggestions if s.strip()}
    negatives_norm = {_normalize_similarity_text(s) for s in negative_patterns if s.strip()}
    all_seen = originals_norm | negatives_norm

    new_norm = [_normalize_similarity_text(s) for s in new_suggestions if s.strip()]
    if not new_norm:
        return {"unique": 0, "total": 0, "duplicates": [], "unique_pct": 0.0}

    duplicates = [s for s in new_suggestions if _normalize_similarity_text(s) in all_seen]
    unique = len(new_norm) - len(duplicates)
    return {
        "unique": unique,
        "total": len(new_norm),
        "duplicates": duplicates,
        "unique_pct": unique / len(new_norm) * 100 if new_norm else 0,
    }


def find_regeneration_fixtures() -> list[dict]:
    """Find fixtures that are regenerations (have negative_patterns)."""
    fixtures = []
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        neg = data.get("request", {}).get("negative_patterns", [])
        if neg and len(neg) > 0:
            fixtures.append(data)
    return fixtures


def main():
    fixtures = find_regeneration_fixtures()
    print(f"Found {len(fixtures)} fixtures with negative patterns\n")

    # Use the fixture with the most negative patterns (worst case for diversity)
    fixtures.sort(key=lambda d: len(d.get("request", {}).get("negative_patterns", [])), reverse=True)

    # Test on up to 3 fixtures
    test_fixtures = fixtures[:3]

    for fixture in test_fixtures:
        req = fixture.get("request", {})
        original_suggestions = fixture.get("suggestions", [])
        if isinstance(original_suggestions[0], dict):
            original_suggestions = [s["text"] for s in original_suggestions]
        negative_patterns = req.get("negative_patterns", [])
        before_cursor = (fixture.get("focused") or {}).get("beforeCursor", "")

        # Strip the existing negative pattern appendage from system prompt
        # so we can re-add it in different positions
        base_system = req.get("system_prompt", "")
        # Find and remove the IMPORTANT/Avoid block at the end
        for marker in ["\n\nIMPORTANT: The user is regenerating", "\n\nAvoid generating suggestions"]:
            idx = base_system.find(marker)
            if idx != -1:
                base_system = base_system[:idx]
                break

        user_msg = req.get("user_prompt", "")
        model = req.get("model", "")
        base_url = req.get("base_url", "")
        max_tokens = req.get("max_tokens", 200)

        avoided = "\n".join(f"- {p}" for p in negative_patterns)

        print(f"{'='*70}")
        print(f"Fixture: gen={fixture.get('generationId')} app={fixture.get('app')}")
        print(f"Original suggestions: {original_suggestions}")
        print(f"Negative patterns ({len(negative_patterns)}): {negative_patterns[:3]}...")
        print(f"Before cursor (last 60): ...{before_cursor[-60:]}")
        print(f"Model: {model}")
        print()

        for name, temp, modifier in STRATEGIES:
            system = modifier(base_system, avoided)
            try:
                t0 = time.time()
                new_suggestions = _call_llm(system, user_msg, model, base_url, temp, max_tokens)
                elapsed = time.time() - t0

                # Postprocess
                mode = AutocompleteMode.CONTINUATION if req.get("mode") == "continuation" else AutocompleteMode.REPLY
                new_suggestions = postprocess_suggestion_texts(
                    new_suggestions, mode=mode, before_cursor=before_cursor,
                )
                new_suggestions = [s for s in new_suggestions if s.strip()]

                score = _diversity_score(new_suggestions, original_suggestions, negative_patterns)
                status = "✓" if score["unique_pct"] == 100 else "✗" if score["unique_pct"] == 0 else "~"

                print(f"  {status} {name}")
                print(f"    suggestions: {new_suggestions}")
                print(f"    diversity: {score['unique']}/{score['total']} unique ({score['unique_pct']:.0f}%)")
                if score["duplicates"]:
                    print(f"    duplicates: {score['duplicates']}")
                print(f"    latency: {elapsed:.2f}s")
            except Exception as e:
                print(f"  ✗ {name}")
                print(f"    ERROR: {e}")
            print()
            time.sleep(0.5)  # Rate limit courtesy


if __name__ == "__main__":
    main()
