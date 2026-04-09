#!/usr/bin/env python3
"""Replay a saved manual invocation artifact.

This reuses the exact stored request payload from a `--dump-dir` artifact so
we can inspect or replay what happened during a real manual autocomplete
invocation without depending on the live app state.

Usage:
    source venv/bin/activate

    # Inspect the artifact
    python replay_invocation.py dumps/manual-invocations/file.json

    # Show the stored prompt payload
    python replay_invocation.py dumps/manual-invocations/file.json --print-request

    # Replay the exact LLM request and print suggestions
    python replay_invocation.py dumps/manual-invocations/file.json --call-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")


def load_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_summary(data: dict) -> None:
    req = data.get("request", {}) or {}
    lat = data.get("latency", {}) or {}
    det = data.get("detection", {}) or {}
    print(f"Artifact: {data.get('artifactType', 'unknown')}")
    print(f"App: {data.get('app', '')}")
    print(f"Window: {data.get('windowTitle', '')}")
    print(f"Captured: {data.get('capturedAt', '')}")
    print(f"Trigger type: {data.get('triggerType', '')}")
    print(
        "Request: "
        f"{req.get('provider', '')}/{req.get('model', '')} "
        f"mode={req.get('mode', '')} shell={req.get('shell_mode', False)} "
        f"temp={req.get('temperature', '')} max_tokens={req.get('max_tokens', '')}"
    )
    print(
        "Detection: "
        f"use_shell={det.get('useShell', False)} "
        f"use_tui={det.get('useTui', False)} "
        f"turns={det.get('conversationTurnCount', 0)} "
        f"visible_source={det.get('visibleSource', '')}"
    )
    if lat:
        print(
            "Latency: "
            f"context={lat.get('context_ms')}ms "
            f"ttft={lat.get('llm_ttft_ms')}ms "
            f"e2e={lat.get('e2e_total_ms')}ms "
            f"fallback={lat.get('fallback_used', False)}"
        )
    if data.get("suggestions"):
        print("\nSuggestions:")
        for i, text in enumerate(data["suggestions"]):
            print(f"  [{i}] {text}")


def print_request(data: dict) -> None:
    req = data.get("request", {}) or {}
    print(json.dumps(req, indent=2, ensure_ascii=False))


def call_llm(data: dict) -> None:
    req = data.get("request", {}) or {}
    provider = req.get("provider") or ""
    base_url = req.get("base_url") or ""
    model = req.get("model") or ""
    system = req.get("system_prompt") or ""
    user = req.get("user_prompt") or ""
    temperature = req.get("temperature")
    max_tokens = req.get("max_tokens")

    if not provider or not model or not system or not user:
        raise SystemExit("Artifact does not contain a complete stored request.")

    if provider == "anthropic":
        import anthropic
        from autocompleter.config import load_config

        cfg = load_config()
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        with client.messages.stream(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        ) as stream:
            chunks = "".join(stream.text_stream)
    else:
        import openai
        from autocompleter.config import load_config

        cfg = load_config()
        client = openai.OpenAI(
            api_key=cfg.openai_api_key,
            base_url=base_url or None,
        )
        extra_body = None
        if "groq.com" in base_url:
            extra_body = {"reasoning_effort": "none"}
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        chunks = resp.choices[0].message.content or ""

    from autocompleter.suggestion_engine import _extract_complete_suggestions

    suggestions = _extract_complete_suggestions(chunks)
    if not suggestions:
        print(chunks)
        return
    for i, text in enumerate(suggestions):
        print(f"[{i}] {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved manual invocation artifact")
    parser.add_argument("artifact", type=str, help="Path to an invocation JSON artifact")
    parser.add_argument("--print-request", action="store_true", help="Print the stored request payload")
    parser.add_argument("--call-llm", action="store_true", help="Replay the stored LLM request")
    args = parser.parse_args()

    data = load_artifact(args.artifact)
    print_summary(data)

    if args.print_request:
        print("\n=== Stored Request ===")
        print_request(data)

    if args.call_llm:
        print("\n=== Replay Result ===")
        call_llm(data)


if __name__ == "__main__":
    main()
