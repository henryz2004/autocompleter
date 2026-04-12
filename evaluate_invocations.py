#!/usr/bin/env python3
"""Evaluate captured manual invocation artifacts across quality variants."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from autocompleter.context_store import ContextStore
from autocompleter.quality_review import (
    REVIEW_VARIANTS,
    QualityVariant,
    apply_quality_variant_to_context,
    load_valid_invocation_artifacts,
    split_context_blocks,
    summarize_context,
)
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    _extract_complete_suggestions,
    build_messages,
    postprocess_suggestion_texts,
)


def _artifact_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    return [input_path]


def _mode_from_artifact(data: dict) -> AutocompleteMode:
    mode = (data.get("detection", {}) or {}).get("mode", "reply")
    return AutocompleteMode.CONTINUATION if mode == "continuation" else AutocompleteMode.REPLY


def _extract_block_body(context: str, header: str) -> str:
    for block in split_context_blocks(context):
        if block.startswith(header):
            _, _, body = block.partition("\n")
            return body.strip()
    return ""


def _rebuild_context(data: dict) -> str:
    original_context = data.get("context") or ""
    with tempfile.TemporaryDirectory(prefix="autocompleter-review-") as tmpdir:
        store = ContextStore(Path(tmpdir) / "review.db")
        store.open()
        try:
            mode = _mode_from_artifact(data)
            cross_app_context = _extract_block_body(
                original_context,
                "[Recent activity from other apps]",
            )
            if cross_app_context:
                cross_app_context = (
                    "[Recent activity from other apps]\n" + cross_app_context
                )
            subtree_context = _extract_block_body(original_context, "Nearby content:")
            visible_text = data.get("visibleTextElements") or None
            source_app = data.get("app") or "Unknown"
            window_title = data.get("windowTitle") or ""
            source_url = (data.get("focused") or {}).get("sourceUrl", "")

            if mode == AutocompleteMode.CONTINUATION:
                return store.get_continuation_context(
                    before_cursor=(data.get("focused") or {}).get("beforeCursor", ""),
                    after_cursor=(data.get("focused") or {}).get("afterCursor", ""),
                    source_app=source_app,
                    window_title=window_title,
                    source_url=source_url,
                    visible_text=visible_text,
                    cross_app_context=cross_app_context,
                    subtree_context=subtree_context or None,
                )

            return store.get_reply_context(
                conversation_turns=data.get("conversationTurns") or [],
                source_app=source_app,
                window_title=window_title,
                source_url=source_url,
                draft_text=(data.get("focused") or {}).get("beforeCursor", ""),
                visible_text=visible_text,
                cross_app_context=cross_app_context,
                subtree_context=subtree_context or None,
            )
        finally:
            store.close()


def _call_llm(req: dict) -> list[str]:
    attempts = 3
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            return _call_llm_once(req)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == attempts:
                break
            time.sleep(1.0 * attempt)
    return [f"[LLM error after {attempts} attempts] {last_error}"]


def _call_llm_once(req: dict) -> list[str]:
    provider = req.get("provider") or ""
    base_url = req.get("base_url") or ""
    model = req.get("model") or ""
    system = req.get("system_prompt") or ""
    user = req.get("user_prompt") or ""
    temperature = req.get("temperature")
    max_tokens = req.get("max_tokens")

    if provider == "anthropic":
        import anthropic
        from autocompleter.config import load_config

        cfg = load_config()
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )
        content = "".join(
            block.text for block in resp.content if getattr(block, "text", None)
        )
    else:
        import openai
        from autocompleter.config import load_config

        cfg = load_config()
        client = openai.OpenAI(
            api_key=cfg.openai_api_key,
            base_url=base_url or None,
        )
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )
        if "groq.com" in base_url:
            kwargs["extra_body"] = {"reasoning_effort": "none"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
    suggestions = _extract_complete_suggestions(content)
    if not suggestions:
        suggestions = [content.strip()] if content.strip() else []

    mode = AutocompleteMode.CONTINUATION if req.get("mode") == "continuation" else AutocompleteMode.REPLY
    return postprocess_suggestion_texts(
        suggestions,
        mode=mode,
        before_cursor=req.get("before_cursor") or "",
    )


def _variant_request(data: dict, variant: QualityVariant) -> dict:
    req = dict((data.get("request") or {}))
    if variant.name == "baseline":
        return req

    mode = _mode_from_artifact(data)
    context = apply_quality_variant_to_context(_rebuild_context(data), variant)
    source_app = data.get("app") or "Unknown"
    num_suggestions = int(req.get("num_suggestions") or max(2, len(data.get("suggestions") or []),))
    system, user_msg = build_messages(
        mode=mode,
        context=context,
        num_suggestions=num_suggestions,
        streaming=True,
        source_app=source_app,
        prompt_placeholder_aware=variant.prompt_placeholder_aware,
    )
    new_req = dict(req)
    new_req["system_prompt"] = system
    new_req["user_prompt"] = user_msg
    if (
        mode == AutocompleteMode.CONTINUATION
        and variant.continuation_temperature is not None
    ):
        new_req["temperature"] = variant.continuation_temperature
    new_req["before_cursor"] = (data.get("focused") or {}).get("beforeCursor", "")
    new_req["variant_context"] = context
    new_req["variant_name"] = variant.name
    return new_req


def _render_report(results: list[dict], skipped: list[tuple[str, str]]) -> str:
    lines: list[str] = ["# Autocomplete Quality Review", ""]
    if skipped:
        lines.append("## Skipped Artifacts")
        for name, reason in skipped:
            lines.append(f"- `{name}`: {reason}")
        lines.append("")

    for result in results:
        lines.append(f"## {result['artifact_name']}")
        lines.append(f"- Mode: `{result['mode']}`")
        lines.append(f"- Before cursor: `{result['before_cursor']}`")
        lines.append("- Context summary:")
        for line in result["context_summary"].splitlines():
            lines.append(f"  - {line}")
        lines.append("")
        for variant in result["variants"]:
            lines.append(f"### {variant['name']}")
            lines.append(f"- Suggestions: {json.dumps(variant['suggestions'], ensure_ascii=False)}")
            lines.append("- Review note: ")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved invocation artifacts across quality variants")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="dumps/manual-invocations",
        help="Artifact file or directory",
    )
    parser.add_argument(
        "--call-llm",
        action="store_true",
        help="Actually replay non-baseline variants against the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dumps/reviews/latest-review.md",
        help="Markdown report path",
    )
    args = parser.parse_args()

    paths = _artifact_paths(Path(args.path))
    artifacts, skipped = load_valid_invocation_artifacts(paths)

    results: list[dict] = []
    for artifact in artifacts:
        variants: list[dict] = []
        for variant in REVIEW_VARIANTS:
            if variant.name == "baseline":
                suggestions = artifact.get("suggestions") or []
            elif args.call_llm:
                suggestions = _call_llm(_variant_request(artifact, variant))
            else:
                suggestions = []
            variants.append({
                "name": variant.name,
                "suggestions": suggestions,
            })
        results.append({
            "artifact_name": Path(artifact["_artifact_path"]).name,
            "mode": (artifact.get("detection") or {}).get("mode", ""),
            "before_cursor": (artifact.get("focused") or {}).get("beforeCursor", "")[:200],
            "context_summary": summarize_context(_rebuild_context(artifact)),
            "variants": variants,
        })

    report = _render_report(results, skipped)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote review report to {output_path}")


if __name__ == "__main__":
    main()
