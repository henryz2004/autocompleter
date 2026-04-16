#!/usr/bin/env python3
"""Replay a saved AX tree fixture through the full autocompleter pipeline.

Runs every stage of the autocomplete pipeline offline against a saved JSON
fixture, so you can iterate on context extraction / prompts / LLM behaviour
without manually triggering in live apps.

Usage:
    source venv/bin/activate

    # Show pipeline stages (no LLM call)
    python replay_fixture.py tests/fixtures/ax_trees/claude-current-chat.json

    # Show both continuation and reply context
    python replay_fixture.py claude.json --both-modes

    # Actually call the LLM and print suggestions
    python replay_fixture.py claude.json --call-llm

    # Override mode (ignore auto-detection)
    python replay_fixture.py claude.json --mode reply

    # Write output to file
    python replay_fixture.py claude.json -o /tmp/replay.log
"""
from __future__ import annotations

import argparse
import json
import os as _os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from autocompleter.fixture_tools import (
    NormalizedFixture,
    build_focused_element,
    extract_conversation_turns,
    load_normalized_fixture,
)

# ---------------------------------------------------------------------------
# Visible text extraction (dict-based, mirrors _collect_text logic)
# ---------------------------------------------------------------------------

# Roles that are considered UI chrome (skip for text extraction).
_CHROME_ROLES = frozenset({
    "AXButton", "AXCheckBox", "AXImage", "AXMenu", "AXMenuBar",
    "AXMenuItem", "AXPopUpButton", "AXProgressIndicator", "AXScrollBar",
    "AXSlider", "AXTabGroup", "AXToolbar", "AXValueIndicator",
    "AXBusyIndicator",
})

# Roles whose value attribute should be captured.
_TEXT_ROLES = frozenset({
    "AXStaticText", "AXTextArea", "AXTextField", "AXHeading",
    "AXLink", "AXParagraph",
})


def collect_text_from_dict(
    node: dict,
    max_depth: int = 15,
    max_items: int = 50,
    depth: int = 0,
) -> list[str]:
    """Extract visible text elements from a serialized AX tree dict.

    Mimics InputObserver._collect_text but works on dict nodes.
    Returns a list of text strings found in the tree.
    """
    results: list[str] = []
    _collect_text_recurse(node, results, max_depth, max_items, depth)
    return results


def _collect_text_recurse(
    node: dict,
    results: list[str],
    max_depth: int,
    max_items: int,
    depth: int,
) -> None:
    if depth > max_depth or len(results) >= max_items:
        return

    role = node.get("role", "")

    # Skip chrome
    if role in _CHROME_ROLES:
        return

    # Extract text from content roles
    if role in _TEXT_ROLES:
        value = node.get("value")
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if len(text) >= 2:  # Skip very short fragments
                results.append(text[:4000])  # Cap individual elements

    # Also capture AXDescription text if it has content
    desc = node.get("description", "")
    if isinstance(desc, str) and len(desc) > 10 and role not in _TEXT_ROLES:
        results.append(desc[:4000])

    # Recurse into children
    for child in node.get("children", []):
        if len(results) >= max_items:
            break
        _collect_text_recurse(child, results, max_depth, max_items, depth + 1)


def extract_subtree(tree: dict, token_budget: int = 500) -> str | None:
    """Run the subtree context walker on a fixture tree."""
    from autocompleter.subtree_context import extract_context_from_tree
    return extract_context_from_tree(tree, token_budget=token_budget)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(
    fixture: NormalizedFixture,
    focused,
    conversation_turns,
    subtree_context: str | None,
    mode,
    num_suggestions: int = 3,
) -> tuple[str, str, str, str | None]:
    """Assemble the LLM context and prompts.

    Uses the same ``build_messages()`` function as the live pipeline so
    prompts stay in sync (including STREAMING_JSON_INSTRUCTION).

    Returns ``(context, system_prompt, user_prompt, tree_overview)``.
    """
    from autocompleter.context_store import ContextStore
    from autocompleter.suggestion_engine import AutocompleteMode, build_messages
    from autocompleter.subtree_context import build_context_bundle_from_tree

    app_name = fixture.app
    window_title = fixture.window_title
    source_url = fixture.source_url

    # Ephemeral context store for assembly
    tmp_dir = tempfile.mkdtemp(prefix="replay_fixture_")
    db_path = Path(tmp_dir) / "replay.db"
    ctx_store = ContextStore(db_path)
    ctx_store.open()

    try:
        # Convert conversation turns to dicts
        turn_dicts = []
        if conversation_turns:
            turn_dicts = [
                {"speaker": t.speaker, "text": t.text}
                for t in conversation_turns
            ]

        before_cursor = focused.before_cursor if focused else ""
        after_cursor = focused.after_cursor if focused else ""
        tree_bundle = build_context_bundle_from_tree(
            fixture.tree,
            token_budget=500,
            overview_token_budget=120,
        )
        tree_overview = tree_bundle.top_down_context if tree_bundle else None
        if subtree_context is None and tree_bundle is not None:
            subtree_context = tree_bundle.bottom_up_context

        if mode == AutocompleteMode.CONTINUATION:
            context = ctx_store.get_continuation_context(
                before_cursor=before_cursor,
                after_cursor=after_cursor,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                cross_app_context="",
                subtree_context=subtree_context,
                tree_overview=tree_overview,
                memory_context="",
                focused_state=_focused_state_payload(focused),
            )
        else:
            context = ctx_store.get_reply_context(
                conversation_turns=turn_dicts,
                source_app=app_name,
                window_title=window_title,
                source_url=source_url,
                draft_text=before_cursor,
                cross_app_context="",
                subtree_context=subtree_context,
                tree_overview=tree_overview,
                memory_context="",
                focused_state=_focused_state_payload(focused),
            )

        # Use the same prompt builder as the live streaming pipeline.
        system_prompt, user_prompt = build_messages(
            mode=mode,
            context=context,
            num_suggestions=num_suggestions,
            streaming=True,  # include STREAMING_JSON_INSTRUCTION
            source_app=app_name,
        )
    finally:
        ctx_store.close()

    return context, system_prompt, user_prompt, tree_overview


def _focused_state_payload(focused) -> dict[str, object] | None:
    """Match the live pipeline's focused-state payload."""
    if focused is None:
        return None
    return {
        "role": focused.role,
        "insertion_point": focused.insertion_point,
        "selection_length": getattr(focused, "selection_length", 0),
        "value_length": len(focused.value),
        "placeholder_detected": focused.placeholder_detected,
        "raw_placeholder_value": getattr(focused, "raw_placeholder_value", ""),
    }


# ---------------------------------------------------------------------------
# LLM call (optional) — streaming with client reuse
# ---------------------------------------------------------------------------

# Module-level client cache for connection reuse across calls.
_client_cache: dict[str, object] = {}


def _get_cached_client(base_url: str, api_key: str):
    """Return a cached OpenAI client for the given base_url."""
    cache_key = base_url or "__default__"
    if cache_key not in _client_cache:
        import openai
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _client_cache[cache_key] = openai.OpenAI(**kwargs)
    return _client_cache[cache_key]


@dataclass
class LLMResult:
    """Result from an LLM call with per-suggestion timing."""
    suggestions: list[str]
    ttft: float  # Time to first suggestion (seconds)
    per_suggestion_times: list[float]  # Elapsed time when each suggestion completed
    total: float  # Total wall time
    provider: str  # Which provider/model was used
    streamed: bool  # Whether streaming was used


# Anthropic client cache
_anthropic_client = None


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    extract_fn,
) -> LLMResult:
    """Call Anthropic API using the native SDK."""
    import anthropic

    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=api_key)

    provider_label = f"anthropic/{model}"

    try:
        if stream:
            t0 = time.time()
            suggestions = []
            per_times = []
            ttft = 0.0
            json_buf = ""
            last_yielded = 0

            with _anthropic_client.messages.stream(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            ) as response:
                for text_chunk in response.text_stream:
                    json_buf += text_chunk
                    complete = extract_fn(json_buf)
                    while len(complete) > last_yielded:
                        s = complete[last_yielded]
                        elapsed = time.time() - t0
                        if last_yielded == 0:
                            ttft = elapsed
                        per_times.append(elapsed)
                        suggestions.append(s)
                        last_yielded += 1

            total = time.time() - t0

            # Fallback: parse full buffer
            if not suggestions and json_buf.strip():
                try:
                    data = json.loads(json_buf)
                    for s in data.get("suggestions", []):
                        text = s.get("text", "") if isinstance(s, dict) else str(s)
                        if text.strip():
                            suggestions.append(text.strip())
                    ttft = total
                    per_times = [total] * len(suggestions)
                except json.JSONDecodeError:
                    suggestions = [json_buf.strip()]
                    ttft = total
                    per_times = [total]

            return LLMResult(
                suggestions=suggestions, ttft=ttft,
                per_suggestion_times=per_times, total=total,
                provider=provider_label, streamed=True,
            )
        else:
            t0 = time.time()
            response = _anthropic_client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            total = time.time() - t0
            content = response.content[0].text if response.content else ""

            suggestions = []
            try:
                data = json.loads(content)
                for s in data.get("suggestions", []):
                    text = s.get("text", "") if isinstance(s, dict) else str(s)
                    if text.strip():
                        suggestions.append(text.strip())
            except json.JSONDecodeError:
                suggestions = [content.strip()]

            return LLMResult(
                suggestions=suggestions, ttft=total,
                per_suggestion_times=[total] * len(suggestions), total=total,
                provider=provider_label, streamed=False,
            )
    except Exception as e:
        return LLMResult(
            suggestions=[f"(LLM error: {e})"], ttft=0.0,
            per_suggestion_times=[0.0], total=0.0,
            provider=provider_label, streamed=stream,
        )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    mode,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    stream: bool = True,
) -> LLMResult:
    """Call the LLM and return suggestions with timing breakdown.

    Uses streaming by default (mirroring the live pipeline) and caches
    the OpenAI client for connection reuse across calls.
    """
    from autocompleter.config import Config, _load_dotenv
    from autocompleter.suggestion_engine import (
        AutocompleteMode,
        _extract_complete_suggestions,
        _strip_think_tags,
    )

    _load_dotenv()
    config = Config()

    base_url = base_url or config.llm_base_url
    api_key = api_key or config.openai_api_key
    model = model or config.llm_model

    temp = 0.3 if mode == AutocompleteMode.CONTINUATION else 0.7
    max_tokens = 80 if mode == AutocompleteMode.CONTINUATION else 200

    # Anthropic SDK path (separate from OpenAI-compatible)
    if base_url == "anthropic":
        api_key = api_key or config.anthropic_api_key
        return _call_anthropic(
            system_prompt, user_prompt, model, api_key,
            temp, max_tokens, stream, _extract_complete_suggestions,
        )

    host = base_url.split("//")[1].split("/")[0] if base_url and "//" in base_url else "?"
    provider_label = f"{host}/{model}"

    # Disable Qwen3 thinking mode on providers that support it
    extra_body: dict = {}
    if base_url:
        from urllib.parse import urlparse
        hostname = urlparse(base_url).hostname or ""
        if hostname.endswith("groq.com"):
            extra_body["reasoning_effort"] = "none"

    try:
        client = _get_cached_client(base_url, api_key)

        # GPT-5+ models require max_completion_tokens instead of max_tokens
        _tok_key = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"

        if stream:
            # Streaming path — mirrors the live pipeline
            t0 = time.time()
            create_kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=temp,
            )
            create_kwargs[_tok_key] = max_tokens
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**create_kwargs)
            json_buf = ""
            last_yielded = 0
            suggestions = []
            per_times = []
            ttft = 0.0

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                content = delta.content if delta else None
                if content:
                    json_buf += content
                    complete = _extract_complete_suggestions(json_buf)
                    while len(complete) > last_yielded:
                        s = complete[last_yielded]
                        elapsed = time.time() - t0
                        if last_yielded == 0:
                            ttft = elapsed
                        per_times.append(elapsed)
                        suggestions.append(s)
                        last_yielded += 1

            total = time.time() - t0

            # Fallback: parse full buffer if streaming yielded nothing
            if not suggestions and json_buf.strip():
                try:
                    json_buf = _strip_think_tags(json_buf)
                    data = json.loads(json_buf)
                    for s in data.get("suggestions", []):
                        text = s.get("text", "") if isinstance(s, dict) else str(s)
                        if text.strip():
                            suggestions.append(text.strip())
                    ttft = total
                    per_times = [total] * len(suggestions)
                except json.JSONDecodeError:
                    suggestions = [json_buf.strip()]
                    ttft = total
                    per_times = [total]

            return LLMResult(
                suggestions=suggestions, ttft=ttft,
                per_suggestion_times=per_times, total=total,
                provider=provider_label, streamed=True,
            )
        else:
            # Blocking path
            t0 = time.time()
            create_kwargs_block: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp,
            )
            create_kwargs_block[_tok_key] = max_tokens
            if extra_body:
                create_kwargs_block["extra_body"] = extra_body
            response = client.chat.completions.create(**create_kwargs_block)
            total = time.time() - t0
            content = _strip_think_tags(response.choices[0].message.content or "")

            suggestions = []
            try:
                data = json.loads(content)
                for s in data.get("suggestions", []):
                    text = s.get("text", "") if isinstance(s, dict) else str(s)
                    if text.strip():
                        suggestions.append(text.strip())
            except json.JSONDecodeError:
                suggestions = [content.strip()]

            return LLMResult(
                suggestions=suggestions, ttft=total,
                per_suggestion_times=[total] * len(suggestions), total=total,
                provider=provider_label, streamed=False,
            )
    except Exception as e:
        return LLMResult(
            suggestions=[f"(LLM error: {e})"], ttft=0.0,
            per_suggestion_times=[0.0], total=0.0,
            provider=provider_label, streamed=stream,
        )


# ---------------------------------------------------------------------------
# Pretty-print sections (modeled after dump_pipeline.py)
# ---------------------------------------------------------------------------

def _indent(text: str, prefix: str = "  | ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) if text else prefix + "(empty)"


def _section(f, title: str) -> None:
    f.write(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


def replay(
    fixture_path: str,
    mode_override: str | None = None,
    both_modes: bool = False,
    do_call_llm: bool = False,
    num_suggestions: int = 3,
    out_file=None,
    provider_base_url: str | None = None,
    provider_api_key: str | None = None,
    provider_model: str | None = None,
):
    """Run the full replay pipeline and print results."""
    from autocompleter.suggestion_engine import (
        AutocompleteMode,
        MODE_THRESHOLD_CHARS,
        detect_mode,
    )

    f = out_file or sys.stdout
    fixture = load_normalized_fixture(fixture_path)

    app_name = fixture.app
    window_title = fixture.window_title
    source_url = fixture.source_url
    tree = fixture.tree

    f.write(f"\nReplay: {fixture_path}\n")
    f.write(f"App: {app_name} | Window: {window_title!r}\n")
    if source_url:
        f.write(f"URL: {source_url}\n")
    f.write(f"Captured: {fixture.captured_at or '?'}\n")
    f.write(f"Artifact type: {fixture.artifact_type}\n")
    if fixture.notes:
        f.write(f"Notes: {fixture.notes}\n")

    # --- A. Focused Element ---
    _section(f, "A. FOCUSED ELEMENT")
    focused = build_focused_element(fixture)
    if focused:
        val_display = focused.value.replace("\n", "\\n")
        if len(val_display) > 200:
            val_display = val_display[:200] + "..."
        f.write(f"  Role: {focused.role}\n")
        f.write(f"  Value ({len(focused.value)} chars): \"{val_display}\"\n")

        before = focused.before_cursor
        after = focused.after_cursor
        before_display = before.replace("\n", "\\n")[:200]
        after_display = after.replace("\n", "\\n")[:200]
        f.write(f"  Before cursor ({len(before)} chars): \"{before_display}\"\n")
        f.write(f"  After cursor ({len(after)} chars): \"{after_display}\"\n")
        f.write(f"  Insertion point: {focused.insertion_point} | Selection: {focused.selection_length}\n")
        f.write(f"  Placeholder detected: {focused.placeholder_detected}\n")
    else:
        f.write("  (no focused element found in fixture)\n")

    # --- B. Visible Text Elements ---
    _section(f, "B. VISIBLE TEXT ELEMENTS")
    visible_text = collect_text_from_dict(tree)
    f.write(f"  Count: {len(visible_text)} elements\n")
    for i, elem in enumerate(visible_text):
        display = elem.replace("\n", "\\n")
        if len(display) > 120:
            display = display[:120] + "..."
        f.write(f"  [{i}]: \"{display}\"\n")
        if i >= 29:
            f.write(f"  ... ({len(visible_text) - 30} more)\n")
            break

    # --- C. Conversation Extraction ---
    _section(f, "C. CONVERSATION EXTRACTION")
    turns, extractor_name = extract_conversation_turns(
        tree,
        app_name=app_name,
        window_title=window_title,
    )
    f.write(f"  Extractor: {extractor_name}\n")
    if turns is None:
        f.write("  No conversation turns extracted (returned None)\n")
    elif len(turns) == 0:
        f.write("  Conversation turns: 0 (empty list)\n")
    else:
        f.write(f"  Turns extracted: {len(turns)}\n")
        for i, turn in enumerate(turns):
            text_display = turn.text.replace("\n", "\\n")
            if len(text_display) > 120:
                text_display = text_display[:120] + "..."
            f.write(f"  [{i}] {turn.speaker}: \"{text_display}\"\n")
            if i >= 19:
                f.write(f"  ... ({len(turns) - 20} more turns)\n")
                break

    # --- D. Subtree Context ---
    _section(f, "D. SUBTREE CONTEXT")
    subtree = extract_subtree(tree)
    if subtree:
        f.write(f"  Length: {len(subtree)} chars\n")
        f.write(_indent(subtree) + "\n")
    else:
        f.write("  (no focused path found — subtree walker returned None)\n")

    # --- E/F/G: Mode, Context, Prompts ---
    # Even without a focused element, we can still assemble context in
    # reply mode (conversation turns don't require cursor position).
    can_proceed = focused or turns or mode_override
    if can_proceed:
        before_cursor = focused.before_cursor if focused else ""

        # Mode detection
        if mode_override:
            detected_mode = AutocompleteMode[mode_override.upper()]
        elif focused:
            detected_mode = detect_mode(before_cursor=before_cursor)
        elif turns:
            # No focused element but we have conversation turns → reply mode
            detected_mode = AutocompleteMode.REPLY
        else:
            detected_mode = AutocompleteMode.REPLY

        modes_to_show = [detected_mode]
        if both_modes:
            other = (
                AutocompleteMode.REPLY
                if detected_mode == AutocompleteMode.CONTINUATION
                else AutocompleteMode.CONTINUATION
            )
            modes_to_show.append(other)

        for mode in modes_to_show:
            # --- E. Mode Detection ---
            _section(f, f"E. MODE: {mode.name}")
            stripped_len = len(before_cursor.strip())
            f.write(f"  Before cursor stripped length: {stripped_len} chars\n")
            f.write(f"  Threshold: {MODE_THRESHOLD_CHARS} chars\n")
            if mode_override:
                f.write(f"  (overridden via --mode {mode_override})\n")
            if not focused:
                f.write("  (no focused element — using conversation turns for context)\n")

            # --- F. Assembled Context ---
            _section(f, f"F. ASSEMBLED CONTEXT ({mode.name})")
            context, system_prompt, user_prompt, tree_overview = assemble_context(
                fixture, focused, turns, subtree,
                mode, num_suggestions,
            )
            if tree_overview:
                f.write("  Focus path overview:\n")
                f.write(_indent(tree_overview) + "\n")
                f.write("  ---\n")
            f.write(_indent(context) + "\n")

            # --- G. LLM Prompts ---
            _section(f, f"G. SYSTEM PROMPT ({mode.name})")
            f.write(_indent(system_prompt) + "\n")

            _section(f, f"H. USER PROMPT ({mode.name})")
            f.write(_indent(user_prompt) + "\n")

            # --- I. LLM Call (optional) ---
            if do_call_llm:
                _section(f, f"I. LLM SUGGESTIONS ({mode.name})")
                f.write("  Calling LLM...\n")
                f.flush()
                result = call_llm(
                    system_prompt, user_prompt, mode,
                    base_url=provider_base_url,
                    api_key=provider_api_key,
                    model=provider_model,
                )
                f.write(f"  Provider: {result.provider}\n")
                f.write(f"  Streamed: {result.streamed}\n")
                f.write(f"  TTFT: {result.ttft:.3f}s\n")
                f.write(f"  Total: {result.total:.3f}s\n")
                f.write(f"  Suggestions ({len(result.suggestions)}):\n")
                for i, s in enumerate(result.suggestions):
                    t = result.per_suggestion_times[i] if i < len(result.per_suggestion_times) else 0.0
                    f.write(f"    [{i+1}] ({t:.3f}s) {s}\n")
    else:
        _section(f, "E-I. SKIPPED")
        f.write("  (no focused element or conversation turns — cannot assemble context)\n")

    f.write(f"\n{'=' * 60}\n")
    f.write("Done.\n")
    f.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Provider presets: name → (base_url, env_var_for_key, default_model)
# base_url="anthropic" is a sentinel for the Anthropic SDK path.
PROVIDER_PRESETS: dict[str, tuple[str, str, str]] = {
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "qwen-3-235b-a22b-instruct-2507"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "qwen/qwen3-32b"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-nano"),
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
    "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "Qwen3-32B"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "Qwen/Qwen3-235B-A22B-FP8"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "accounts/fireworks/models/qwen3-235b-a22b"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "Qwen/Qwen3-235B-A22B"),
}


def _resolve_provider(name: str) -> tuple[str, str, str]:
    """Resolve a provider preset name to (base_url, api_key, model)."""
    from autocompleter.config import _load_dotenv
    _load_dotenv()
    preset = PROVIDER_PRESETS.get(name.lower())
    if preset is None:
        available = ", ".join(sorted(PROVIDER_PRESETS.keys()))
        raise ValueError(f"Unknown provider {name!r}. Available: {available}")
    base_url, env_var, model = preset
    api_key = _os.environ.get(env_var, "")
    if not api_key:
        raise ValueError(f"Provider {name!r} requires {env_var} environment variable")
    return base_url, api_key, model


def main():
    parser = argparse.ArgumentParser(
        description="Replay a saved AX tree fixture through the autocompleter pipeline",
    )
    parser.add_argument(
        "fixture",
        help="Path to the JSON fixture file",
    )
    parser.add_argument(
        "--mode", choices=["continuation", "reply"],
        default=None,
        help="Override auto-detected mode",
    )
    parser.add_argument(
        "--both-modes", action="store_true",
        help="Show context for both CONTINUATION and REPLY modes",
    )
    parser.add_argument(
        "--call-llm", action="store_true",
        help="Actually call the LLM and print suggestions",
    )
    parser.add_argument(
        "--num-suggestions", type=int, default=3,
        help="Number of suggestions (default: 3)",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help=f"Provider preset: {', '.join(sorted(PROVIDER_PRESETS.keys()))}",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Custom LLM base URL (overrides --provider)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Custom model name (overrides --provider default)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    # Resolve provider overrides
    provider_base_url = args.base_url
    provider_api_key = None
    provider_model = args.model

    if args.provider:
        base_url, api_key, default_model = _resolve_provider(args.provider)
        provider_base_url = provider_base_url or base_url
        provider_api_key = api_key
        provider_model = provider_model or default_model
        # --call-llm implied when --provider is set
        args.call_llm = True

    out_file = None
    if args.output:
        out_file = open(args.output, "w")

    try:
        replay(
            fixture_path=args.fixture,
            mode_override=args.mode,
            both_modes=args.both_modes,
            do_call_llm=args.call_llm,
            num_suggestions=args.num_suggestions,
            out_file=out_file,
            provider_base_url=provider_base_url,
            provider_api_key=provider_api_key,
            provider_model=provider_model,
        )
    finally:
        if out_file:
            out_file.close()
            print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
