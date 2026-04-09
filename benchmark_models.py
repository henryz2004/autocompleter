#!/usr/bin/env python3
"""Benchmark harness for comparing LLM models on autocomplete tasks.

Runs every (model, scenario) combination via streaming APIs, measures
latency metrics, scores response quality, and writes detailed results.
Supports multiple runs per combo for statistical significance.

Usage:
    python benchmark_models.py
    python benchmark_models.py --runs 3 --delay 1.5
    python benchmark_models.py --providers groq cerebras
    python benchmark_models.py --scenarios continuation_mid_sentence reply_slack
    python benchmark_models.py --runs 5 --providers groq cerebras --scenarios continuation_code
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path so we can import from autocompleter
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autocompleter.config import _load_dotenv
from autocompleter.suggestion_engine import (
    AutocompleteMode,
    _extract_complete_suggestions,
    build_messages,
)

# ---------------------------------------------------------------------------
# Provider / model definitions
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    base_url: str | None  # None = SDK default
    env_var: str
    sdk: str  # "openai" or "anthropic"


PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai"),
    "cerebras": ProviderConfig("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "openai"),
    "sambanova": ProviderConfig("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "openai"),
    "fireworks": ProviderConfig("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "openai"),
    "together": ProviderConfig("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "openai"),
    "deepinfra": ProviderConfig("deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "openai"),
    "openrouter": ProviderConfig("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai"),
    "openai": ProviderConfig("openai", None, "OPENAI_API_KEY", "openai"),
    "anthropic": ProviderConfig("anthropic", None, "ANTHROPIC_API_KEY", "anthropic"),
}


@dataclass
class ModelSpec:
    provider: str
    model_id: str


MODELS: list[ModelSpec] = [
    # Groq
    ModelSpec("groq", "openai/gpt-oss-20b"),
    ModelSpec("groq", "openai/gpt-oss-120b"),
    ModelSpec("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"),
    ModelSpec("groq", "qwen/qwen3-32b"),
    # Cerebras
    ModelSpec("cerebras", "gpt-oss-120b"),
    ModelSpec("cerebras", "zai-glm-4.7"),
    ModelSpec("cerebras", "qwen-3-235b-a22b-instruct-2507"),
    # SambaNova
    ModelSpec("sambanova", "Qwen3-32B"),
    # Fireworks
    ModelSpec("fireworks", "accounts/fireworks/models/gpt-oss-20b"),
    ModelSpec("fireworks", "accounts/fireworks/models/gpt-oss-120b"),
    ModelSpec("fireworks", "accounts/fireworks/models/glm-4p7"),
    ModelSpec("fireworks", "accounts/fireworks/models/deepseek-v3p1"),
    ModelSpec("fireworks", "accounts/fireworks/models/deepseek-v3p2"),
    # Together
    ModelSpec("together", "openai/gpt-oss-20b"),
    ModelSpec("together", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
    ModelSpec("together", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
    ModelSpec("together", "deepseek-ai/DeepSeek-V3.1"),
    ModelSpec("together", "Qwen/Qwen3.5-397B-A17B"),
    # DeepInfra (needs DEEPINFRA_API_KEY)
    ModelSpec("deepinfra", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ModelSpec("deepinfra", "deepseek-ai/DeepSeek-V3.1"),
    # OpenRouter (needs OPENROUTER_API_KEY)
    ModelSpec("openrouter", "qwen/qwen3-235b-a22b-2507"),
    ModelSpec("openrouter", "qwen/qwen3.5-397b-a17b"),
    ModelSpec("openrouter", "qwen/qwen3.5-35b-a3b"),
    # OpenAI
    ModelSpec("openai", "gpt-5-nano"),
    # Anthropic
    ModelSpec("anthropic", "claude-haiku-4-5-20251001"),
]

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    mode: str  # "continuation" or "reply"
    temperature: float
    context: str
    before_cursor: str  # text before cursor (used as current_input for continuation)
    description: str
    max_tokens: int = 200         # per-mode: production uses config.max_tokens (200)
    source_app: str = ""          # passed to build_messages for shell prompt selection
    shell_mode: bool | None = None  # explicit override for shell prompts


NUM_SUGGESTIONS = 3

SCENARIOS: list[Scenario] = [
    # ===================================================================
    # CONTINUATION MODE (temp 0.3, max_tokens 200)
    #
    # Production context format (get_continuation_context):
    #   Tier 3: "App: {app} | Window: {title} | URL: {url}"
    #   Tier 2.5a: cross-app context (optional)
    #   Tier 2: "Visible context:\n{visible text}"
    #   Tier 2.5b: "Background context (lower priority):\n{semantic}"
    #   Tier 1: "Text before cursor:\n{before}\n\nText after cursor:\n{after}"
    # Joined by "\n\n"
    # ===================================================================
    Scenario(
        id="continuation_mid_sentence",
        mode="continuation",
        temperature=0.3,
        context=(
            "App: Slack | Window: #engineering — Acme Corp\n\n"
            "Visible context:\n"
            "Alice: Has anyone looked at the new caching layer?\n"
            "Bob: Yeah, I ran some benchmarks yesterday. Latency dropped by 40%.\n"
            "Alice: Nice! What about memory usage?\n\n"
            "Text before cursor:\n"
            "I think the best approach would be to"
        ),
        before_cursor="I think the best approach would be to",
        description="Mid-sentence continuation in Slack",
        source_app="Slack",
    ),
    Scenario(
        id="continuation_code",
        mode="continuation",
        temperature=0.3,
        context=(
            "App: Visual Studio Code | Window: analytics.py — project\n\n"
            "Visible context:\n"
            "def compute_metrics(values: list[float]) -> dict:\n"
            '    """Compute summary statistics for a list of values."""\n'
            "    total = sum(values)\n"
            "    count = len(values)\n\n"
            "Text before cursor:\n"
            "    avg = "
        ),
        before_cursor="    avg = ",
        description="Code completion in VS Code",
        source_app="Visual Studio Code",
    ),
    Scenario(
        id="continuation_formal_email",
        mode="continuation",
        temperature=0.3,
        context=(
            "App: Mail | Window: New Message\n\n"
            "Visible context:\n"
            "To: hiring-committee@example.com\n"
            "Subject: Candidate Recommendation — Jane Doe\n\n"
            "Dear Hiring Committee,\n\n"
            "Text before cursor:\n"
            "I would like to recommend that we"
        ),
        before_cursor="I would like to recommend that we",
        description="Formal email continuation",
        source_app="Mail",
    ),
    Scenario(
        id="continuation_short_input",
        mode="continuation",
        temperature=0.3,
        context=(
            "App: Notes | Window: Meeting Notes\n\n"
            "Text before cursor:\n"
            "Not"
        ),
        before_cursor="Not",
        description="Very short input (3 chars, boundary case)",
        source_app="Notes",
    ),
    Scenario(
        id="continuation_tui_draft",
        mode="continuation",
        temperature=0.3,
        context=(
            "App: Terminal | Window: claude — ~/project\n\n"
            "Visible context:\n"
            "Conversation:\n"
            "- User: Can you help me refactor the authentication module to use JWT tokens?\n"
            "- Claude: I'll help refactor the authentication module. Let me look at the "
            "current implementation first.\n\n"
            "Text before cursor:\n"
            "I also want to make sure the refresh token"
        ),
        before_cursor="I also want to make sure the refresh token",
        description="TUI draft continuation in Claude Code",
        source_app="Terminal",
        shell_mode=False,  # TUI inside terminal — use generic prompts
    ),
    Scenario(
        id="continuation_cross_app",
        mode="continuation",
        temperature=0.3,
        context=(
            "App: Slack | Window: #engineering — Acme Corp\n\n"
            "Recently visited:\n"
            "  [Visual Studio Code — auth_service.py] def refresh_token(user_id: str) -> Token: "
            "\"\"\"Issue a new JWT refresh token.\"\"\" ...\n"
            "  [Chrome — Jira: AUTH-1234] Ticket: Migrate auth to JWT. "
            "Acceptance: refresh tokens must auto-rotate on use.\n\n"
            "Visible context:\n"
            "Bob: How's the JWT migration going?\n"
            "Alice: Almost done, just finishing the refresh token logic.\n\n"
            "Text before cursor:\n"
            "The refresh token rotation is working but I still need to"
        ),
        before_cursor="The refresh token rotation is working but I still need to",
        description="Continuation with cross-app context (VS Code + Jira)",
        source_app="Slack",
    ),
    # ===================================================================
    # REPLY MODE (temp 0.8, max_tokens 200)
    #
    # Production context format (get_reply_context):
    #   Tier 3: "App: {app} | Channel: {title} | URL: {url}"
    #   Tier 2.5: cross-app context (optional)
    #   Tier 1: "Conversation:\n- Speaker: text\n- Speaker: text"
    #     OR fallback: "Visible page content (no conversation detected):\n..."
    #   Tier 1.5: "Background context (lower priority):\n{semantic}"
    #   Tier 2: "Draft so far:\n{draft}"
    # Joined by "\n\n"
    # ===================================================================
    Scenario(
        id="reply_slack",
        mode="reply",
        temperature=0.8,
        context=(
            "App: Slack | Channel: #standup — Acme Corp\n\n"
            "Conversation:\n"
            "- Alice: Good morning! Here's my standup:\n"
            "- Yesterday: Finished the auth migration\n"
            "- Today: Starting on the dashboard redesign\n"
            "- Blockers: Waiting on design specs from UI team\n"
            "- Bob: Morning all!\n"
            "- Yesterday: Code reviews and bug fixes\n"
            "- Today: Performance optimization sprint\n"
            "- Blockers: None"
        ),
        before_cursor="",
        description="Slack standup thread — suggest reply",
        source_app="Slack",
    ),
    Scenario(
        id="reply_imessage",
        mode="reply",
        temperature=0.8,
        context=(
            "App: Messages | Channel: Chat with Alex\n\n"
            "Conversation:\n"
            "- Alex: Hey! Are you free this weekend? Was thinking we could check out "
            "that new ramen place downtown"
        ),
        before_cursor="",
        description="Casual iMessage — suggest reply",
        source_app="Messages",
    ),
    Scenario(
        id="reply_whatsapp",
        mode="reply",
        temperature=0.8,
        context=(
            "App: WhatsApp | Channel: Chat with Mom\n\n"
            "Conversation:\n"
            "- Mom: Hi sweetie! Just wanted to check in. Dad and I are thinking "
            "of visiting next month. Would the 15th work for you?"
        ),
        before_cursor="",
        description="WhatsApp family reply",
        source_app="WhatsApp",
    ),
    Scenario(
        id="reply_empty_email",
        mode="reply",
        temperature=0.8,
        context=(
            "App: Mail | Channel: New Message\n\n"
            "Visible page content (no conversation detected):\n"
            "To: team@example.com\n"
            "Subject: (empty)"
        ),
        before_cursor="",
        description="Empty email compose — suggest conversation starter",
        source_app="Mail",
    ),
    Scenario(
        id="reply_imessage_draft",
        mode="reply",
        temperature=0.8,
        context=(
            "App: Messages | Channel: Chat with Alex\n\n"
            "Conversation:\n"
            "- Alex: Hey! Are you free this weekend? Was thinking we could check out "
            "that new ramen place downtown\n\n"
            "Draft so far:\n"
            "Yeah sounds good! What time were you"
        ),
        before_cursor="Yeah sounds good! What time were you",
        description="iMessage draft completion",
        source_app="Messages",
    ),
    Scenario(
        id="reply_visible_fallback",
        mode="reply",
        temperature=0.8,
        context=(
            "App: Chrome | Channel: Gmail — Inbox | URL: https://mail.google.com\n\n"
            "Visible page content (no conversation detected):\n"
            "From: Sarah Chen <sarah@example.com>\n"
            "Subject: Q3 Planning Session — Thursday 2pm\n"
            "Date: March 3, 2026\n\n"
            "Hi team,\n\n"
            "I'd like to schedule our Q3 planning session for this Thursday at 2pm. "
            "Please review the OKR draft in the shared doc before the meeting. "
            "If Thursday doesn't work, let me know your availability for Friday.\n\n"
            "Thanks,\nSarah"
        ),
        before_cursor="",
        description="Reply to email via Chrome (visible text fallback, no turns extracted)",
        source_app="Chrome",
    ),
    # ===================================================================
    # SHELL MODE
    #
    # Production context format (get_shell_context):
    #   "App: {app} | Window: {title}"
    #   [cross-app context]
    #   "Recent commands:\n  cmd1\n  cmd2"
    #   "Recent terminal session:\n$ cmd\noutput..."
    #   "Command being typed:\n{current_command}" OR "Command line is empty (at prompt)."
    #   "Prompt: {prompt_string}"
    # Uses SYSTEM_PROMPT_SHELL_COMPLETION / SYSTEM_PROMPT_SHELL_REPLY
    # ===================================================================
    Scenario(
        id="shell_continuation",
        mode="continuation",
        temperature=0.2,  # shell continuation temp
        context=(
            "App: Terminal | Window: zsh — ~/project\n\n"
            "Recent commands:\n"
            "  git status\n"
            "  git add -A\n"
            "  npm test\n"
            "  git log --oneline -5\n\n"
            "Recent terminal session:\n"
            "$ npm test\n"
            "\n"
            "> project@1.0.0 test\n"
            "> jest --coverage\n"
            "\n"
            "PASS  src/auth.test.ts\n"
            "PASS  src/api.test.ts\n"
            "Tests: 42 passed, 42 total\n"
            "Coverage: 87.3%\n\n"
            "Command being typed:\n"
            "git commit -m \"\n\n"
            "Prompt: user@macbook ~/project % "
        ),
        before_cursor="git commit -m \"",
        description="Shell command completion — git commit message",
        source_app="Terminal",
        shell_mode=True,
    ),
    Scenario(
        id="shell_reply",
        mode="reply",
        temperature=0.5,  # shell reply temp
        context=(
            "App: iTerm2 | Window: zsh — ~/project\n\n"
            "Recent commands:\n"
            "  cd ~/project\n"
            "  python -m pytest tests/ -v\n"
            "  git diff\n\n"
            "Recent terminal session:\n"
            "$ python -m pytest tests/ -v\n"
            "\n"
            "FAILED tests/test_auth.py::test_refresh_token - AssertionError: "
            "expected 200 got 401\n"
            "FAILED tests/test_auth.py::test_token_expiry - TimeoutError\n"
            "2 failed, 15 passed\n\n"
            "Command line is empty (at prompt).\n\n"
            "Prompt: user@macbook ~/project % "
        ),
        before_cursor="",
        description="Shell command suggestion — after test failures",
        source_app="iTerm2",
        shell_mode=True,
    ),
]

SCENARIO_MAP: dict[str, Scenario] = {s.id: s for s in SCENARIOS}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    ttft_ms: float | None = None       # Time to first token
    ttfs_ms: float | None = None       # Time to first suggestion (via incremental parser)
    total_ms: float = 0.0              # Total latency
    tokens_received: int = 0           # Approximate token count (chunks)
    tok_per_sec: float = 0.0           # tokens / total_seconds
    json_valid: bool = False           # Full json.loads() succeeds
    suggestions_parsed: int = 0        # Via _extract_complete_suggestions (incremental)
    suggestions_json: int = 0          # Via full json.loads
    raw_response: str = ""             # Full accumulated response
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "ttfs_ms": round(self.ttfs_ms, 2) if self.ttfs_ms is not None else None,
            "total_ms": round(self.total_ms, 2),
            "tokens_received": self.tokens_received,
            "tok_per_sec": round(self.tok_per_sec, 1),
            "json_valid": self.json_valid,
            "suggestions_parsed": self.suggestions_parsed,
            "suggestions_json": self.suggestions_json,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Quality assessment of a single benchmark response."""
    json_compliance: float = 0.0       # 1.0 = valid JSON, 0.5 = parseable via incremental, 0.0 = broken
    count_accuracy: float = 0.0        # 1.0 = exact count, 0.5 = close, 0.0 = wrong
    no_duplicates: float = 0.0         # 1.0 = all unique, penalty per duplicate
    no_prefix_repeat: float = 0.0      # 1.0 = none repeat before_cursor, 0.0 = all do
    length_appropriate: float = 0.0    # 1.0 = within expected range, penalty for too short/long
    no_think_tags: float = 0.0         # 1.0 = no <think> in output, 0.0 = has think tags
    overall: float = 0.0              # Weighted composite

    def to_dict(self) -> dict:
        return {
            "json_compliance": round(self.json_compliance, 3),
            "count_accuracy": round(self.count_accuracy, 3),
            "no_duplicates": round(self.no_duplicates, 3),
            "no_prefix_repeat": round(self.no_prefix_repeat, 3),
            "length_appropriate": round(self.length_appropriate, 3),
            "no_think_tags": round(self.no_think_tags, 3),
            "overall": round(self.overall, 3),
        }


def score_quality(
    suggestions: list[str],
    metrics: RunMetrics,
    scenario: Scenario,
    expected_count: int = NUM_SUGGESTIONS,
) -> QualityScore:
    """Score the quality of a single benchmark response.

    Weights:
        json_compliance:   0.15  (can we parse it at all?)
        count_accuracy:    0.20  (did we get the right number of suggestions?)
        no_duplicates:     0.15  (are all suggestions distinct?)
        no_prefix_repeat:  0.20  (continuation: does it NOT repeat before_cursor?)
        length_appropriate:0.15  (reasonable length for the mode?)
        no_think_tags:     0.15  (no reasoning/think leakage in output?)
    """
    score = QualityScore()

    # --- JSON compliance ---
    if metrics.json_valid:
        score.json_compliance = 1.0
    elif metrics.suggestions_parsed > 0:
        # Incremental parser worked but strict JSON failed (e.g. markdown fences, think tags)
        score.json_compliance = 0.5
    else:
        score.json_compliance = 0.0

    # --- Count accuracy ---
    n = len(suggestions)
    if n == expected_count:
        score.count_accuracy = 1.0
    elif n == expected_count - 1 or n == expected_count + 1:
        score.count_accuracy = 0.5
    else:
        score.count_accuracy = 0.0

    # --- No duplicates ---
    if n > 0:
        unique = len(set(s.strip().lower() for s in suggestions))
        score.no_duplicates = unique / n
    else:
        score.no_duplicates = 0.0

    # --- No prefix repeat (continuation mode) ---
    before = scenario.before_cursor.strip()
    if before and n > 0:
        repeaters = 0
        for s in suggestions:
            s_stripped = s.strip()
            # Check if suggestion starts with the before_cursor text
            if s_stripped.lower().startswith(before.lower()):
                repeaters += 1
        score.no_prefix_repeat = 1.0 - (repeaters / n)
    elif n > 0:
        # Reply mode with no before_cursor — always passes
        score.no_prefix_repeat = 1.0
    else:
        score.no_prefix_repeat = 0.0

    # --- Length appropriate ---
    is_shell = getattr(scenario, "shell_mode", None) is True
    if n > 0:
        good_count = 0
        for s in suggestions:
            length = len(s.strip())
            if is_shell:
                # Shell: commands are typically 3-200 chars
                if 3 <= length <= 200:
                    good_count += 1
                elif length > 0:
                    good_count += 0.5
            elif scenario.mode == "continuation":
                # Continuation: expect 3-150 chars (few words to half a sentence)
                if 3 <= length <= 150:
                    good_count += 1
                elif length > 0:
                    good_count += 0.5  # partial credit
            else:
                # Reply: expect 5-500 chars (short message to paragraph)
                if 5 <= length <= 500:
                    good_count += 1
                elif length > 0:
                    good_count += 0.5
        score.length_appropriate = good_count / n
    else:
        score.length_appropriate = 0.0

    # --- No think tags ---
    raw = metrics.raw_response
    has_think = "<think>" in raw.lower() or "</think>" in raw.lower()
    has_reasoning = raw.lstrip().startswith("<think")
    if not has_think:
        score.no_think_tags = 1.0
    elif has_reasoning and n == expected_count:
        # Has think tags but still produced good output — partial credit
        score.no_think_tags = 0.5
    else:
        score.no_think_tags = 0.0

    # --- Weighted overall ---
    score.overall = (
        0.15 * score.json_compliance +
        0.20 * score.count_accuracy +
        0.15 * score.no_duplicates +
        0.20 * score.no_prefix_repeat +
        0.15 * score.length_appropriate +
        0.15 * score.no_think_tags
    )

    return score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_mode(scenario: Scenario) -> AutocompleteMode:
    """Convert scenario mode string to AutocompleteMode enum."""
    if scenario.mode == "continuation":
        return AutocompleteMode.CONTINUATION
    return AutocompleteMode.REPLY


def _scenario_messages(scenario: Scenario) -> tuple[str, str]:
    """Build (system, user) messages for a scenario using production code."""
    return build_messages(
        mode=_scenario_mode(scenario),
        context=scenario.context,
        num_suggestions=NUM_SUGGESTIONS,
        max_suggestion_lines=10,
        streaming=True,
        source_app=scenario.source_app,
        shell_mode=scenario.shell_mode,
    )

# ---------------------------------------------------------------------------
# Streaming runners
#
# The incremental parse loop mirrors _call_llm_stream() in
# suggestion_engine.py: on each chunk we call _extract_complete_suggestions()
# so the benchmark exercises the exact same parsing algorithm as production.
# ---------------------------------------------------------------------------

def _create_with_fallbacks(client, kwargs: dict):
    """Call chat.completions.create, retrying with adjusted params on 400 errors.

    Handles providers that reject ``max_tokens`` (want ``max_completion_tokens``)
    or reject specific ``temperature`` values. Retries up to 3 times since
    each retry may reveal a different unsupported parameter.
    """
    for _attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e)
            changed = False
            if "max_tokens" in err and "max_completion_tokens" in err:
                val = kwargs.pop("max_tokens", 200)
                kwargs["max_completion_tokens"] = val
                changed = True
            if "temperature" in err:
                kwargs.pop("temperature", None)
                changed = True
            if not changed:
                raise
    # Should not reach here, but just in case
    return client.chat.completions.create(**kwargs)


def run_openai_compatible(
    model_spec: ModelSpec,
    provider_cfg: ProviderConfig,
    scenario: Scenario,
) -> RunMetrics:
    """Run a scenario against an OpenAI-compatible provider via streaming."""
    import openai

    api_key = os.environ.get(provider_cfg.env_var, "")
    client_kwargs: dict = {"api_key": api_key}
    if provider_cfg.base_url:
        client_kwargs["base_url"] = provider_cfg.base_url
    client = openai.OpenAI(**client_kwargs)

    system, user_msg = _scenario_messages(scenario)
    metrics = RunMetrics()
    json_buf = ""
    last_parsed = 0  # mirrors last_yielded in _call_llm_stream

    t0 = time.perf_counter()
    try:
        create_kwargs: dict = {
            "model": model_spec.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": True,
            "temperature": scenario.temperature,
            "max_tokens": scenario.max_tokens,
        }
        # Qwen 3.5 thinks by default — disable to get plain JSON output.
        if "qwen3.5" in model_spec.model_id.lower():
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        response = _create_with_fallbacks(client, create_kwargs)
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            # Reasoning models (gpt-oss, etc.) stream thinking in
            # reasoning_content/reasoning before the actual content.
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                metrics.tokens_received += 1
                continue
            content = delta.content
            if content:
                if metrics.ttft_ms is None:
                    metrics.ttft_ms = (time.perf_counter() - t0) * 1000
                json_buf += content
                metrics.tokens_received += 1
                # Incremental parse — same as production _call_llm_stream
                complete = _extract_complete_suggestions(json_buf)
                if len(complete) > last_parsed:
                    if metrics.ttfs_ms is None:
                        metrics.ttfs_ms = (time.perf_counter() - t0) * 1000
                    last_parsed = len(complete)
    except Exception as e:
        metrics.error = str(e)

    metrics.total_ms = (time.perf_counter() - t0) * 1000
    metrics.raw_response = json_buf
    metrics.suggestions_parsed = last_parsed
    _finalize_metrics(metrics, json_buf)
    return metrics


def run_anthropic(
    model_spec: ModelSpec,
    provider_cfg: ProviderConfig,
    scenario: Scenario,
) -> RunMetrics:
    """Run a scenario against the Anthropic API via streaming."""
    import anthropic

    api_key = os.environ.get(provider_cfg.env_var, "")
    client = anthropic.Anthropic(api_key=api_key)

    system, user_msg = _scenario_messages(scenario)
    metrics = RunMetrics()
    json_buf = ""
    last_parsed = 0  # mirrors last_yielded in _call_llm_stream

    t0 = time.perf_counter()
    try:
        with client.messages.stream(
            model=model_spec.model_id,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            temperature=scenario.temperature,
            max_tokens=scenario.max_tokens,
        ) as stream:
            for text_chunk in stream.text_stream:
                if metrics.ttft_ms is None:
                    metrics.ttft_ms = (time.perf_counter() - t0) * 1000
                json_buf += text_chunk
                metrics.tokens_received += 1
                # Incremental parse — same as production _call_llm_stream
                complete = _extract_complete_suggestions(json_buf)
                if len(complete) > last_parsed:
                    if metrics.ttfs_ms is None:
                        metrics.ttfs_ms = (time.perf_counter() - t0) * 1000
                    last_parsed = len(complete)
    except Exception as e:
        metrics.error = str(e)

    metrics.total_ms = (time.perf_counter() - t0) * 1000
    metrics.raw_response = json_buf
    metrics.suggestions_parsed = last_parsed
    _finalize_metrics(metrics, json_buf)
    return metrics


def _finalize_metrics(metrics: RunMetrics, json_buf: str) -> None:
    """Compute derived metrics from the raw response buffer."""
    # Strict JSON validity (independent of incremental parser)
    try:
        data = json.loads(json_buf)
        metrics.json_valid = True
        if isinstance(data, dict) and "suggestions" in data:
            metrics.suggestions_json = len(data["suggestions"])
    except (json.JSONDecodeError, ValueError):
        metrics.json_valid = False

    # Throughput
    total_sec = metrics.total_ms / 1000.0
    if total_sec > 0 and metrics.tokens_received > 0:
        metrics.tok_per_sec = metrics.tokens_received / total_sec

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _stat_summary(values: list[float]) -> dict:
    """Return mean, p50, p95, stddev for a list of values."""
    if not values:
        return {"mean": None, "p50": None, "p95": None, "stddev": None}
    return {
        "mean": statistics.mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _fmt_stat(stat: dict, unit: str = "ms") -> str:
    """Format a stat summary as a compact string."""
    if stat["mean"] is None:
        return "N/A"
    mean = stat["mean"]
    p95 = stat["p95"]
    return f"{mean:.0f}{unit} (p95={p95:.0f})"


def _fmt_avg(val: float | None, unit: str = "ms") -> str:
    """Format a single avg value."""
    if val is None:
        return "N/A"
    return f"{val:.0f}{unit}"

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _run_provider_models(
    provider_models: list[ModelSpec],
    provider_cfg: ProviderConfig,
    scenarios: list[Scenario],
    runs: int,
    delay: float,
    results_lock: threading.Lock,
    all_results: list[dict],
    key_mapping: dict[str, dict],
    completed_counter: list[int],
    total: int,
    print_lock: threading.Lock,
) -> None:
    """Run all (model, scenario, run) combos for one provider, sequentially."""
    for model_spec in provider_models:
        for scenario in scenarios:
            for run_idx in range(runs):
                run_id = str(uuid.uuid4())
                label = f"{model_spec.provider}/{model_spec.model_id}"
                run_label = f"run {run_idx + 1}/{runs}" if runs > 1 else ""

                with print_lock:
                    completed_counter[0] += 1
                    n = completed_counter[0]
                    suffix = f" ({run_label})" if run_label else ""
                    print(f"  [{n}/{total}] {label} x {scenario.id}{suffix} ... ", end="", flush=True)

                if provider_cfg.sdk == "anthropic":
                    metrics = run_anthropic(model_spec, provider_cfg, scenario)
                else:
                    metrics = run_openai_compatible(model_spec, provider_cfg, scenario)

                suggestion_texts = _extract_complete_suggestions(metrics.raw_response)
                quality = score_quality(suggestion_texts, metrics, scenario)

                with print_lock:
                    if metrics.error:
                        print(f"\n    -> ERROR: {metrics.error[:120]}")
                    elif metrics.ttft_ms is not None:
                        ttfs_part = f"TTFS={metrics.ttfs_ms:.0f}ms  " if metrics.ttfs_ms else ""
                        print(
                            f"\n    -> TTFT={metrics.ttft_ms:.0f}ms  "
                            f"{ttfs_part}"
                            f"total={metrics.total_ms:.0f}ms  "
                            f"quality={quality.overall:.2f}  "
                            f"suggestions={metrics.suggestions_parsed}"
                        )
                    else:
                        print(
                            f"\n    -> total={metrics.total_ms:.0f}ms  "
                            f"quality={quality.overall:.2f}  "
                            f"suggestions={metrics.suggestions_parsed}"
                        )

                result = {
                    "run_id": run_id,
                    "run_index": run_idx,
                    "scenario_id": scenario.id,
                    "scenario_description": scenario.description,
                    "mode": scenario.mode,
                    "suggestions": suggestion_texts,
                    "raw_response": metrics.raw_response,
                    "metrics": metrics.to_dict(),
                    "quality": quality.to_dict(),
                }
                key_entry = {
                    "provider": model_spec.provider,
                    "model": model_spec.model_id,
                }

                with results_lock:
                    all_results.append(result)
                    key_mapping[run_id] = key_entry

                time.sleep(delay)


def run_benchmark(
    models: list[ModelSpec],
    scenarios: list[Scenario],
    runs: int = 1,
    delay: float = 1.0,
) -> tuple[list[dict], dict[str, dict]]:
    """Run all (model, scenario, run) combinations.

    Providers run in parallel (one thread each); models within a provider
    run sequentially with ``delay`` between calls for rate limiting.
    """
    by_provider: dict[str, list[ModelSpec]] = defaultdict(list)
    for model_spec in models:
        provider_cfg = PROVIDERS.get(model_spec.provider)
        if not provider_cfg:
            print(f"  [SKIP] Unknown provider: {model_spec.provider}")
            continue
        api_key = os.environ.get(provider_cfg.env_var, "")
        if not api_key:
            print(f"  [SKIP] {model_spec.provider}/{model_spec.model_id} — "
                  f"no {provider_cfg.env_var}")
            continue
        by_provider[model_spec.provider].append(model_spec)

    total = sum(len(ms) for ms in by_provider.values()) * len(scenarios) * runs
    if not total:
        return [], {}

    print(f"  Running {len(by_provider)} provider(s) in parallel, "
          f"{runs} run(s) per combo\n")

    all_results: list[dict] = []
    key_mapping: dict[str, dict] = {}
    results_lock = threading.Lock()
    print_lock = threading.Lock()
    completed_counter = [0]

    threads: list[threading.Thread] = []
    for provider_name, provider_models in by_provider.items():
        provider_cfg = PROVIDERS[provider_name]
        t = threading.Thread(
            target=_run_provider_models,
            args=(
                provider_models, provider_cfg, scenarios, runs, delay,
                results_lock, all_results, key_mapping,
                completed_counter, total, print_lock,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return all_results, key_mapping

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict], key_mapping: dict[str, dict]) -> None:
    """Print comprehensive summary tables."""
    if not all_results:
        print("\nNo results to summarize.")
        return

    # Build per-model data structures
    model_results: dict[str, list[dict]] = {}
    for result in all_results:
        run_id = result["run_id"]
        info = key_mapping[run_id]
        label = f"{info['provider']}/{info['model']}"
        model_results.setdefault(label, []).append(result)

    _print_overall_table(model_results)
    _print_latency_table(model_results)
    _print_quality_table(model_results)
    _print_scenario_breakdown(model_results)


def _print_overall_table(model_results: dict[str, list[dict]]) -> None:
    """Print the main overview table."""
    print("\n" + "=" * 120)
    print("OVERALL SUMMARY")
    print("=" * 120)

    col_m = 45
    col = 12

    header = (
        f"{'Model':<{col_m}}"
        f"{'Runs':>{col}}"
        f"{'TTFT avg':>{col}}"
        f"{'TTFS avg':>{col}}"
        f"{'Total avg':>{col}}"
        f"{'Quality':>{col}}"
        f"{'JSON OK':>{col}}"
        f"{'Errors':>{col}}"
    )
    print(header)
    print("-" * len(header))

    for label, results in sorted(model_results.items()):
        n = len(results)
        ttfts = [r["metrics"]["ttft_ms"] for r in results if r["metrics"]["ttft_ms"] is not None]
        ttfss = [r["metrics"]["ttfs_ms"] for r in results if r["metrics"]["ttfs_ms"] is not None]
        totals = [r["metrics"]["total_ms"] for r in results if not r["metrics"]["error"]]
        qualities = [r["quality"]["overall"] for r in results if not r["metrics"]["error"]]
        json_ok = sum(1 for r in results if r["metrics"]["json_valid"])
        errors = sum(1 for r in results if r["metrics"]["error"])

        avg_ttft = _fmt_avg(statistics.mean(ttfts)) if ttfts else "N/A"
        avg_ttfs = _fmt_avg(statistics.mean(ttfss)) if ttfss else "N/A"
        avg_total = _fmt_avg(statistics.mean(totals)) if totals else "N/A"
        avg_qual = f"{statistics.mean(qualities):.3f}" if qualities else "N/A"

        print(
            f"{label:<{col_m}}"
            f"{n:>{col}}"
            f"{avg_ttft:>{col}}"
            f"{avg_ttfs:>{col}}"
            f"{avg_total:>{col}}"
            f"{avg_qual:>{col}}"
            f"{f'{json_ok}/{n}':>{col}}"
            f"{errors:>{col}}"
        )

    print("=" * len(header))


def _print_latency_table(model_results: dict[str, list[dict]]) -> None:
    """Print detailed latency statistics with p50, p95, stddev."""
    print("\n" + "=" * 130)
    print("LATENCY DETAILS (ms)")
    print("=" * 130)

    col_m = 45
    col = 14

    header = (
        f"{'Model':<{col_m}}"
        f"{'TTFT p50':>{col}}"
        f"{'TTFT p95':>{col}}"
        f"{'TTFS p50':>{col}}"
        f"{'TTFS p95':>{col}}"
        f"{'Total p50':>{col}}"
        f"{'Total p95':>{col}}"
    )
    print(header)
    print("-" * len(header))

    for label, results in sorted(model_results.items()):
        ok_results = [r for r in results if not r["metrics"]["error"]]
        ttfts = [r["metrics"]["ttft_ms"] for r in ok_results if r["metrics"]["ttft_ms"] is not None]
        ttfss = [r["metrics"]["ttfs_ms"] for r in ok_results if r["metrics"]["ttfs_ms"] is not None]
        totals = [r["metrics"]["total_ms"] for r in ok_results]

        ttft_stats = _stat_summary(ttfts)
        ttfs_stats = _stat_summary(ttfss)
        total_stats = _stat_summary(totals)

        print(
            f"{label:<{col_m}}"
            f"{_fmt_avg(ttft_stats['p50']):>{col}}"
            f"{_fmt_avg(ttft_stats['p95']):>{col}}"
            f"{_fmt_avg(ttfs_stats['p50']):>{col}}"
            f"{_fmt_avg(ttfs_stats['p95']):>{col}}"
            f"{_fmt_avg(total_stats['p50']):>{col}}"
            f"{_fmt_avg(total_stats['p95']):>{col}}"
        )

    print("=" * len(header))


def _print_quality_table(model_results: dict[str, list[dict]]) -> None:
    """Print quality score breakdown."""
    print("\n" + "=" * 130)
    print("QUALITY BREAKDOWN (avg across runs)")
    print("=" * 130)

    col_m = 45
    col = 14

    header = (
        f"{'Model':<{col_m}}"
        f"{'JSON':>{col}}"
        f"{'Count':>{col}}"
        f"{'NoDupes':>{col}}"
        f"{'NoRepeat':>{col}}"
        f"{'Length':>{col}}"
        f"{'NoThink':>{col}}"
        f"{'Overall':>{col}}"
    )
    print(header)
    print("-" * len(header))

    for label, results in sorted(model_results.items()):
        ok_results = [r for r in results if not r["metrics"]["error"]]
        if not ok_results:
            print(f"{label:<{col_m}}  (all errors)")
            continue

        def _avg(key: str) -> float:
            vals = [r["quality"][key] for r in ok_results]
            return statistics.mean(vals)

        print(
            f"{label:<{col_m}}"
            f"{_avg('json_compliance'):>{col}.3f}"
            f"{_avg('count_accuracy'):>{col}.3f}"
            f"{_avg('no_duplicates'):>{col}.3f}"
            f"{_avg('no_prefix_repeat'):>{col}.3f}"
            f"{_avg('length_appropriate'):>{col}.3f}"
            f"{_avg('no_think_tags'):>{col}.3f}"
            f"{_avg('overall'):>{col}.3f}"
        )

    print("=" * len(header))


def _print_scenario_breakdown(model_results: dict[str, list[dict]]) -> None:
    """Print per-scenario latency and quality for each model."""
    print("\n" + "=" * 120)
    print("PER-SCENARIO BREAKDOWN")
    print("=" * 120)

    # Collect all scenario IDs present
    scenario_ids: list[str] = []
    seen = set()
    for results in model_results.values():
        for r in results:
            sid = r["scenario_id"]
            if sid not in seen:
                scenario_ids.append(sid)
                seen.add(sid)

    col_m = 45
    col_s = 25
    col = 12

    for sid in scenario_ids:
        print(f"\n--- {sid} ---")
        header = (
            f"{'Model':<{col_m}}"
            f"{'Runs':>{col}}"
            f"{'Avg TTFS':>{col}}"
            f"{'Avg Total':>{col}}"
            f"{'Quality':>{col}}"
            f"{'Sugg Count':>{col}}"
        )
        print(header)
        print("-" * len(header))

        for label, results in sorted(model_results.items()):
            scenario_results = [r for r in results if r["scenario_id"] == sid and not r["metrics"]["error"]]
            if not scenario_results:
                continue

            n = len(scenario_results)
            ttfss = [r["metrics"]["ttfs_ms"] for r in scenario_results if r["metrics"]["ttfs_ms"] is not None]
            totals = [r["metrics"]["total_ms"] for r in scenario_results]
            qualities = [r["quality"]["overall"] for r in scenario_results]
            sugg_counts = [r["metrics"]["suggestions_parsed"] for r in scenario_results]

            avg_ttfs = _fmt_avg(statistics.mean(ttfss)) if ttfss else "N/A"
            avg_total = _fmt_avg(statistics.mean(totals)) if totals else "N/A"
            avg_qual = f"{statistics.mean(qualities):.3f}" if qualities else "N/A"
            avg_sugg = f"{statistics.mean(sugg_counts):.1f}" if sugg_counts else "N/A"

            print(
                f"{label:<{col_m}}"
                f"{n:>{col}}"
                f"{avg_ttfs:>{col}}"
                f"{avg_total:>{col}}"
                f"{avg_qual:>{col}}"
                f"{avg_sugg:>{col}}"
            )

    print("\n" + "=" * 120)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models for autocomplete quality and latency",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs per (model, scenario) combination (default: 1)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Only test these providers (e.g. --providers groq anthropic)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Only test models whose ID contains one of these substrings "
             "(e.g. --models qwen3 glm gpt-5)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Only run these scenario IDs (e.g. --scenarios continuation_mid_sentence reply_slack)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write results (default: project root)",
    )
    args = parser.parse_args()

    # Load .env from project root
    _load_dotenv()

    # Filter models by provider if requested
    models = MODELS
    if args.providers:
        allowed = set(p.lower() for p in args.providers)
        models = [m for m in models if m.provider in allowed]
        if not models:
            print(f"Error: no models match providers {args.providers}")
            print(f"Available providers: {', '.join(PROVIDERS.keys())}")
            sys.exit(1)

    # Filter models by ID substring
    if args.models:
        filtered = []
        for m in models:
            if any(substr.lower() in m.model_id.lower() for substr in args.models):
                filtered.append(m)
        models = filtered
        if not models:
            print(f"Error: no models match --models {args.models}")
            sys.exit(1)

    # Filter scenarios if requested
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = []
        for sid in args.scenarios:
            if sid in SCENARIO_MAP:
                scenarios.append(SCENARIO_MAP[sid])
            else:
                print(f"Warning: unknown scenario '{sid}', skipping")
        if not scenarios:
            print(f"Error: no valid scenarios specified")
            print(f"Available: {', '.join(SCENARIO_MAP.keys())}")
            sys.exit(1)

    total_runs = len(models) * len(scenarios) * args.runs
    print(f"Benchmark: {len(models)} model(s) x {len(scenarios)} scenario(s) "
          f"x {args.runs} run(s) = {total_runs} total calls")
    print(f"Delay between calls: {args.delay}s\n")

    all_results, key_mapping = run_benchmark(
        models, scenarios, runs=args.runs, delay=args.delay,
    )

    # Write results
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_path = output_dir / "benchmark_responses.jsonl"
    with open(responses_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    print(f"\nResults written to: {responses_path}")

    key_path = output_dir / "benchmark_key.json"
    with open(key_path, "w") as f:
        json.dump(key_mapping, f, indent=2)
    print(f"Key mapping written to: {key_path}")

    # Print summary tables
    print_summary(all_results, key_mapping)


if __name__ == "__main__":
    main()
