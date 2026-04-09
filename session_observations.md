# Autocomplete Session Observations — 2026-02-23

## Session Setup
- **Model**: Cerebras `qwen-3-235b-a22b-instruct-2507`
- **App under test**: ChatGPT (macOS app)
- **Mode**: All triggers were continuation mode (no reply mode triggers observed)
- **Total triggers**: 34 (only 7 produced suggestions — rest debounced or from observer)
- **Temperature**: 0.2 (continuation mode)
- **Max tokens**: 256

---

## Trigger Log

| # | Time | Before Cursor | Suggestions | Notes |
|---|------|---------------|-------------|-------|
| 1 | 21:45:07 | "no girl i'm saying we're going from cancun to isla mujeres " (58 chars) | "in the morning and coming back the same day." / "on the first ferry, like 5:30 am, so we can maximize the day." / "—just check the Ultramar schedule for the early departure." | Good quality, contextual |
| 2 | 21:45:22 | "...we're staying on the island. also when do bars " (139 chars) | "on the island usually close?" / "do they shut down by midnight?" / "close over there? trying to plan the night." | Good, natural completions |
| 3 | 21:46:21 | "what if we get tired " (20 chars) | "we can always head back early and skip the late ferry" / "just grab a taxi to the hotel and rest up" / "maybe book a daytime return and chill on the beach instead" | **ISSUE**: These are 3rd-person advice, not continuing the user's question. User is asking "what if we get tired..." and the model should continue the question, not answer it. |
| 4 | 21:46:28 | "what if we get tired before the ferry " (37 chars) | "and just crash at a friend's place in town instead?" / "and bail on the early ferry altogether?" / "should we book a room near the dock as backup?" | Better — question continuations, but still slightly off-voice (the user doesn't appear to use "bail" style language) |
| 5 | 21:46:55 | "yeah tell me, and ho wmuch " (26 chars, typo "ho w") | "much does a last-minute room usually run?" / "much time do we have between the last ferry and clubs closing?" / "much does a taxi to puerto juárez cost at that hour?" | Good recovery from typo. Correctly continues "how much" |
| 6 | 21:49:16 | "just heard its a law in cancun that bars only open " (50 chars) | "from 8 PM to 4 AM, so nothing runs all night." / "after 8 PM and must close by 3 AM, no exceptions." / "at 8 PM and can't serve past 3 AM, so plan accordingly." | **ISSUE**: Model is hallucinating/confabulating specific legal facts (8 PM, 3 AM, 4 AM). The autocomplete shouldn't generate specific factual claims the user hasn't indicated. |
| 7 | 21:49:41 | "why were you saying " (19 chars) | "the cutoff was 11 pm when it's actually later?" / "it shut down at 11 when most places go till 3?" / "bars closed so early when the rule's actually 2 or 3 am?" | Good — references the conversation context accurately |

---

## Issues Found

### 1. CRITICAL: Conversation turns always = 0 in ChatGPT app
Every trigger logged `conversation_turns: 0` even though the user is in a multi-turn ChatGPT conversation. The ChatGPT extractor isn't parsing the conversation structure, so the model only sees raw visible text without speaker attribution. This means:
- The LLM can't tell which messages are the user's and which are ChatGPT's
- In continuation mode, it may try to continue ChatGPT's response instead of the user's input
- Cross-turn context (who said what) is lost

### 2. CRITICAL: "Related context" is just keystroke echo
The "Related context" section in the prompt contains the same text as "before cursor" repeated with slight truncations (the observer capturing each keystroke as a separate entry):
```
Related context:
what if we get tired before the ferry
what if we get tired before the
what if we get tired before
```
This is wasting ~30-50% of the context window on repetitive keystroke snapshots instead of useful cross-app or historical context. The context store seems to be recording every keystroke delta as a separate entry, then the "related context" retrieval pulls back the most recent N entries, which are all just slightly shorter versions of the same input.

### 3. CRITICAL: Negative patterns list has massive duplication
The dismissed suggestions list appended to the system prompt has the same 3 suggestions repeated 5x:
```
- Thanks for the info! So the first ferry back to Cancún from Isla Mujeres leaves around 6 am?
- Got it. Do you know which ferry company operates the latest night service?
- That makes sense. Should I book the return ticket in advance to be safe?
```
These appear 5 times each in the negative patterns, inflating the system prompt by ~500+ tokens with redundant content. This wastes context window and may over-bias the model against these patterns.

### 4. MEDIUM: Continuation mode suggests answers instead of continuations (Trigger #3)
When the user typed "what if we get tired " (clearly mid-question), the model generated 3 answers/advice rather than continuing the question. This is likely because:
- The visible context is ChatGPT's long response about ferry options
- Without conversation turn attribution, the model may think it's continuing ChatGPT's advisory response rather than the user's question
- The "Related context" section with repeated keystroke fragments doesn't help disambiguate

### 5. MEDIUM: Model hallucinating specific facts (Trigger #6)
For "just heard its a law in cancun that bars only open ", the model confidently completed with specific hours (8 PM, 3 AM, 4 AM). These are fabricated facts. Autocomplete should be careful about generating specific factual claims the user hasn't established. One option: lower confidence on factual completions or add a prompt instruction about not inventing specific facts.

### 6. LOW: STREAMING_JSON_INSTRUCTION duplicated in system prompt
Looking at the raw request, the system prompt contains the `STREAMING_JSON_INSTRUCTION` block **twice** — once from `build_messages(streaming=True)` appending it to the system prompt, and then again from `_call_llm_stream()` which also appends it via `json_system = system + STREAMING_JSON_INSTRUCTION`. This wastes ~60 tokens.

### 7. LOW: Visible context truncated mid-sentence
In trigger #4, the visible context cuts off at "clubs in the hotel zo" — the AX API only captures what's physically visible on screen. When the ChatGPT response is long, the top portions scroll out of view. This means the model only sees a fragment. Not easily fixable but worth noting.

---

## Quality Assessment

### What's working well
- **Latency**: All triggers got first suggestion in 190-290ms (steady state). Excellent.
- **Reliability**: 7/7 triggers produced exactly 3 valid suggestions. Zero JSON failures.
- **Typo handling**: Model gracefully handled "ho wmuch" → "much does..." (trigger #5)
- **Tone matching**: Casual, lowercase style maintained consistently
- **Brevity**: The new max_tokens=256 and prompt changes are working — suggestions are appropriately short (5-15 words each)

### Improvement priorities (ranked)
1. **Fix conversation turn extraction for ChatGPT** — turns=0 means no speaker attribution
2. **Deduplicate the Related Context / keystroke echo problem** — wasting context on repeated fragments
3. **Deduplicate negative patterns list** — same dismissed suggestion shouldn't appear 5x
4. **Better continuation vs answer detection** — when user is mid-question, continue the question
5. **Factual guardrail** — don't generate specific invented facts in continuations
6. **Fix double STREAMING_JSON_INSTRUCTION** — minor token waste

---

## Session 2 — Post-Fix Observations (22:03+)

### Fixes applied
- Keystroke echo filter in `get_semantically_relevant()`
- Negative patterns dedup in `get_recent_dismissed_patterns()`
- Removed double `STREAMING_JSON_INSTRUCTION`

### Results after fix
- **Conversation turns: 15** in Claude app (extractor working — ChatGPT app had 0 turns, that's a separate issue)
- **Related context now shows actual useful content**: restaurant recommendations, island info, prior conversation — no more keystroke echoes
- **No duplicate negative patterns** in system prompt
- Suggestions are contextual and appropriately short

### New bugs observed

#### BUG: Cursor jumps to beginning after accepting suggestion
After the user accepts (Tab/Enter) a suggestion and text is injected, the cursor position jumps to the beginning of the field instead of staying at the end of the injected text. This makes the autocomplete unusable for chained typing — user has to manually click back to where they were.

**Likely cause**: The text injection method (AX value setting, clipboard paste, or keystrokes) may be replacing the entire field value without preserving/restoring cursor position, or the AX `AXSelectedTextRange` isn't being set after injection.

**Files to investigate**: `autocompleter/text_injector.py`, `autocompleter/cdp_injector.py`

#### Note: Related context still has some noise
The "Related context" section still contains some noise entries like:
- `"can you do more\xa0"` and `"can y"` — these are keystroke fragments that weren't fully filtered (the filter checks substring relationship, but these are partial fragments captured before the 5-second dedup window)
- Claude's internal thinking text leaking through: `"Good results. Let me summarize..."`, `"The user wants restaurant recommendations..."` — these are Claude's reasoning artifacts being captured by the observer and stored as context entries

**Potential improvement**: Filter out Claude/ChatGPT "thinking" artifacts from context entries (entries that look like assistant reasoning rather than user/conversation content).

### Session 2 Trigger Log

| # | Time | App | Mode | Before Cursor | Suggestions | Notes |
|---|------|-----|------|---------------|-------------|-------|
| 1 | 22:03:29 | Claude | continuation | "can you do more research into the cartel " (41 chars) | "situation in Tijuana?" / "activity in Cancun lately?" / "presence near tourist areas in Quintana Roo?" | Good continuations. Conversation turns=15 (extractor working). Rich context from trip planning conversation. |
| 2 | 22:05:18 | Claude | reply | "" (0 chars, placeholder detected) | "Thanks for the detailed breakdown on the cartel situation. Should I still feel s..." / "Got it on the cenotes—thanks. Which one would you recommend for a half-day trip..." / "Any must-try non-seafood restaurants you'd personally recommend on the island?" | First reply mode trigger! Suggestions are contextual and varied. But suggestion #1 and #2 are truncated (end mid-word) — likely hitting max_tokens=256 cutoff. |

### Session 2 Injection Observations

| # | Method | Notes |
|---|--------|-------|
| 1 | AX API | Trigger 1 — "Injected text via AX API". Cursor reportedly jumped to beginning after this. |
| 2 | Clipboard paste | Trigger 2 — "No CDP debug port found for 'Claude'", fell back to clipboard paste. This is the placeholder-detected path. |

### Session 2 Key Findings

1. **FIXED: Keystroke echo** — Related context now shows real conversation history (restaurant recs, island info) instead of repeated keystroke fragments. Major improvement.

2. **FIXED: Negative patterns duplication** — System prompt is clean, no repeats.

3. **FIXED: Double STREAMING_JSON_INSTRUCTION** — Only appears once now.

4. **NEW: Reply mode suggestions getting truncated** — Suggestion #1 ended with "Should I still feel s" (cut mid-word), suggestion #2 ended with "half-day trip " (cut mid-thought). The max_tokens=256 budget is too tight for reply mode where all 3 suggestions need to fit in one JSON response. Consider using separate max_tokens for continuation (256) vs reply (512).

5. **NEW: Claude app thinking artifacts in context** — The observer captures Claude's reasoning text ("Good results. Let me summarize...", "The user wants restaurant recommendations...") as visible page content. These get stored as context entries and pollute the "Related context" and "Visible context" sections. The conversation extractor detects 15 turns but the visible text still includes reasoning artifacts alongside actual conversation.

6. **CONFIRMED: Cursor jump bug** — After AX API injection on trigger 1, user reported cursor jumped to beginning. The AX API path likely sets AXValue without updating AXSelectedTextRange to the end of the new text.

7. **Keystroke echo filter partially working** — Still seeing `"can you do more\xa0"` and `"can y"` in Related context. The substring filter catches exact matches but doesn't handle the non-breaking space (`\xa0`) or very short fragments well. Need to normalize whitespace before comparison.

---

## Cumulative Bug/Improvement Tracker

| # | Priority | Issue | Status | Fix Location |
|---|----------|-------|--------|-------------|
| 1 | CRITICAL | Cursor jumps to beginning after injection | FIXED | `text_injector.py` — NSValue-based cursor positioning with AXValueCreate fallback |
| 2 | HIGH | ChatGPT conversation turns always = 0 | FIXED | `conversation_extractors.py` — added SectionList strategy for ChatGPT 5.x + generic fallback |
| 3 | HIGH | Reply mode suggestions truncated at 256 tokens | OPEN | `config.py` / `suggestion_engine.py` — need per-mode max_tokens |
| 4 | MEDIUM | Claude thinking artifacts in context | OPEN | `context_store.py` or `conversation_extractors.py` — filter reasoning text |
| 5 | MEDIUM | Continuation mode answers questions instead of continuing them | OPEN | `suggestion_engine.py` — prompt tuning for question continuation |
| 6 | MEDIUM | Model hallucinating specific facts | OPEN | `suggestion_engine.py` — add "don't invent facts" to prompt |
| 7 | LOW | Keystroke echo filter doesn't handle `\xa0` / whitespace variants | OPEN | `context_store.py` — normalize whitespace in echo filter |
| 8 | HIGH | ChatGPT app: only 3 text elements visible, 0 conversation turns, 31 chars context | FIXED | ChatGPT extractor updated for 5.x AX tree + all extractors now have generic fallback |
| 9 | LOW | Visible context truncated mid-sentence (AX viewport limit) | WONTFIX | Inherent to AX API — can't read off-screen content |
| 10 | DONE | Keystroke echo in Related Context | FIXED | `context_store.py:get_semantically_relevant()` |
| 11 | DONE | Negative patterns 5x duplication | FIXED | `context_store.py:get_recent_dismissed_patterns()` |
| 12 | DONE | Double STREAMING_JSON_INSTRUCTION | FIXED | `suggestion_engine.py:_call_llm_stream()` |
| 13 | DONE | Extractors too brittle to app updates | FIXED | All extractors now fall back to ActionDelimitedExtractor → GenericExtractor |
| 14 | DONE | GenericExtractor ignores AXDescription text | FIXED | `_try_parse_message_group` now reads AXDescription as fallback |
| 15 | DONE | Diagnostic logging for missing context | FIXED | `context_store.py:get_reply_context()` warns when visible_text exists but context is minimal |
| 16 | DONE | WhatsApp: wrong mode (continuation instead of reply) | FIXED | `app.py` — force reply mode when conversation turns extracted |
| 17 | DONE | WhatsApp: raw AXDescription polluting context | FIXED | `app.py` — suppress visible_text_elements when conversation turns exist |
| 18 | DONE | tmux: split-pane content interleaved in before_cursor | FIXED | `shell_parser.py` — `_strip_tmux_split_panes()` |
| 19 | DONE | tmux: TUI detection intermittently failing (shell detected instead) | FIXED | `app.py` — TUI detection runs before shell detection |
| 20 | DONE | TUI: semantic context pulling noise from other tmux pane | FIXED | `app.py` — suppress semantic context for TUI mode |
| 21 | DONE | Reply mode: suggestions replace draft instead of completing it | FIXED | `suggestion_engine.py` — prompt updated to complete "Draft so far" |

---

## Trigger Dump System

### Overview

Every hotkey trigger can now dump a full diagnostic snapshot to disk, capturing everything needed to reproduce and debug the autocomplete pipeline.

### Usage

```bash
python -m autocompleter --dump-dir dumps/ --log-file /tmp/autocompleter.log
```

Each trigger produces a JSON file like `dumps/20260305-143956-gen11-terminal.json`.

### Dump Contents

| Field | Description |
|---|---|
| `app` | App name (e.g., "Terminal", "WhatsApp") |
| `windowTitle` | Window title |
| `capturedAt` | ISO timestamp |
| `generationId` | Monotonic generation counter |
| `tree` | Full AX tree (same format as `dump_ax_tree_json.py`) |
| `focused.role` | AX role of focused element |
| `focused.beforeCursor` | Full text before cursor |
| `focused.afterCursor` | Full text after cursor |
| `focused.insertionPoint` | Cursor position (character offset) |
| `focused.valueLength` | Total text length |
| `focused.placeholderDetected` | Whether placeholder was detected |
| `detection.mode` | `continuation` or `reply` |
| `detection.useShell` | Whether shell parser was used |
| `detection.useTui` | Whether TUI (e.g., Claude Code) was detected |
| `detection.tuiName` | TUI identifier (e.g., `claude_code`) |
| `detection.tuiUserInput` | Extracted user input from TUI |
| `detection.hasConversationTurns` | Whether conversation turns were extracted |
| `detection.conversationTurnCount` | Number of turns |
| `context` | Exact context string sent to the LLM |
| `conversationTurns` | List of `{speaker, text}` dicts |
| `visibleTextElements` | Raw visible text from AX tree |
| `suggestions` | List of suggestion strings from LLM |
| `suggestionLatencyMs` | End-to-end latency in milliseconds |

### Building Tests from Dumps

The AX tree in each dump uses the same format as `dump_ax_tree_json.py`, so the existing `tests/ax_fixture_loader.py` can load them directly:

```python
from tests.ax_fixture_loader import load_fixture, load_fixture_metadata

# Load the AX tree as mock elements for testing
root = load_fixture("dumps/20260305-143956-gen11-terminal.json")

# Load metadata (app name, window title, etc.)
meta = load_fixture_metadata("dumps/20260305-143956-gen11-terminal.json")
```

To build a regression test from a dump:

```python
import json
from pathlib import Path
from unittest.mock import patch
from tests.ax_fixture_loader import load_fixture

def test_whatsapp_group_chat_extraction():
    """Regression test from captured trigger dump."""
    dump = json.loads(Path("dumps/20260305-whatsapp.json").read_text())
    root = load_fixture("dumps/20260305-whatsapp.json")

    with patch("autocompleter.conversation_extractors.ax_get_attribute",
               side_effect=_ax_dispatch):
        ext = WhatsAppExtractor()
        turns = ext.extract(root)

    # Assert against captured ground truth
    assert len(turns) == dump["detection"]["conversationTurnCount"]
```

### Workflow

1. Run with `--dump-dir dumps/` during normal usage
2. When something goes wrong, inspect the dump for that trigger
3. Copy the AX tree JSON to `tests/fixtures/ax_trees/` as a regression fixture
4. Write a test that loads the fixture and asserts expected behavior
5. Fix the extractor/parser, verify the test passes

---

## Production Autocomplete Latency Research

### Latency Targets

| Threshold | Feel | Use Case |
|---|---|---|
| <100ms | Invisible | Inline autocomplete (ideal) |
| 100-200ms | Acceptable | Maintains flow state |
| 200-300ms | Noticeable | Chat/reply suggestions |
| >300ms | Breaks rhythm | Avoid for any interactive use |

### How Production Tools Achieve Low Latency

#### Cursor Tab
- **Custom ~7B models** for inline autocomplete
- **Speculative edits**: Feed unchanged code chunks back to the model, only generate at change points. Achieves ~1,000 tok/s on 70B models (13x speedup)
- **Three-layer KV cache**: (1) cache warming as user types, (2) caching-aware prompt design with stable prefixes, (3) speculative caching of predicted next suggestions
- **Shadow workspace**: Hidden VSCode window for background LSP feedback without disrupting UI
- Partnership with **Fireworks AI** for speculative decoding API support

#### GitHub Copilot
- **Multi-model routing**: Fast small models for inline, large models for chat
- **Go HTTP/2 proxy**: Near-instant request cancellation when user keeps typing
- **Pre-filtering classifier**: Lightweight binary model predicts acceptance likelihood before calling LLM, skipping low-value suggestions
- 110-140ms inline completion latency. Handles 400M+ completions/day
- Latest custom model: 12% higher acceptance, 3x throughput, 35% lower latency

#### Supermaven
- **Custom "Babble" architecture** replacing Transformers
- 300K-1M token context window at 4K-token cost/latency
- ~250ms latency (3x faster than Copilot)
- **Edit-sequence understanding**: Sees code as sequence of diffs, not static files

#### Codeium
- **Custom-pretrained 1B-7B models** trained from scratch with fill-in-the-middle objectives
- **Riptide**: Proprietary code reasoning engine for relevance scoring (200% retrieval recall improvement vs. embeddings)
- <200ms average. Single H100 serves ~1,000 engineers

### Model Sizes by Use Case

| Use Case | Model Size | Examples |
|---|---|---|
| Inline autocomplete | 1B-7B | Qwen2.5-Coder 1.5B/3B/7B, Phi-1/3 |
| Chat/reply | 13B-70B | Llama 3.1 70B, DeepSeek Coder |
| Complex reasoning | 70B-405B | GPT-4, Claude, Llama 405B |

### Top Small Models for Code Completion

| Model | Size | HumanEval | Notes |
|---|---|---|---|
| Qwen2.5-Coder 1.5B | 1.5B | 54% | <10GB VRAM, <1s inference |
| Phi-1 | 1.3B | 50.6% | Outperforms Codex-12B |
| Qwen2.5-Coder 3B | 3B | 59% | Top 5 ranking |
| Llama 3.2 3B | 3B | -- | Rivals 10x larger models |
| Qwen2.5-Coder 7B | 7B | 65% | #1 ranking, 1.00 stability |

### Inception Mercury (Diffusion LLM)

Mercury uses **diffusion-based generation** -- tokens are produced in parallel through iterative denoising, not sequentially. Fundamentally different architecture from autoregressive transformers.

**Mercury Coder:**
- ~1,000 tok/s output throughput on H100
- OpenAI-compatible API (drop-in replacement)
- $0.25/M input, $0.75/M output
- #1 speed, tied #2 quality on Copilot Arena
- Explicitly designed for autocomplete/tab-completion workflows

**Mercury 2 (Feb 2026):**
- 1,009 tok/s on Blackwell GPUs
- 1.7s e2e latency vs. 14.4s Gemini 3 Flash, 23.4s Claude Haiku 4.5
- First reasoning-capable diffusion LLM
- Quality trails frontier models by 5-15% on complex reasoning

**Trade-offs:**
- Higher TTFT than autoregressive models (first denoising step processes entire sequence)
- Better for complete-response display than streaming
- For short completions (10-30 tokens), the parallel generation shines -- the entire response lands at once
- For longer replies (50-200+ tokens), higher TTFT hurts streaming UX

**Assessment for this project:**
- CONTINUATION mode: Excellent fit. Short completions, no streaming needed, parallel generation advantage
- REPLY mode: TTFT penalty matters since overlay shows suggestions incrementally. Cerebras/Groq may be better here

### Provider Comparison

| Provider | Architecture | Throughput (70B) | TTFT | Pricing (70B) | Best For |
|---|---|---|---|---|---|
| **Cerebras** | Wafer-scale chip | 1,800-3,000 tok/s | ~0.35s | $0.60/M | Bulk throughput |
| **Mercury** | Diffusion LLM | ~1,000 tok/s | Higher (parallel) | $0.25/$0.75/M | Short completions |
| **Groq** | Custom LPU | 241 tok/s | ~0.45s | $0.64-0.99/M | Lowest TTFT |
| **Fireworks** | Software-optimized | High | ~0.4s | $0.90/M | Structured output |
| **Together** | Standard GPUs | Moderate | ~0.5s | $0.88/M | Experimentation |
| **DeepInfra** | Cost-optimized | -- | -- | $0.36/M | Cheapest fallback |

### Latency Optimization Techniques

#### Request Cancellation
- Cancel in-flight LLM requests when user re-triggers (saves ~50% of calls)
- HTTP/2 enables fine-grained cancellation propagation
- Already partially implemented via `_generation_id` counter -- stale results are discarded but the HTTP connection isn't cancelled

#### Prefix Caching
- Structure prompts with stable prefixes (system prompt, app metadata)
- Put variable content (before_cursor, after_cursor) at the end
- Anthropic: 90% discount on cached tokens, auto-caches 1024+ token prefixes
- 50-90% cost savings, up to 85% latency reduction

#### Pre-filtering
- Lightweight local classifier predicts acceptance likelihood
- Features: recent accept/dismiss rate, cursor position, typing momentum
- Skip LLM call entirely when predicted acceptance is low

#### Semantic Caching
- Compute embedding of context before calling LLM
- Check for similar cached contexts (cosine similarity > 0.9)
- Return cached suggestions on hit (100% cost savings)
- Adds 5-20ms for similarity search, saves 1-5s by skipping LLM

#### Multi-Provider Routing
- Route by mode: CONTINUATION -> Mercury/Groq (speed), REPLY -> Cerebras (quality+throughput)
- Fallback chain: Primary -> Secondary -> Local Ollama (offline)
- Token bucket rate limiting with burst allowance

#### Local Inference
- **Qwen2.5-Coder 1.5B via Ollama**: <100ms latency, no rate limits, free
- **vLLM**: 8,033 tok/s on Blackwell (16.6x faster than Ollama), best for multi-user
- **llama.cpp**: Best mixed CPU+GPU, good for edge/constrained hardware

### Recommended Architecture

```
Trigger
  |
  v
Pre-filter classifier (local, <5ms)
  |-- low acceptance predicted --> skip
  |-- high acceptance predicted:
  v
Semantic cache check (local, ~10ms)
  |-- cache hit --> return cached suggestions
  |-- cache miss:
  v
Mode router
  |-- CONTINUATION --> Mercury Coder / local Qwen2.5-Coder 1.5B
  |-- REPLY --> Cerebras / Groq
  |-- rate limited --> fallback provider or local Ollama
  v
LLM inference (100-500ms)
  |
  v
Overlay display
```

### Key Numbers

- **Latency target**: <200ms for autocomplete, <500ms for reply
- **Inline autocomplete models**: 1B-7B parameters
- **Prefix cache savings**: 50-90% cost, up to 85% latency
- **Request cancellation savings**: ~50% of LLM calls
- **Semantic cache hit savings**: 100% (skip LLM entirely)
- **Local Qwen2.5-Coder 1.5B**: <100ms, 54% HumanEval, <2GB VRAM
- **Mercury Coder**: ~1,000 tok/s, $0.25/$0.75 per M tokens
- **Single H100 capacity**: ~1,000 engineers (Codeium benchmark)
