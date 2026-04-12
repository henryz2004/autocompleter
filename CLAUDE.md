# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

macOS system-wide contextual autocomplete tool. Observes text input via the Accessibility API, collects context from the active window, and generates LLM-powered suggestions shown in a floating overlay.

## Commands

```bash
# Activate virtualenv
source venv/bin/activate

# Run the app
python -m autocompleter
python -m autocompleter --log-file /tmp/autocompleter.log --log-level DEBUG

# Run all tests
./venv/bin/python -m pytest tests/ -v

# Run a single test file or test
./venv/bin/python -m pytest tests/test_suggestion_engine.py -v
./venv/bin/python -m pytest tests/test_suggestion_engine.py::test_name -v

# Dump AX tree of focused app (debugging tool)
python dump_ax_tree.py
```

## Architecture

### Data flow

Hotkey trigger → `app.py` captures focused element + visible content → `context_store.py` assembles tiered context (metadata → cross-app → visible text → semantic → cursor state) → `suggestion_engine.py` streams LLM response → `overlay.py` shows suggestions incrementally → `text_injector.py` or `cdp_injector.py` injects accepted text.

### Threading model

Four threads interact carefully:
- **Event tap thread** (hotkey): Must return within ~1s or macOS disables the CGEvent tap. Dispatches work immediately.
- **Main thread**: AppKit event loop + UI updates via `_run_on_main()` queue.
- **Worker threads**: One daemon thread per LLM trigger. Generation ID counter (`_generation_id`) discards stale results when user re-triggers.
- **Observer thread**: Polls visible content every 2s, feeds context store and context trail.

### Mode-aware pipeline

`detect_mode()` in `suggestion_engine.py` classifies input:
- **CONTINUATION** (before_cursor >= 3 chars): Low temperature (0.3), short max tokens (80), predicts next words.
- **REPLY** (empty/short input): Higher temperature (0.8), more tokens (200), suggests full response.

Each mode has its own system prompt, context assembly strategy in `context_store.py`, and temperature/token config.

### Streaming path

LLM outputs JSON (`{"suggestions": [{"text": "..."}]}`). `generate_suggestions_stream()` incrementally parses the growing buffer via `_extract_complete_suggestions()`, yielding each `Suggestion` as its JSON object closes. The overlay updates with each arrival. Timeout-based escalation fires a fallback provider (Groq) if the primary (Cerebras) hasn't responded within 400ms.

### Regenerate

When the overlay is visible, `Ctrl+R` (configurable via `AUTOCOMPLETER_REGENERATE_HOTKEY`) re-triggers the LLM with the same context but fresh sampling, producing different suggestions. Validates the user is still in the same app (by PID) and re-captures the caret position to track window movement. Saved trigger state is cleared on dismiss to avoid memory leaks.

### Context assembly tiers (continuation mode)

1. **Metadata**: App name, window title, URL
2. **Cross-app**: Recent snapshots from other apps via `context_trail.py`
3. **Live surroundings**: Visible text from window (preferred) or DB entries
4. **Semantic**: Embeddings-based retrieval if enabled (`embeddings.py`, TF-IDF default)
5. **Cursor state**: Before/after cursor text (always included, raw)

### Conversation extraction

`conversation_extractors.py` has pluggable per-app extractors (Gemini, Slack, ChatGPT, Claude Desktop, iMessage) that parse AX trees into `ConversationTurn` objects. Falls back to `GenericExtractor` heuristic (short child = speaker, long child = body). `get_extractor(app_name)` dispatches.

### Text injection strategies

`text_injector.py` tries in order: AX value setting → clipboard paste → keystrokes. Placeholder-detected fields skip AX (bypasses web app JS handlers). `cdp_injector.py` provides CDP-based injection for Chromium apps via `Input.insertText` (requires `--remote-debugging-port`).

### Follow-up suggestions

After a suggestion is accepted, the system can automatically re-trigger with the updated context to offer a follow-up continuation. Controlled by `followup_after_accept_enabled` config / `AUTOCOMPLETER_FOLLOWUP_AFTER_ACCEPT` env var. Reuses the existing trigger context rather than re-capturing, so it's fast.

### Prompt management

`prompts.py` centralizes all system prompts and prompt assembly helpers. Mode-specific prompts (`SYSTEM_PROMPT_COMPLETION`, `SYSTEM_PROMPT_REPLY`) live here, along with `build_prompt_extra_rules()` for dynamic rule injection based on context (e.g., placeholder-aware rules). The pipeline is app-agnostic — no shell-specific prompts or temperature overrides. Terminal apps are treated the same as any other app for context assembly and prompt selection. `quality_review.py` defines `QualityVariant` dataclasses that control prompt/context modifications for A/B testing prompt strategies offline.

### Feedback loop

Accepts/dismisses recorded in SQLite `suggestion_feedback` table. Accept rate feeds `adjust_temperature()` (< 30% → lower temp). Recently dismissed patterns appended to system prompt as negative examples.

## Configuration

All config is in `config.py` via environment variables (prefixed `AUTOCOMPLETER_`) with sensible defaults. A `.env` file at the project root is auto-loaded. Key env vars:

- `AUTOCOMPLETER_LLM_PROVIDER` / `AUTOCOMPLETER_LLM_BASE_URL` / `AUTOCOMPLETER_LLM_MODEL` — primary LLM
- `AUTOCOMPLETER_FALLBACK_*` — fallback LLM (fires after escalation timeout)
- `AUTOCOMPLETER_HOTKEY` / `AUTOCOMPLETER_REGENERATE_HOTKEY` — keybindings
- `CEREBRAS_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY` — API keys (auto-resolved by provider URL)
- `AUTOCOMPLETER_MEMORY` — enable long-term memory (mem0-based)

Data is stored in `~/.autocompleter/` (SQLite DB, etc.).

## Key patterns

- **Coordinate systems**: AX API uses top-left origin (Y down). AppKit uses bottom-left (Y up). Conversion: `appkit_y = primary_screen_height - ax_y`. Handled in `overlay.py`.
- **Placeholder detection**: 4 strategies — (1) AXPlaceholderValue match, (2) AXNumberOfCharacters==0, (3) cursor at position 0 + short value, (4) app-specific known prefixes (`_APP_PLACEHOLDER_PREFIXES`). Routes injection through clipboard paste.
- **Web app depth handling**: `_collect_text` starts from AXWebArea for PWAs/Electron, skipping Chromium scaffolding. Adjacent `AXStaticText` fragments concatenated via `_collect_child_text()`.
- **SQLite thread safety**: Per-thread connections via `threading.local()`, WAL mode.
- **Python >=3.11** required (see `pyproject.toml`).

## Debugging

Use `--log-file` for full DEBUG output. Key log prefixes:
- `[CTX]` — context extraction and assembly
- `--- TRIGGER ---` — hotkey fired, focused element info
- `--- REGENERATE ---` — regenerate hotkey fired
- `--- SUGGESTIONS ---` — LLM results with generation ID and timing
- `--- CONTEXT ---` — full context sent to the LLM

`dump_ax_tree.py` dumps the full AX tree of the focused app for debugging content extraction.
