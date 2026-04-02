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

# Clear the context database
rm -f ~/.autocompleter/context.db

# Dump AX tree of focused app (debugging tool)
python dump_ax_tree.py
python dump_ax_tree.py -o /tmp/ax_dump.log

# Dump full context pipeline (shows what gets sent to LLM)
python dump_pipeline.py
python dump_pipeline.py --both-modes -o /tmp/pipeline.log
```

## Architecture

### Data flow

Hotkey trigger or auto-trigger (typing pause) → `app.py` captures focused element + visible content → `context_store.py` assembles tiered context (metadata → cross-app → visible text → semantic → cursor state) → `suggestion_engine.py` streams LLM response → `overlay.py` shows suggestions incrementally → `text_injector.py` or `cdp_injector.py` injects accepted text.

### Threading model

Four threads interact carefully:
- **Event tap thread** (hotkey): Must return within ~1s or macOS disables the CGEvent tap. Dispatches work immediately.
- **Main thread**: AppKit event loop + UI updates via `_run_on_main()` queue.
- **Worker threads**: One daemon thread per LLM trigger. Generation ID counter (`_generation_id`) discards stale results when user re-triggers.
- **Observer thread**: Polls visible content with dynamic intervals, feeds context store and context trail (`context_trail.py` tracks app/window switches for cross-app context).

### Auto-trigger

`_Debouncer` in `app.py` provides event-driven auto-trigger on typing pauses. Configurable via `auto_trigger_enabled`, `auto_trigger_delay_ms`, and `auto_trigger_cooldown_ms` in config. Replaces the older polling-based approach.

### Mode-aware pipeline

`detect_mode()` in `suggestion_engine.py` classifies input:
- **CONTINUATION** (before_cursor >= 3 chars): Low temperature (0.3), short max tokens (80), predicts next words.
- **REPLY** (empty/short input): Higher temperature (0.8), more tokens (200), suggests full response.

Each mode has its own system prompt, context assembly strategy in `context_store.py`, and temperature/token config.

### LLM integration

Two paths:
- **Blocking** (`_call_llm`): Uses `instructor.from_provider()` with Pydantic `SuggestionList` model (validates exact suggestion count via `min_length`, retries up to `max_retries=2`).
- **Streaming** (`_call_llm_stream`): Uses raw Anthropic/OpenAI SDK with plain-text JSON output (`{"suggestions": [{"text": ...}]}`). Appends `STREAMING_JSON_INSTRUCTION` to the system prompt. Incrementally parses the growing JSON buffer via `_extract_complete_suggestions()` — yields each `Suggestion` as soon as its `{"text": "..."}` object closes. Falls back to blocking path on failure or zero results.

### Context assembly tiers (continuation mode)

1. **Metadata**: App name, window title, URL
2. **Cross-app**: Recent snapshots from other apps via `context_trail.py`
3. **Live surroundings**: Visible text from window (preferred) or DB entries
4. **Semantic**: Embeddings-based retrieval if enabled (`embeddings.py`, TF-IDF default)
5. **Cursor state**: Before/after cursor text (always included, raw)

### Cross-app context

`context_trail.py` maintains a rolling deque of `AppSnapshot` objects. On each app/window switch, it snapshots the previous app's visible text. `get_recent_cross_app_context()` retrieves snapshots from other apps within a time window (default 60s, max 3 entries) for inclusion in LLM context.

### Conversation extraction

`conversation_extractors.py` has pluggable per-app extractors (Gemini, Slack, ChatGPT, Claude Desktop, iMessage, ActionDelimited) that parse AX trees into `ConversationTurn` objects. Falls back to `GenericExtractor` heuristic (short child = speaker, long child = body). `get_extractor(app_name)` dispatches.

### Shell parsing

`shell_parser.py` parses terminal buffers to extract prompt, current command, and recent history (up to 15 commands). Handles zsh/bash prompt patterns and skips terminal UI chrome (Warp separators, status bars).

### Text injection strategies

`text_injector.py` tries in order: AX value setting → CDP (`cdp_injector.py`) → clipboard paste → keystrokes. Placeholder-detected fields skip AX (bypasses web app JS handlers, routes through clipboard which triggers placeholder-clearing). CDP injection uses Chrome DevTools Protocol `Input.insertText` for Chromium-based apps (requires `--remote-debugging-port`), falling back to `document.execCommand('insertText')`.

### Feedback loop

Accepts/dismisses recorded in SQLite `suggestion_feedback` table. Accept rate feeds `adjust_temperature()` (< 30% → lower temp). Recently dismissed patterns appended to system prompt as negative examples.

## Key patterns

- **Coordinate systems**: AX API uses top-left origin (Y down). AppKit uses bottom-left (Y up). Conversion: `appkit_y = primary_screen_height - ax_y`. Handled in `overlay.py`.
- **Placeholder detection**: 4 strategies — (1) AXPlaceholderValue match, (2) AXNumberOfCharacters==0, (3) cursor at position 0 + short value, (4) app-specific known prefixes (`_APP_PLACEHOLDER_PREFIXES`). Routes injection through clipboard paste.
- **Web app depth handling**: `_collect_text` starts from AXWebArea for PWAs/Electron, skipping Chromium scaffolding. Adjacent `AXStaticText` fragments concatenated via `_collect_child_text()`.
- **SQLite thread safety**: Per-thread connections via `threading.local()`, WAL mode.
- **Config loading**: `config.py` loads from `.env` file at project root + env vars (`AUTOCOMPLETER_LLM_PROVIDER`, `AUTOCOMPLETER_LLM_MODEL`, `AUTOCOMPLETER_HOTKEY`). API keys from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.
- **AX utilities**: `ax_utils.py` provides shared helpers (`ax_get_attribute`, `ax_set_attribute`, `ax_get_position`, `dump_ax_tree`, `serialize_ax_tree`) used across `input_observer.py`, `text_injector.py`, and `conversation_extractors.py`.
- **Python >=3.11** required (see `pyproject.toml`).

## Testing

Tests use `pytest` with fixtures in `tests/fixtures/ax_trees/` (~30 JSON AX tree dumps from Claude, ChatGPT, Gemini, Terminal, Slack, etc.). The `tests/conftest.py` provides an `ax_tree_fixture` factory for loading these. Extractor regression tests in `test_extractor_regression.py` compare against expected outputs in `tests/fixtures/expected/`. Use `tests/generate_expected.py` to regenerate expected outputs after intentional extractor changes.

## Debugging

Use `--log-file` for full DEBUG output. Key log prefixes:
- `[CTX]` — context extraction and assembly
- `--- TRIGGER ---` — hotkey fired, focused element info
- `--- SUGGESTIONS ---` — LLM results with generation ID and timing
- `--- CONTEXT ---` — full context sent to the LLM

`dump_ax_tree.py` dumps the full AX tree of the focused app for debugging content extraction. `dump_pipeline.py` shows the assembled context that would be sent to the LLM (use `--both-modes` to see both continuation and reply contexts).

## LLM Provider Benchmarks

See `benchmark_analysis.md` for the full report. Summary of top configurations (1008 runs, 24 provider/model combos):

| Tier | Provider | Model | Quality | Avg Latency |
|---|---|---|---|---|
| 1 | Cerebras | Qwen3-235B | 1.000 | 254ms |
| 1 | Groq | Qwen3-32B | 1.000 | 293ms |
| 1 | Groq | GPT-OSS-20B | 1.000 | 345ms |
| 2 | SambaNova | Qwen3-32B | 1.000 | 650ms |
| 2 | Together | GPT-OSS-20B | 1.000 | 749ms |

**Recommended production config**: Cerebras Qwen3-235B (primary), Groq Qwen3-32B (fallback). Same model can vary up to 14x in latency across providers.
- Commit significant changes after making them.
- you are in charge of running and restarting the autocompleter as a background process and examining logs from the process.