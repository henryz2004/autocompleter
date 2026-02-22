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

LLM outputs suggestions separated by `---SUGGESTION---` delimiters. `generate_suggestions_stream()` buffers tokens, yields each `Suggestion` as its delimiter is hit, so the overlay updates incrementally.

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

### Feedback loop

Accepts/dismisses recorded in SQLite `suggestion_feedback` table. Accept rate feeds `adjust_temperature()` (< 30% → lower temp). Recently dismissed patterns appended to system prompt as negative examples.

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
- `--- SUGGESTIONS ---` — LLM results with generation ID and timing
- `--- CONTEXT ---` — full context sent to the LLM

`dump_ax_tree.py` dumps the full AX tree of the focused app for debugging content extraction.
