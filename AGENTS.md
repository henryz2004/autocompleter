# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

macOS system-wide contextual autocomplete tool. It watches the focused input via the Accessibility API, collects surrounding window context, and generates LLM-backed suggestions rendered in a floating overlay.

## Commands

```bash
# Activate virtualenv
source venv/bin/activate

# Run the app
python -m autocompleter
python -m autocompleter --log-file /tmp/autocompleter.log --log-level DEBUG

# Dump per-trigger diagnostics (AX tree + context + suggestions)
python -m autocompleter --dump-dir dumps

# Print latency stats from the last 50 triggers
python -m autocompleter --stats
python -m autocompleter --stats 100

# Run all tests
./venv/bin/python -m pytest tests/ -v

# Run a single test file or test
./venv/bin/python -m pytest tests/test_suggestion_engine.py -v
./venv/bin/python -m pytest tests/test_suggestion_engine.py::test_name -v

# Dump AX tree of focused app
python dump_ax_tree.py
python dump_ax_tree.py -o /tmp/ax_dump.log

# Dump assembled pipeline context for debugging
python dump_pipeline.py
python dump_pipeline.py --both-modes
```

## Architecture

### Data flow

Hotkey or auto-trigger -> `app.py` captures the focused element, conversation metadata, and subtree XML near the caret -> terminal/TUI inputs are normalized via `shell_parser.py` when the focused app is a terminal -> `context_store.py` assembles mode-specific context from metadata, cross-app trail, subtree XML, conversation turns, and optional memory -> `suggestion_engine.py` streams JSON suggestions from the configured LLM with fallback escalation -> `overlay.py` renders suggestions incrementally -> `text_injector.py` injects accepted text via AX/CDP/clipboard/keystrokes.

### Threading model

Five execution paths matter:
- **Event tap thread**: Hotkey callbacks must return quickly or macOS disables the tap. Heavy work is dispatched immediately.
- **Main thread**: AppKit loop plus all overlay/UI work via `_run_on_main()`.
- **Worker threads**: One generation thread per trigger; `_generation_id` drops stale streams/results.
- **Observer thread**: Polls visible content with dynamic backoff, stores context, updates context trail, and pre-warms memory.
- **Debouncer thread**: Powers auto-trigger after typing pauses.

### Trigger model

- Manual trigger: `AUTOCOMPLETER_HOTKEY` (default `ctrl+space`)
- Auto-trigger toggle: `shift+<hotkey>`
- Regenerate: `AUTOCOMPLETER_REGENERATE_HOTKEY` (default `ctrl+r`)
- Navigation: `up` / `down` / `tab` / `return` / `shift+tab` / `escape` / `1..3`

### Mode-aware pipeline

`detect_mode()` in `suggestion_engine.py` uses `before_cursor`, not the full field value:
- **CONTINUATION**: `len(before_cursor.strip()) >= 3`
- **REPLY**: shorter or empty draft

Shell apps also switch to shell-specific prompts. Terminal buffers can be routed through `parse_terminal_buffer()` for shell command suggestions or `parse_tui_buffer()` for Claude Code-like TUIs so prose/chat context is not mistaken for shell input.

### Streaming and fallback

`generate_suggestions_stream()` uses raw provider clients and incrementally parses JSON from the growing output buffer via `_extract_complete_suggestions()`. Some providers emit `<think>` blocks before JSON; `_strip_think_tags()` removes those safely. If the primary provider has not produced output within `AUTOCOMPLETER_ESCALATION_TIMEOUT_MS` (default 400ms), a fallback OpenAI-compatible provider is fired in parallel.

### Context assembly tiers

`context_store.py` builds different context for continuation, reply, and shell modes. Current inputs are assembled from:
1. **Metadata**: app name, window title, URL
2. **Cross-app trail**: recent app/window snapshots from `context_trail.py`
3. **Long-term memory**: optional mem0-backed facts/style hints from `memory.py`
4. **Live local context**: conversation turns and subtree XML from `subtree_context.py`
5. **Cursor/draft state**: `before_cursor`, `after_cursor`, or parsed shell command state

`context_store.py` no longer persists flat visible-text history for prompt assembly. The SQLite DB is now primarily for `suggestion_feedback`; cross-app context is held in-memory by `context_trail.py`.

### Conversation extraction

`conversation_extractors.py` contains dedicated extractors for:
- Gemini
- Slack
- ChatGPT
- Claude Desktop
- iMessage / Messages
- WhatsApp
- Discord

Browser-hosted chat UIs are dispatched by window title keywords (`Gemini`, `ChatGPT`, `Claude`). Unknown apps fall back to `ActionDelimitedExtractor`, which then falls back to `GenericExtractor`.

### Input observation

`input_observer.py` handles:
- Focused element capture
- Placeholder detection across `AXPlaceholderValue`, `AXNumberOfCharacters`, cursor-at-zero heuristics, and app-specific prefixes
- Conversation extraction caching
- Browser URL lookup
- Subtree XML extraction via `get_subtree_context()`

`AXWebArea` and `AXGroup` can be editable or read-only; `app.py` explicitly filters ambiguous empty cases before triggering.

### Text injection strategies

`text_injector.py` tries, in order:
1. AX value setting plus cursor restoration
2. CDP injection for Chromium apps (`Input.insertText`, then JS fallback)
3. Clipboard paste
4. Simulated keystrokes

When placeholder text is baked into the field, AX injection is skipped so the app's own JS/input handlers clear the placeholder correctly.

### Persistence and feedback

- `context_store.py`: SQLite `suggestion_feedback` plus context string assembly helpers
- `context_trail.py`: in-memory rolling snapshots of recently visited apps/windows
- `latency_tracker.py`: per-trigger latency breakdowns persisted in `latency_metrics`
- `memory.py`: optional mem0 + FAISS long-term memory store
- `trigger_dump.py`: optional per-trigger JSON dumps of AX/context/request state

Feedback is used to compute accept-rate stats, adjust temperature, and feed recently dismissed patterns back into prompts as negative examples.

## Key patterns

- **Coordinate systems**: AX uses top-left origin, AppKit uses bottom-left. Overlay positioning must convert between them.
- **Caret position**: `app.py` prefers `AXBoundsForRange` on `AXSelectedTextRange` over element bounds when possible.
- **Dynamic polling**: observer poll interval decays from 0.5s toward 4.0s when idle and resets on activity.
- **SQLite thread safety**: both context and latency stores use per-thread `sqlite3` connections with WAL mode.
- **TUI vs shell**: Claude Code-like panes inside terminals should go through `parse_tui_buffer()` so suggestions act like chat/reply rather than command completion.
- **Subtree-first context**: current prompting relies on subtree XML near the focused element, not flat visible-text scraping.
- **Cross-app snapshots**: `ContextTrail.record()` now stores subtree-derived summaries keyed by app/window switches rather than raw `VisibleContent`.
- **Python requirement**: Python `>=3.11` per `pyproject.toml`.

## Testing

High-signal tests for behavior changes:
- `tests/test_e2e_pipeline.py`: end-to-end AX fixture coverage across extraction, context assembly, and mode detection
- `tests/test_subtree_context.py`: subtree XML extraction, pruning, and serialization behavior
- `tests/test_streaming.py`: streaming JSON parsing, fallback behavior, and error handling
- `tests/test_conversation_extractors.py`, `tests/test_extractor_regression.py`, `tests/test_whatsapp_extractor.py`: extractor regressions
- `tests/test_text_injector.py`, `tests/test_cdp_injector.py`: injection behavior
- `tests/test_context_store.py`, `tests/test_memory.py`, `tests/test_context_trail.py`: context assembly, trail formatting, and retrieval
- `tests/test_shell_parser.py`: terminal and Claude Code TUI parsing
- `tests/test_latency_tracker.py`: timing persistence and reporting

Legacy note:
- `tests/test_embeddings.py` still targets the older persisted-context and semantic-retrieval APIs. If you continue the subtree-only refactor, update or remove that coverage in the same change so the suite stays internally consistent.

## Debugging

Use `--log-file` for full DEBUG logs. Important log markers:
- `[CTX]` for context extraction and placeholder handling
- `--- TRIGGER ---` for hotkey/manual trigger flow
- `--- REGENERATE ---` for regenerate flow
- `--- LLM REQUEST ... ---` and streaming logs in `suggestion_engine.py`
- `[LATENCY]` for per-trigger timing summaries

Useful tools:
- `dump_ax_tree.py`: inspect the focused app/window AX tree
- `dump_pipeline.py`: inspect assembled context for both modes
- `python -m autocompleter --dump-dir dumps`: capture full trigger snapshots for later fixture-based debugging
