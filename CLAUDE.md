# Autocompleter

macOS system-wide contextual autocomplete tool. Observes text input via the Accessibility API, collects context from the active window, and generates LLM-powered suggestions shown in a floating overlay.

## Quick start

```bash
# Activate the virtualenv
source venv/bin/activate

# Set your API key (or add to .env)
export ANTHROPIC_API_KEY=sk-...

# Run
python -m autocompleter

# Run with full debug log to file
python -m autocompleter --log-file /tmp/autocompleter.log --log-level INFO
```

## Testing

```bash
./venv/bin/python -m pytest tests/ -v
```

## Architecture

| Module | Role |
|--------|------|
| `app.py` | Main orchestrator — wires all components, runs the AppKit event loop, dispatches LLM calls to worker threads |
| `input_observer.py` | Reads focused text fields and visible window content via AX API; 4-strategy placeholder detection; conversation turn extraction |
| `suggestion_engine.py` | Calls Anthropic/OpenAI APIs to generate completions; mode-aware prompts (continuation vs reply) |
| `overlay.py` | Floating NSWindow overlay — multi-monitor clamping, dynamic text measurement, caret-anchored positioning |
| `text_injector.py` | Injects accepted text via AX value setting, clipboard paste, or keystrokes; skips AX for placeholder fields |
| `hotkey.py` | Global hotkey listener using CGEvent taps with auto-reenable on timeout |
| `context_store.py` | SQLite-backed store (WAL mode, per-thread connections) with mode-aware context assembly |
| `config.py` | Dataclass config with .env and env var loading; per-mode temperature/token settings |
| `ax_utils.py` | Shared Accessibility API helpers (get/set attributes, position, size, PID) |

## Key patterns

- **Thread safety**: Hotkey callbacks run on the CGEvent tap thread and must return within ~1s (macOS disables the tap otherwise). UI work is dispatched to the main thread via `_run_on_main()` queue. LLM calls run on daemon worker threads. The generation ID counter discards stale results from superseded triggers.
- **Coordinate systems**: AX API uses top-left origin (Y down). AppKit uses bottom-left origin (Y up). Conversion: `appkit_y = primary_screen_height - ax_y`. Handled in `overlay.py`.
- **Python 3.9 compat**: Uses `from __future__ import annotations` and `typing.Optional` / `typing.Union` where needed.
- **Mode-aware pipeline**: `detect_mode()` classifies input as CONTINUATION (user has draft text, predict next words) or REPLY (input empty/short, suggest a full response). Each mode has its own system prompt, temperature, max tokens, and context assembly strategy.
- **Placeholder detection**: 4 strategies in order — (1) AXPlaceholderValue match, (2) AXNumberOfCharacters==0, (3) cursor at position 0 + short value, (4) app-specific known prefixes (`_APP_PLACEHOLDER_PREFIXES`). When detected, `placeholder_detected=True` routes injection through clipboard paste instead of AXValue setting (which bypasses web app JS handlers).
- **Web app depth handling**: For PWAs and Electron apps, `_collect_text` starts from the AXWebArea instead of the window root, skipping Chromium's deep scaffolding layers (~10 levels of wrapper AXGroups).

## Debugging

Use `--log-file` for full DEBUG output. Key log prefixes:
- `[CTX]` — context extraction and assembly (visible text, conversation turns, placeholder detection)
- `--- TRIGGER ---` — hotkey fired, shows focused element info
- `--- SUGGESTIONS ---` — LLM results with generation ID and timing
- `--- CONTEXT ---` — full assembled context sent to the LLM

The `dump_ax_tree.py` script dumps the full AX tree of the focused app to a file for debugging content extraction issues.
