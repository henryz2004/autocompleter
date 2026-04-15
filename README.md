# autocompleter

`autocompleter` is a macOS system-wide contextual autocomplete tool. It watches the focused input via the Accessibility API, collects nearby UI context, asks an LLM for suggestions, and renders them in a floating overlay that you can accept directly into the active app.

This project is early and macOS-specific. It is designed for local use on your machine, with your own model provider credentials.

## What It Does

- Works across native macOS apps, browsers, chat apps, and terminals.
- Uses Accessibility data and subtree context near the caret instead of flat screen scraping.
- Switches between continuation, reply, shell, and TUI-aware prompting.
- Streams multiple suggestions into an overlay.
- Injects accepted text with AX, CDP, clipboard paste, or simulated keystrokes.
- Optionally keeps short feedback history and long-term memory under `~/.autocompleter`.

## Requirements

- macOS
- Python 3.11+
- Accessibility permission for the host app you launch from
- At least one supported LLM provider API key

## Install

```bash
git clone <your-repo-url>
cd autocompleter
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e '.[dev]'
cp .env.example .env
```

Then edit `.env` with your provider settings and API key.

## Friend Beta

For the current friend beta, the intended path is:

1. Copy `.env.example` to `.env`
2. Paste in the beta proxy URL, install id, and install key
3. Run `python -m autocompleter`

Telemetry is enabled by default for the beta, but you can opt out by setting:

```bash
AUTOCOMPLETER_TELEMETRY_ENABLED=0
```

If you prefer to use your own provider instead of the beta proxy, set:

```bash
AUTOCOMPLETER_PROXY_ENABLED=0
```

Then fill in the relevant BYO provider credentials below it in `.env`.

See [docs/friend-beta.md](docs/friend-beta.md) for the beta-specific setup flow.
If you want to run the included beta backend from this repo, see [docs/backend-beta.md](docs/backend-beta.md).
For the Supabase dev/prod migration workflow, see [docs/supabase-environments.md](docs/supabase-environments.md).

## Configuration

Configuration is loaded from environment variables and an optional local `.env` file in the repo root.

The current defaults are tuned for OpenAI-compatible endpoints:

- Primary provider: Cerebras-compatible OpenAI endpoint
- Fallback provider: Groq-compatible OpenAI endpoint
- Anthropic is also supported

Start with `.env.example`, then set at least the variables needed for the provider you want to use.

Common options:

- `AUTOCOMPLETER_LLM_PROVIDER`
- `AUTOCOMPLETER_LLM_BASE_URL`
- `AUTOCOMPLETER_LLM_MODEL`
- `AUTOCOMPLETER_FALLBACK_PROVIDER`
- `AUTOCOMPLETER_FALLBACK_BASE_URL`
- `AUTOCOMPLETER_FALLBACK_MODEL`
- `AUTOCOMPLETER_HOTKEY`
- `AUTOCOMPLETER_REGENERATE_HOTKEY`
- `AUTOCOMPLETER_AUTO_TRIGGER`
- `AUTOCOMPLETER_MEMORY`

Provider keys are resolved from standard env vars such as:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `CEREBRAS_API_KEY`
- `GROQ_API_KEY`
- `SAMBANOVA_API_KEY`
- `TOGETHER_API_KEY`
- `FIREWORKS_API_KEY`
- `DEEPINFRA_API_KEY`

## Run

```bash
source venv/bin/activate
python -m autocompleter
```

Useful variants:

```bash
python -m autocompleter --log-file /tmp/autocompleter.log --log-level DEBUG
python -m autocompleter --dump-dir dumps
python -m autocompleter --stats
python -m autocompleter --stats 100
python -m autocompleter --consolidate-memory
```

## Hotkeys

- Trigger suggestions: `ctrl+space`
- Regenerate suggestions: `ctrl+r`
- Toggle auto-trigger: `shift+ctrl+space`
- Accept: `tab` or `return`
- Partial accept: `shift+tab`
- Navigate: `up` and `down`
- Dismiss: `escape`
- Accept suggestion 1-3 directly: `1`, `2`, `3`

## Development

Run the test suite:

```bash
./venv/bin/python -m pytest tests/ -v
```

Targeted examples:

```bash
./venv/bin/python -m pytest tests/test_suggestion_engine.py -v
./venv/bin/python -m pytest tests/test_e2e_pipeline.py -v
./venv/bin/python -m pytest tests/test_streaming.py -v
```

Debug helpers:

```bash
python dump_ax_tree.py
python dump_pipeline.py
```

## Privacy And Safety Notes

- The app reads focused-input context and nearby Accessibility tree data from the active app.
- Prompt context may include message drafts, visible conversation context, window titles, URLs, and terminal content.
- Generated requests are sent to the configured LLM provider over the network.
- Local data is stored under `~/.autocompleter`.
- Do not use this on apps or data you are not comfortable sharing with your configured provider.
- Review `--dump-dir` output before sharing it, because dumps can contain sensitive context.

See also:

- [PRIVACY.md](PRIVACY.md)
- [docs/data-flows.md](docs/data-flows.md)
- [docs/friend-beta.md](docs/friend-beta.md)
- [docs/backend-beta.md](docs/backend-beta.md)
- [docs/supabase-environments.md](docs/supabase-environments.md)

## Project Layout

- [autocompleter/app.py](autocompleter/app.py)
- [autocompleter/context_store.py](autocompleter/context_store.py)
- [autocompleter/input_observer.py](autocompleter/input_observer.py)
- [autocompleter/suggestion_engine.py](autocompleter/suggestion_engine.py)
- [autocompleter/text_injector.py](autocompleter/text_injector.py)
- [tests](tests)

## Open-Source Readiness

This branch now includes contributor, security, and conduct docs, plus a sample environment file. One important release task is still intentionally left for a human decision: choose and add the final open-source license before publishing publicly.

See:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [PRIVACY.md](PRIVACY.md)
