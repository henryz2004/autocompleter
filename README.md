# autocompleter

`autocompleter` is a macOS system-wide contextual autocomplete tool. It watches the focused input via the Accessibility API, collects nearby UI context, asks an LLM for suggestions, and renders them in a floating overlay that you can accept directly into the active app.

This project is early and macOS-specific. It is designed for local use on your machine, with your own model provider credentials.

For the beta backend and Supabase integration, prefer the modern Supabase API key types:

- `sb_publishable_...` for client-side use
- `sb_secret_...` for backend/server-side use

## What It Does

- Works across native macOS apps, browsers, chat apps, and terminals.
- Uses Accessibility data and subtree context near the caret instead of flat screen scraping.
- Switches between continuation, reply, shell, and TUI-aware prompting.
- Streams multiple suggestions into an overlay.
- Injects accepted text with AX, CDP, clipboard paste, or simulated keystrokes.
- Optionally keeps short feedback history and long-term memory under `~/.autocompleter`.

## Requirements

- macOS
- `uv` for the recommended bootstrap flow
- Accessibility permission for the host app you launch from
- At least one supported LLM provider API key

## Install

Recommended bootstrap:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone https://github.com/henryz2004/autocompleter.git
cd autocompleter
make friend-beta-bootstrap
```

That will use `uv` to install Python 3.11 if needed, create `./venv`, install the app, and create `./.env` from `./.env.example` if it does not already exist.

Then edit `.env` with either:
- beta proxy credentials: `AUTOCOMPLETER_INSTALL_ID` and `AUTOCOMPLETER_PROXY_API_KEY`
- or your own provider settings if you are explicitly turning proxy mode off

Manual alternative:

```bash
git clone https://github.com/henryz2004/autocompleter.git
cd autocompleter
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

## Friend Beta

For the current friend beta, the intended path is:

1. Install `uv` with `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Restart your terminal or run `source "$HOME/.local/bin/env"`
3. `git clone https://github.com/henryz2004/autocompleter.git`
4. `cd autocompleter`
5. Run `make friend-beta-run`
6. Fill in `.env` with the beta `AUTOCOMPLETER_INSTALL_ID` and `AUTOCOMPLETER_PROXY_API_KEY` values if the launcher created it
7. Grant Accessibility access to your terminal app in macOS
8. Run `make friend-beta-run` again

Telemetry is enabled by default for the beta, but you can opt out by setting:

```bash
AUTOCOMPLETER_TELEMETRY_ENABLED=0
```

If you prefer to use your own provider instead of the beta proxy, set:

```bash
AUTOCOMPLETER_PROXY_ENABLED=0
```

Then fill in the relevant BYO provider credentials below it in `.env`.
When `AUTOCOMPLETER_PROXY_ENABLED=1`, the app routes generation through the hosted proxy only. Local `AUTOCOMPLETER_LLM_*` and fallback model/provider settings do not control hosted beta routing.

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
In beta proxy mode, the important settings are `AUTOCOMPLETER_PROXY_*`, `AUTOCOMPLETER_TELEMETRY_*`, and `AUTOCOMPLETER_INSTALL_ID`; the local `AUTOCOMPLETER_LLM_*` settings only apply in BYO mode.

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
make friend-beta-run
```

Useful variants:

```bash
./venv/bin/python -m autocompleter --log-file /tmp/autocompleter.log --log-level DEBUG
./venv/bin/python -m autocompleter --dump-dir dumps
./venv/bin/python -m autocompleter --stats
./venv/bin/python -m autocompleter --stats 100
./venv/bin/python -m autocompleter --consolidate-memory
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

If you want the backend/test tooling too, install the dev extras:

```bash
./venv/bin/python -m pip install -e '.[dev]'
```

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
python dump_ax_tree_json.py -d tests/fixtures/ax_trees --notes "what was on screen"
python dump_pipeline.py
python analyze_fixture.py tests/fixtures/ax_trees/claude-current-chat.json
python replay_fixture.py tests/fixtures/ax_trees/claude-current-chat.json
./venv/bin/python -m pytest tests/test_fixture_workflow.py -v
```

For the full fixture capture, analysis, and benchmark workflow, see [docs/fixture-workflow.md](docs/fixture-workflow.md).

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
- [docs/fixture-workflow.md](docs/fixture-workflow.md)
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
