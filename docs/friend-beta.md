# Friend Beta Setup

This beta is still distributed as a repository plus manual installation steps. It is not yet a packaged macOS app.

If you are using the shared beta backend hosted by the Autocompleter team, you do not need to run `uvicorn` locally. The backend is already running for you. Your local setup only needs to launch the desktop app.

## Install

First install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal, or run:

```bash
source "$HOME/.local/bin/env"
```

```bash
git clone https://github.com/henryz2004/autocompleter.git
cd autocompleter
make friend-beta-run
```

On first run, the launcher will:

- use `uv` to install Python 3.11 if needed
- create `./venv`
- install the Python dependencies
- create `./.env` from `./.env.example` if needed
- stop with clear instructions if your beta config is incomplete

If you prefer not to use `make`, this works too:

```bash
make friend-beta-bootstrap
./venv/bin/python scripts/run_friend_beta.py
```

## Required Beta Config

Edit `.env` and set these fields:

```bash
AUTOCOMPLETER_BETA_MODE=1
AUTOCOMPLETER_PROXY_ENABLED=1
AUTOCOMPLETER_PROXY_BASE_URL=https://autocompleter-beta-backend.onrender.com/v1
AUTOCOMPLETER_PROXY_API_KEY=<install key>
AUTOCOMPLETER_TELEMETRY_ENABLED=1
AUTOCOMPLETER_TELEMETRY_URL=https://autocompleter-beta-backend.onrender.com/v1/telemetry/events
AUTOCOMPLETER_INSTALL_ID=<install id>
```

When `AUTOCOMPLETER_PROXY_ENABLED=1`, the hosted backend owns model/provider selection. Do not try to combine the beta proxy with local `AUTOCOMPLETER_LLM_*` or fallback provider settings in the same run.

Then rerun:

```bash
make friend-beta-run
```

## Accessibility Permission

Autocompleter needs macOS Accessibility permission to read focused text fields and inject accepted suggestions.

Grant access to the app you launch it from, usually:

- Terminal
- iTerm
- Warp
- Visual Studio Code terminal

In macOS, go to `System Settings > Privacy & Security > Accessibility`, enable your terminal app, then relaunch that terminal and run:

```bash
make friend-beta-run
```

## Telemetry Opt-Out

Telemetry in this beta is content-free product telemetry. It is separate from prompt routing and can be disabled without disabling the beta proxy.

To opt out:

```bash
AUTOCOMPLETER_TELEMETRY_ENABLED=0
```

## Switching To BYO Provider

If you want to use your own provider instead of the beta proxy:

```bash
AUTOCOMPLETER_PROXY_ENABLED=0
```

Then set the relevant provider values in `.env`, such as:

```bash
AUTOCOMPLETER_LLM_PROVIDER=openai
AUTOCOMPLETER_LLM_BASE_URL=
OPENAI_API_KEY=<your key>
```

Requests sent to your own provider are governed by that provider's terms and privacy practices, not Autocompleter's.

If you want to run the included beta backend yourself, see [docs/backend-beta.md](backend-beta.md).
