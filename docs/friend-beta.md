# Friend Beta Setup

This beta is still distributed as a repository plus manual installation steps. It is not yet a packaged macOS app.

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

## Required Beta Config

Edit `.env` and set these fields:

```bash
AUTOCOMPLETER_BETA_MODE=1
AUTOCOMPLETER_PROXY_ENABLED=1
AUTOCOMPLETER_PROXY_BASE_URL=<shared beta proxy url>
AUTOCOMPLETER_PROXY_API_KEY=<install key>
AUTOCOMPLETER_TELEMETRY_ENABLED=1
AUTOCOMPLETER_TELEMETRY_URL=<proxy base url>/telemetry/events
AUTOCOMPLETER_INSTALL_ID=<install id>
```

Then run:

```bash
source venv/bin/activate
python -m autocompleter
```

Grant Accessibility permission to the app you launch it from, usually Terminal or iTerm.

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
