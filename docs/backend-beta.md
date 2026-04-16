# Beta Backend

This repo now includes a small FastAPI backend for the friend beta. It provides:

- an OpenAI-compatible proxy at `/v1/chat/completions`
- a telemetry ingest endpoint at `/v1/telemetry/events`
- admin endpoints for minting and revoking per-install keys

## Install

From the repo root:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e '.[dev]'
```

Run the backend with:

```bash
source venv/bin/activate
autocompleter-beta-backend
```

Or with Uvicorn directly:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Deploy To Render

This repo now includes Render deployment scaffolding for the backend:

- [render.yaml](../render.yaml)
- [backend/Dockerfile](../backend/Dockerfile)
- [requirements-backend.txt](../requirements-backend.txt)

The deployment uses a backend-only Docker image on Render instead of the root `requirements.txt`. This matters because the desktop app depends on macOS-only `pyobjc` packages, which are not valid on Render's Linux runtime.

Recommended Render flow:

1. Push this repo to GitHub.
2. In Render, create a new `Blueprint` and point it at this repository.
3. Render will detect [render.yaml](../render.yaml) and create a Docker-based web service from [backend/Dockerfile](../backend/Dockerfile).
4. In the Blueprint setup flow, provide values for the required secret environment variables.
5. After the first deploy, open `https://<your-service>.onrender.com/health` and confirm it returns `{"ok": true, ...}`.

Render-specific notes:

- Render Blueprints use `render.yaml` in the repo root by default.
- The configured `healthCheckPath` is `/health`, which Render recommends for zero-downtime deploys.
- Render injects a `PORT` variable for web services. The backend Docker command already binds Uvicorn to `${PORT}`.
- The service is intended to be deployed as a single web backend. Do not expose the admin secret to clients.

If you still prefer Railway later, the repo also contains [railway.json](../railway.json), but Render is now the recommended hosted path for the friend beta.

If you prefer repo-local helper commands, use:

```bash
cp .env.backend.dev.example .env.backend.dev
make backend-run-dev
```

## Backend Env Vars

Required:

```bash
AUTOCOMPLETER_BACKEND_ADMIN_SECRET=<admin secret>
AUTOCOMPLETER_SUPABASE_URL=https://<project>.supabase.co
AUTOCOMPLETER_SUPABASE_SECRET_KEY=<secret key>

AUTOCOMPLETER_PROXY_PRIMARY_BASE_URL=https://<primary-openai-compatible-upstream>/v1
AUTOCOMPLETER_PROXY_PRIMARY_API_KEY=<primary upstream key>
AUTOCOMPLETER_PROXY_PRIMARY_DEFAULT_MODEL=<default model name>
```

The backend uses the modern Supabase server-side secret key format (`sb_secret_...`). A legacy `AUTOCOMPLETER_SUPABASE_SERVICE_ROLE_KEY` still works as a fallback, but the preferred variable is `AUTOCOMPLETER_SUPABASE_SECRET_KEY`.

You do not need a publishable key for the current backend flow because the desktop app talks to your FastAPI backend, not directly to Supabase. If you later add direct client-side Supabase access, that is when `sb_publishable_...` becomes relevant.

Optional fallback:

```bash
AUTOCOMPLETER_PROXY_FALLBACK_BASE_URL=https://<fallback-openai-compatible-upstream>/v1
AUTOCOMPLETER_PROXY_FALLBACK_API_KEY=<fallback upstream key>
AUTOCOMPLETER_PROXY_FALLBACK_DEFAULT_MODEL=<fallback model name>
```

Optional model mapping and routing controls:

```bash
AUTOCOMPLETER_PROXY_PRIMARY_MODEL_MAP={"beta-model":"provider/model"}
AUTOCOMPLETER_PROXY_FALLBACK_MODEL_MAP={"beta-model":"provider/model"}
AUTOCOMPLETER_PROXY_PRIMARY_ALLOWED_MODELS=beta-model,another-model
AUTOCOMPLETER_PROXY_FALLBACK_ALLOWED_MODELS=beta-model,another-model
AUTOCOMPLETER_BACKEND_REQUEST_TIMEOUT_S=15
AUTOCOMPLETER_BACKEND_STREAM_TIMEOUT_S=60
AUTOCOMPLETER_BACKEND_ALLOW_UPSTREAM_OVERRIDE_HEADERS=0
```

## Supabase Schema

The schema now lives in versioned Supabase migrations instead of a one-off SQL snapshot.

Start with:

- [supabase/migrations/20260415000100_create_beta_backend_tables.sql](../supabase/migrations/20260415000100_create_beta_backend_tables.sql)

The backend expects these tables:

- `beta_installs`
- `beta_invocations`
- `beta_proxy_requests`
- `beta_proxy_attempts`
- `beta_telemetry_events`

The backend stores install/auth data, invocation-level product analytics, metadata-only proxy request rows, per-attempt upstream metadata, and raw telemetry events. It does not persist raw prompt or completion bodies by default.

Current correlation model:

- the desktop client sends one `invocation_id` per autocomplete lifecycle
- the proxy stores one logical `beta_proxy_requests` row per LLM call plus one `beta_proxy_attempts` row per upstream attempt
- telemetry events upsert `beta_invocations` so request performance can be joined directly to accept/dismiss/no-suggestion outcomes

For the recommended dev vs prod workflow, see [docs/supabase-environments.md](supabase-environments.md).

Important current implementation note:

- the backend store currently talks to Supabase via the REST API
- because of that, these beta tables currently live in `public`
- if you later want them in a truly backend-only `private` schema, switch the backend store to a direct Postgres connection first

## Mint Install Keys

Create an install key:

```bash
curl -X POST http://127.0.0.1:8000/admin/install-keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Secret: $AUTOCOMPLETER_BACKEND_ADMIN_SECRET" \
  -d '{"label":"henrys-friend","notes":"beta cohort 1"}'
```

The response returns:

- `install_id`
- `install_key`

Only the hashed key is stored in Supabase. The plaintext key is returned once for manual distribution.

Revoke an install:

```bash
curl -X POST http://127.0.0.1:8000/admin/install-keys/<install_id>/revoke \
  -H "X-Admin-Secret: $AUTOCOMPLETER_BACKEND_ADMIN_SECRET"
```

## Desktop App Config

Point the desktop app at the backend with:

```bash
AUTOCOMPLETER_BETA_MODE=1
AUTOCOMPLETER_PROXY_ENABLED=1
AUTOCOMPLETER_PROXY_BASE_URL=http://127.0.0.1:8000/v1
AUTOCOMPLETER_PROXY_API_KEY=<install_key>
AUTOCOMPLETER_TELEMETRY_ENABLED=1
AUTOCOMPLETER_TELEMETRY_URL=http://127.0.0.1:8000/v1/telemetry/events
AUTOCOMPLETER_INSTALL_ID=<install_id>
```

For Render, replace `http://127.0.0.1:8000` with your public Render domain:

```bash
AUTOCOMPLETER_PROXY_BASE_URL=https://<your-service>.onrender.com/v1
AUTOCOMPLETER_TELEMETRY_URL=https://<your-service>.onrender.com/v1/telemetry/events
```

You can mint `install_id` and `install_key` locally with:

```bash
make backend-mint-install-dev INSTALL_LABEL=friend-name
```

Telemetry auth defaults to the proxy install key, so `AUTOCOMPLETER_TELEMETRY_API_KEY` is usually unnecessary unless you want to override it.

If a friend prefers their own provider, they can set:

```bash
AUTOCOMPLETER_PROXY_ENABLED=0
```

Then use the BYO provider fields instead. Requests sent to a BYO provider are governed by that provider's terms and privacy practices.
