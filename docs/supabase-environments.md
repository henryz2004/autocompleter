# Supabase Environments

This repo now treats Supabase schema as code.

The source of truth for backend tables is:

- [supabase/migrations/20260415000100_create_beta_backend_tables.sql](../supabase/migrations/20260415000100_create_beta_backend_tables.sql)

## Why Migrations Instead Of Dashboard Edits

The goal is to make schema changes reviewable, reproducible, and promotable across environments.

Instead of:

- opening the Supabase dashboard
- creating tables by hand
- trying to remember what changed between dev and prod

use versioned migration files in `supabase/migrations/`.

That gives you a simple workflow:

1. Add a migration in the repo.
2. Apply it to the dev Supabase project.
3. Test the backend against dev.
4. Apply the same migration to prod.

## Recommended Environment Layout

Use separate Supabase projects:

- `autocompleter-dev`
- `autocompleter-prod`

Keep separate backend env files outside git or in ignored files such as:

- `.env.backend.dev`
- `.env.backend.prod`

Start from:

- `.env.backend.dev.example`
- `.env.backend.prod.example`

These should contain different values for:

- `AUTOCOMPLETER_SUPABASE_URL`
- `AUTOCOMPLETER_SUPABASE_SERVICE_ROLE_KEY`
- upstream provider keys
- `AUTOCOMPLETER_BACKEND_ADMIN_SECRET`
- `SUPABASE_PROJECT_REF`
- optionally `SUPABASE_DB_PASSWORD`

## Standard CLI Flow

Install the Supabase CLI, then initialize the project once from the repo root:

```bash
cp .env.backend.dev.example .env.backend.dev
cp .env.backend.prod.example .env.backend.prod
```

Fill in the real values for each environment, then link the repo to your dev project:

```bash
make supabase-link-dev
```

Push repo migrations to dev:

```bash
make supabase-db-push-dev
```

When you are ready to promote the same schema to prod, relink to prod and push again:

```bash
make supabase-link-prod
make supabase-db-push-prod
```

These targets read from `.env.backend.dev` or `.env.backend.prod` so you do not have to remember which project ref is active.

## Safe Promotion Pattern

Recommended order:

1. Write a new migration in `supabase/migrations/`
2. Run `make supabase-db-push-dev`
3. Run backend tests locally
4. Run the backend against dev with `make backend-run-dev`
5. Smoke test install-key minting, proxy requests, and telemetry ingest
6. Run `make supabase-db-push-prod`

## First Dev Smoke Test

After the dev migration is pushed:

1. Start the backend:

```bash
make backend-run-dev
```

2. In another shell, verify health:

```bash
make backend-health-dev
```

3. Mint a test install key:

```bash
make backend-mint-install-dev INSTALL_LABEL=henry-local
```

4. Take the returned `install_id` and `install_key` and drop them into the desktop app `.env`.

5. Point the app at the local backend:

```bash
AUTOCOMPLETER_PROXY_BASE_URL=http://127.0.0.1:8000/v1
AUTOCOMPLETER_TELEMETRY_URL=http://127.0.0.1:8000/v1/telemetry/events
```

## Current Schema Boundary

Right now the backend store uses the Supabase REST API, so these beta tables live in `public` to match that implementation.

That is a practical beta tradeoff, not the long-term ideal.

If you later want truly backend-only tables in a `private` schema, the clean next step is:

- move the backend off the Supabase REST API for these tables
- use a direct Postgres connection from the backend
- keep `private.*` tables out of the client-facing Data API surface

Until then, be careful not to treat `public` as a junk drawer. Only keep tables there that this backend actually needs.

## What Belongs In Git

Commit:

- SQL migrations
- backend schema docs
- environment examples

Do not commit:

- service-role keys
- admin secrets
- real install keys
- production data
