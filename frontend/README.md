# Autocompleter Landing

Short Astro landing page for the Autocompleter macOS beta. Submits applications
to the FastAPI backend at `/v1/beta/applications` and shows the install
credentials immediately on success.

## Local development

```bash
cd frontend
cp .env.example .env   # edit PUBLIC_BACKEND_BASE_URL
npm install
npm run dev
```

For local backend work, set `PUBLIC_BACKEND_BASE_URL=http://127.0.0.1:8000`.
For deployed Pages builds, set `PUBLIC_BACKEND_BASE_URL` in Cloudflare Pages to
the public backend origin without `/v1`, for example
`https://autocompleter-beta-backend.onrender.com`.

The form posts to `${PUBLIC_BACKEND_BASE_URL}/v1/beta/applications`. Run the
backend locally from the repo root with:

```bash
uvicorn backend.app:app --reload
```

## Build

```bash
npm run build
npm run preview
```

## Tests

Unit + DOM tests run under Vitest with happy-dom:

```bash
npm run test
```

## Structure

- `src/pages/index.astro` — single-page flow (hero → visual → proof → how → beta notice → waitlist → footer)
- `src/components/` — one file per section
- `src/lib/validation.ts` — pure validation helpers (mirrors backend rules)
- `src/lib/api.ts` — `submitBetaApplication` fetch wrapper
- `src/lib/form.ts` — DOM wiring invoked from `WaitlistForm.astro`
- `tests/` — Vitest suites for validation, api, and the form flow
- The backend must allow CORS from `autocompleter.dev`, local Astro dev, and Pages previews for browser submissions to succeed
