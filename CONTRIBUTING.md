# Contributing

Thanks for helping improve `autocompleter`.

## Development Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
cp .env.example .env
```

The project is macOS-specific, and many features require Accessibility permissions and a real focused input to exercise manually.

## Before Opening A PR

- Run `./venv/bin/python -m pytest tests/ -v`
- Add or update tests when behavior changes
- Keep changes scoped and explain user-visible behavior clearly
- Avoid committing `.env`, dumps, or local diagnostic artifacts

## Testing Notes

High-signal suites for most changes:

- `tests/test_e2e_pipeline.py`
- `tests/test_streaming.py`
- `tests/test_context_store.py`
- `tests/test_subtree_context.py`
- `tests/test_suggestion_engine.py`

If you change shell parsing, extractors, or injection behavior, update the related targeted tests in the same PR.

## Pull Request Guidance

- Describe the user-facing problem first
- Call out privacy or context-capture implications explicitly
- Include screenshots or logs when changing overlay or AX behavior
- Mention any macOS version or app-specific assumptions

## Code Style

- Python 3.11+
- Keep behavior changes covered by tests where practical
- Prefer small, focused helpers over broad rewrites
- Preserve existing architecture boundaries between observation, context assembly, suggestion generation, overlay rendering, and injection

## Release Notes For Contributors

This repository is being prepared for a public launch. If you add new setup requirements, env vars, provider integrations, or local persistence, update `README.md` and `.env.example` in the same change.

