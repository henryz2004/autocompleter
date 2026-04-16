SHELL := /bin/sh

SUPABASE ?= supabase
UV ?= uv
ifneq ("$(wildcard ./venv/bin/python)","")
PYTHON := ./venv/bin/python
else ifneq ("$(wildcard ./.venv/bin/python)","")
PYTHON := ./.venv/bin/python
else ifneq ("$(wildcard ./venv/bin/uvicorn)","")
PYTHON := ./venv/bin/python
else ifneq ("$(wildcard ./.venv/bin/uvicorn)","")
PYTHON := ./.venv/bin/python
else
PYTHON ?= python3
endif
BACKEND_ENV_FILE ?= .env.backend.dev
INSTALL_LABEL ?= local-smoke
INSTALL_NOTES ?= created-via-make

ifneq ("$(wildcard $(BACKEND_ENV_FILE))","")
include $(BACKEND_ENV_FILE)
export
endif

.PHONY: help
help:
	@printf '%s\n' \
		'make friend-beta-bootstrap   Use uv to install Python 3.11, create ./venv, and install the app' \
		'make friend-beta-run         Bootstrap the repo-local beta app and launch it' \
		'make supabase-link-dev        Link the repo to the dev Supabase project' \
		'make supabase-db-push-dev     Push repo migrations to the dev Supabase project' \
		'make supabase-link-prod       Link the repo to the prod Supabase project' \
		'make supabase-db-push-prod    Push repo migrations to the prod Supabase project' \
		'make backend-run-dev          Run the backend with .env.backend.dev' \
		'make backend-run-prod         Run the backend with .env.backend.prod' \
		'make backend-health-dev       Check the local backend /health endpoint' \
		'make backend-health-prod      Check the local backend /health endpoint using prod env values' \
		'make backend-mint-install-dev Mint a beta install key against the local backend' \
		'make backend-mint-install-prod Mint a beta install key against the local backend using prod env values'

.PHONY: check-uv
check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv is required for friend-beta bootstrap."; \
		echo "Install it with:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}

.PHONY: friend-beta-bootstrap
friend-beta-bootstrap: check-uv
	@$(UV) python install 3.11
	@$(UV) venv --python 3.11 --seed --allow-existing venv
	@./venv/bin/python -c "import importlib.util, sys; required=('pydantic','openai','anthropic','instructor'); missing=[name for name in required if importlib.util.find_spec(name) is None]; sys.exit(1 if missing else 0)" >/dev/null 2>&1 || { \
		echo "Installing autocompleter into ./venv"; \
		./venv/bin/python -m pip install -r requirements.txt; \
		./venv/bin/python -m pip install -e .; \
	}
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created ./.env from ./.env.example"; \
		echo "Fill in your beta proxy URL, install id, and install key, then rerun make friend-beta-run."; \
	fi

.PHONY: friend-beta-run
friend-beta-run: friend-beta-bootstrap
	@./venv/bin/python scripts/run_friend_beta.py

.PHONY: check-supabase-cli
check-supabase-cli:
	@command -v $(word 1,$(SUPABASE)) >/dev/null 2>&1 || { echo "Supabase CLI not found. Install it or run with SUPABASE='npx supabase'"; exit 1; }

.PHONY: check-backend-env
check-backend-env:
	@test -f "$(BACKEND_ENV_FILE)" || { echo "Missing $(BACKEND_ENV_FILE). Copy one of the .example files first."; exit 1; }

.PHONY: check-project-ref
check-project-ref: check-backend-env
	@test -n "$(SUPABASE_PROJECT_REF)" || { echo "Set SUPABASE_PROJECT_REF in $(BACKEND_ENV_FILE)"; exit 1; }

.PHONY: check-backend-config
check-backend-config: check-backend-env
	@test -n "$(AUTOCOMPLETER_BACKEND_ADMIN_SECRET)" || { echo "Set AUTOCOMPLETER_BACKEND_ADMIN_SECRET in $(BACKEND_ENV_FILE)"; exit 1; }
	@test -n "$(AUTOCOMPLETER_SUPABASE_URL)" || { echo "Set AUTOCOMPLETER_SUPABASE_URL in $(BACKEND_ENV_FILE)"; exit 1; }
	@test -n "$(AUTOCOMPLETER_SUPABASE_SECRET_KEY)" || { echo "Set AUTOCOMPLETER_SUPABASE_SECRET_KEY in $(BACKEND_ENV_FILE)"; exit 1; }
	@test -n "$(AUTOCOMPLETER_PROXY_PRIMARY_BASE_URL)" || { echo "Set AUTOCOMPLETER_PROXY_PRIMARY_BASE_URL in $(BACKEND_ENV_FILE)"; exit 1; }
	@test -n "$(AUTOCOMPLETER_PROXY_PRIMARY_API_KEY)" || { echo "Set AUTOCOMPLETER_PROXY_PRIMARY_API_KEY in $(BACKEND_ENV_FILE)"; exit 1; }
	@test -n "$(AUTOCOMPLETER_PROXY_PRIMARY_DEFAULT_MODEL)" || { echo "Set AUTOCOMPLETER_PROXY_PRIMARY_DEFAULT_MODEL in $(BACKEND_ENV_FILE)"; exit 1; }

.PHONY: supabase-link
supabase-link: check-supabase-cli check-project-ref
	@if [ -n "$(SUPABASE_DB_PASSWORD)" ]; then \
		$(SUPABASE) link --project-ref "$(SUPABASE_PROJECT_REF)" -p "$(SUPABASE_DB_PASSWORD)"; \
	else \
		$(SUPABASE) link --project-ref "$(SUPABASE_PROJECT_REF)"; \
	fi

.PHONY: supabase-db-push
supabase-db-push: supabase-link
	$(SUPABASE) db push

.PHONY: supabase-link-dev
supabase-link-dev:
	@$(MAKE) supabase-link BACKEND_ENV_FILE=.env.backend.dev

.PHONY: supabase-db-push-dev
supabase-db-push-dev:
	@$(MAKE) supabase-db-push BACKEND_ENV_FILE=.env.backend.dev

.PHONY: supabase-link-prod
supabase-link-prod:
	@$(MAKE) supabase-link BACKEND_ENV_FILE=.env.backend.prod

.PHONY: supabase-db-push-prod
supabase-db-push-prod:
	@$(MAKE) supabase-db-push BACKEND_ENV_FILE=.env.backend.prod

.PHONY: backend-run
backend-run: check-backend-config
	@set -a; . "$(BACKEND_ENV_FILE)"; set +a; \
	PYTHONPATH=. $(PYTHON) -m uvicorn backend.app:app --host "$${BACKEND_HOST:-127.0.0.1}" --port "$${BACKEND_PORT:-8000}"

.PHONY: backend-run-dev
backend-run-dev:
	@$(MAKE) backend-run BACKEND_ENV_FILE=.env.backend.dev

.PHONY: backend-run-prod
backend-run-prod:
	@$(MAKE) backend-run BACKEND_ENV_FILE=.env.backend.prod

.PHONY: backend-health
backend-health: check-backend-env
	@curl --fail --silent --show-error "$${BACKEND_BASE_URL:-http://127.0.0.1:8000}/health"

.PHONY: backend-health-dev
backend-health-dev:
	@$(MAKE) backend-health BACKEND_ENV_FILE=.env.backend.dev

.PHONY: backend-health-prod
backend-health-prod:
	@$(MAKE) backend-health BACKEND_ENV_FILE=.env.backend.prod

.PHONY: backend-mint-install
backend-mint-install: check-backend-config
	@curl --fail --silent --show-error \
		-X POST "$${BACKEND_BASE_URL:-http://127.0.0.1:8000}/admin/install-keys" \
		-H "Content-Type: application/json" \
		-H "X-Admin-Secret: $(AUTOCOMPLETER_BACKEND_ADMIN_SECRET)" \
		-d '{"label":"$(INSTALL_LABEL)","notes":"$(INSTALL_NOTES)"}'

.PHONY: backend-mint-install-dev
backend-mint-install-dev:
	@$(MAKE) backend-mint-install BACKEND_ENV_FILE=.env.backend.dev

.PHONY: backend-mint-install-prod
backend-mint-install-prod:
	@$(MAKE) backend-mint-install BACKEND_ENV_FILE=.env.backend.prod
