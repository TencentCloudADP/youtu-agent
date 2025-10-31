SHELL := /bin/bash
.SHELLFLAGS := -e -c

.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format: 
	uv run ruff format
	uv run ruff check --fix

.PHONY: format-check
format-check:
	uv run ruff format --check

.PHONY: lint
lint: 
	uv run ruff check

.PHONY: build-docs
build-docs:
	uv run mkdocs build

.PHONY: serve-docs
serve-docs:
	uv run mkdocs serve

.PHONY: deploy-docs
deploy-docs:
	uv run mkdocs gh-deploy --force --verbose

.PHONY: build-ui
build-ui:
	uv pip install build
	npm --version || echo "npm not found, please install npm"
	cd frontend/webui && npm install && bash build.sh
	uv pip install --force-reinstall frontend/webui/build/utu_agent_ui-0.2.0-py3-none-any.whl

.PHONY: demo
demo: build-ui
	uv run python -m demo.demo
