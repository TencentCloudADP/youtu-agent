SHELL := /bin/bash
.SHELLFLAGS := -e -c

.PHONY: sync
sync:
	pip install -r requirements-all.txt

.PHONY: format
format: 
	python -m ruff format
	python -m ruff check --fix

.PHONY: format-check
format-check:
	python -m ruff format --check

.PHONY: lint
lint: 
	python -m ruff check

.PHONY: build-docs
build-docs:
	python -m mkdocs build

.PHONY: serve-docs
serve-docs:
	python -m mkdocs serve

.PHONY: deploy-docs
deploy-docs:
	python -m mkdocs gh-deploy --force --verbose

.PHONY: build-ui
build-ui:
	pip install build
	npm --version || echo "npm not found, please install npm"
	cd utu/ui/frontend && npm install && bash build.sh
	pip install --force-reinstall utu/ui/frontend/build/utu_agent_ui-0.2.0-py3-none-any.whl

.PHONY: demo
demo: build-ui
	python -m demo.demo_universal
