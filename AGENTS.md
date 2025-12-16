# Repository Guidelines

## Project Structure & Module Organization
- `utu/` — core Python package: agents (`utu/agents`), tools (`utu/tools`), config loaders (`utu/config`), env, tracing, ui, utils.
- `configs/` — YAML configs for models, toolkits, and agents (e.g., `configs/agents/simple/base.yaml`).
- `tests/` — pytest suite mirroring package layout (e.g., `tests/tools/...`, `tests/agents/...`).
- `scripts/` — runnable utilities (CLI chat, eval, tool generation).
- `examples/` — ready-to-run samples; some require API keys.
- `docs/` + `mkdocs.yml` — documentation site.
- `frontend/`, `docker/`, `demo/`, `workspace/`, `logs/` — optional UI, container, demos, scratch, and runtime outputs.

## Build, Test, and Development Commands
- Sync env: `make sync` (uses `uv` to install dev extras).
- Lint/format: `make format` or `make lint`; check only: `make format-check`.
- Tests: `uv run pytest -q` or `pytest tests/tools/test_search_toolkit.py`.
- Docs: `make build-docs` (static), `make serve-docs` (live), `make deploy-docs` (gh-pages).
- Run CLI: `uv run python scripts/cli_chat.py --config simple/base.yaml`.
- Build UI wheel: `make build-ui` (requires `npm`).

## Coding Style & Naming Conventions
- Python 3.12; line length 120 (ruff). Use Google-style docstrings.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `CamelCase`.
- Imports: first-party recognized as `utu` (ruff isort). Keep functions small and typed where practical (mypy enabled, strict base with selected relaxations).
- Run `pre-commit install` to enable ruff format/check on commit.

## Testing Guidelines
- Framework: `pytest` (async supported via `pytest-asyncio`).
- Location: mirror source tree under `tests/`; name files `test_*.py` and functions `test_*`.
- Coverage: add tests for new features and bug fixes; include edge cases and config loading paths.
- Quick example: `uv run pytest tests/test_config.py -q`.

## Commit & Pull Request Guidelines
- Link PRs to an issue; keep changes focused and documented.
- Commit style: prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) or scoped prefixes (e.g., `tool_auto_gen:`) observed in history.
- PRs must pass lint, format, and tests; include a clear description, rationale, and any screenshots/logs for UI or eval changes.

## Security & Configuration Tips
- Do not commit secrets. Copy `.env.example` to `.env` and set required keys (LLM/tool APIs). `.env` is git-ignored.
- Prefer `uv run ...` to ensure the virtual env and pinned deps are used.
