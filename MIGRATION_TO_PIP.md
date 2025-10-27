# Migration from UV to Pip

This document explains the changes made to convert the project from `uv` to standard `pip` for package management.

## Summary of Changes

The project has been updated to use standard Python package management tools (`pip` and `venv`) instead of `uv`.

## Quick Start (New Installation)

Instead of the old uv-based installation:

```bash
# OLD (uv-based)
uv venv
source .venv/bin/activate
uv sync --group dev
```

Use the new pip-based installation:

```bash
# NEW (pip-based)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## For Existing Users

If you already have the project set up with `uv`, you can migrate to `pip`:

1. **Remove the existing virtual environment:**
   ```bash
   rm -rf .venv
   ```

2. **Create a new virtual environment with venv:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # For development (includes all dev tools)
   pip install -r requirements-dev.txt
   
   # OR for all features (includes dev + all optional dependencies)
   pip install -r requirements-all.txt
   
   # OR for minimal installation (core only)
   pip install -r requirements.txt
   ```

## Requirements Files

The project now includes several requirements files for different use cases:

- **`requirements.txt`** - Core dependencies only
- **`requirements-dev.txt`** - Core + development tools (linting, testing, docs)
- **`requirements-litllm.txt`** - Core + LiteLLM support
- **`requirements-documents.txt`** - Core + document processing
- **`requirements-gaia.txt`** - Core + GAIA benchmark tools
- **`requirements-python-executor.txt`** - Core + Python execution tools
- **`requirements-search.txt`** - Core + web search/crawling
- **`requirements-all.txt`** - Everything (all of the above)

## Makefile Changes

Common `make` commands still work the same way:

```bash
make sync          # Install all dependencies
make format        # Format code
make lint          # Run linter
make build-docs    # Build documentation
make serve-docs    # Serve documentation locally
```

These commands now use `pip` and `python -m` internally instead of `uv`.

## Running Scripts

Instead of:
```bash
uv run python scripts/cli_chat.py --config simple/base.yaml
```

Use:
```bash
python scripts/cli_chat.py --config simple/base.yaml
```

Make sure your virtual environment is activated first!

## Files Changed

- **Created:**
  - `requirements.txt` and various `requirements-*.txt` files
  - This migration guide

- **Updated:**
  - `README.md` - Installation instructions
  - `Makefile` - Build commands
  - `docs/quickstart.md` - Quick start guide
  - `docs/quickstart_beginner.md` - Beginner's guide
  - `docs/frontend.md` - Frontend installation
  - `docs/auto_generation.md` - Tool generation
  - `utu/tools/search/duckduckgo_search.py` - Error messages
  - `utu/tools/search/crawl4ai_crawl.py` - Error messages
  - `utu/meta/tool_generator_mcp.py` - Virtual environment creation

## Note on pyproject.toml

The `pyproject.toml` file has been kept as-is. It still contains all dependency information and is used as the source of truth for the requirements files. The `[tool.uv]` section can be safely ignored when using pip.

## Getting Help

If you encounter any issues during migration:

1. Make sure you're using Python 3.12 or higher
2. Delete your old `.venv` directory completely
3. Create a fresh virtual environment
4. Install dependencies from scratch using the appropriate requirements file

For additional help, please open an issue on GitHub.

