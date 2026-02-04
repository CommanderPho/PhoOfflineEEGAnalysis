# Template Processing

This directory contains template files for generating `[tool.uv.sources]` sections in `pyproject.toml`.

## Machine-Specific Path Configuration

The templates use the `{ACTIVE_DEV_PATH_PREFIX}` placeholder to support different directory structures across machines. This functionality is integrated into the `uv-deps-switcher` tool.

### Setup

**On this machine** (repos are siblings in `ACTIVE_DEV`):
- Leave `ACTIVE_DEV_PATH_PREFIX` unset or set it to empty string
- Paths will be: `../PhoPyLSLhelper`, `../PhoPyMNEHelper`, etc.

**On other machine** (repos are in `ACTIVE_DEV` folder):
- Set environment variable: `ACTIVE_DEV_PATH_PREFIX=ACTIVE_DEV/`
- Paths will be: `../ACTIVE_DEV/PhoPyLSLhelper`, `../ACTIVE_DEV/PhoPyMNEHelper`, etc.

### Usage

Use `uv-deps-switcher` to process templates and update `pyproject.toml`:

```bash
# Switch to dev mode (processes templates with environment variable substitution)
uv-deps-switcher dev

# Switch to release mode
uv-deps-switcher release

# On other machine (with prefix set in environment)
set ACTIVE_DEV_PATH_PREFIX=ACTIVE_DEV/  # Windows
export ACTIVE_DEV_PATH_PREFIX=ACTIVE_DEV/  # Linux/Mac
uv-deps-switcher dev
```

The `uv-deps-switcher` tool automatically:
- Reads template files from this `templating/` directory
- Substitutes `{ACTIVE_DEV_PATH_PREFIX}` with the environment variable value
- Updates the `[tool.uv.sources]` section in `pyproject.toml`

### Files

- `pyproject_template_dev.toml_fragment`: Template for development dependencies (local paths)
- `pyproject_template_release.toml_fragment`: Template for release dependencies (git sources)

See the `uv-deps-switcher` README for more information about template processing and usage.
