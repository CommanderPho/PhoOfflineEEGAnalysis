---
name: Restructure to src layout
overview: Restructure PhoPyMNEHelper package to use the conventional `src/...` directory layout and update pyproject.toml configuration accordingly.
todos: []
---

# Restructure PhoPyMNEHelper to src/ Layout

## Current Structure

- Package files are at the root level
- `main.py` is a standalone script
- `pyproject.toml` lacks build system configuration

## Target Structure

```javascript
PhoPyMNEHelper/
  src/
    phopymnehelper/
      __init__.py
      __main__.py
  pyproject.toml
  README.md
  ...
```



## Implementation Steps

### 1. Create src/ directory structure

- Create `src/phopymnehelper/` directory
- Create `src/phopymnehelper/__init__.py` (can be empty or contain package-level exports)
- Create `src/phopymnehelper/__main__.py` with the content from `main.py` to allow running as `python -m phopymnehelper`

### 2. Update pyproject.toml

- Add `[build-system]` section with `hatchling` (consistent with PhoOfflineEEGAnalysis)
- Add `[tool.hatchling.build.targets.wheel]` section to specify `packages = ["src/phopymnehelper"]` or use automatic discovery
- Optionally add `[project.scripts]` entry point if a CLI command is desired

### 3. Remove old main.py

- Delete `main.py` from root after moving its content to `__main__.py`

## Files to Modify

- [pyproject.toml](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\pyproject.toml) - Add build system and package configuration
- [main.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\main.py) - Delete (content moved to `__main__.py`)