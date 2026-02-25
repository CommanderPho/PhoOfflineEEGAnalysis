<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Cursor Cloud specific instructions

### Project overview

PhoOfflineEEGAnalysis is a Python 3.10 EEG analysis toolkit for Emotiv Epoc/Epoc X recordings. It is a local analysis/notebook project, not a web service. See `openspec/project.md` for full architecture details.

### Sibling repos

The project depends on 4 editable sibling packages configured in `[tool.uv.sources]` of `pyproject.toml`. In the cloud environment these are cloned to `/ACTIVE_DEV/`:

| Package | Local path | GitHub repo |
|---|---|---|
| `mne` (fork) | `/ACTIVE_DEV/mne-python` | `CommanderPho/mne-python` |
| `phopylslhelper` | `/ACTIVE_DEV/PhoPyLSLhelper` | `CommanderPho/phopylslhelper` |
| `phopymnehelper` | `/ACTIVE_DEV/PhoPyMNEHelper` | `CommanderPho/PhoPyMNEHelper` |
| `py-pho-timeline` | Not available | Repo `pyPhoTimeline` is not public |

**mne-python version tag gotcha:** The fork uses `hatch-vcs` for versioning. A shallow clone has no reachable tags, so `git describe` fails and the package reports version `0.0.0.dev...`, breaking resolution for `phopymnehelper` (which requires `mne>=1.8.0`). After cloning, run `git fetch --tags --depth=1 origin` and then `git tag v1.10.3 HEAD` (or whatever the latest upstream tag is) to fix this.

### Running the application

- **CLI entrypoint (stub):** `uv run phoofflineeeganalysis` — prints a hello message
- **Jupyter notebooks:** `uv run jupyter lab` — primary development workflow; notebooks are in `examples_jupyter/`
- **Script runner:** `uv run python examples_jupyter/main_analyze_run.py`

### Linting and testing

- No formal test suite or linter config exists yet (see `openspec/project.md` Testing Strategy for aspirational plan)
- Verify imports with: `uv run python -c "import phoofflineeeganalysis; import mne; import phopylslhelper; import phopymnehelper"`

### Key caveats

- **Python 3.10 required** — pinned in `.python-version`; `pyproject.toml` requires `>=3.10,<3.11`
- **PyQt5 GUI widgets need a display** — use `xvfb-run` or `DISPLAY=:99` with Xvfb for headless environments
- **EEG data paths are developer-local** — notebooks reference `E:/Dropbox (Personal)/Databases/...`; these won't exist in cloud
- **numpy < 1.24 constraint** — project pins `numpy>=1.20,<1.24` which conflicts with some newer deps; the lockfile resolves this