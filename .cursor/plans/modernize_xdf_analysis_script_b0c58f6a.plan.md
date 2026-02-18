---
name: Modernize XDF analysis script
overview: Update imports in main_analyze_run.py to remove duplicates and dead code, use standard typing, and ensure the script runs as a standalone (including optional hardening for headless/CLI use).
todos: []
isProject: false
---

# Modernize standalone XDF analysis script

## Current state

- [main_analyze_run.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py) is an ~829-line script with a `if __name__ == "__main__"` block that calls `process_XDFs_main()` to process Lab Recorder XDF files, run EEG computations, and export spectrograms/summaries.
- **Import issues:**
  - **Duplicate NDArray:** Both `from nptyping import NDArray` and `from numpy.typing import NDArray` (lines 18–19). The second shadows the first; only one should remain.
  - **Unused imports:** `read_raw` (mne.io), `PlayerLSL`/`StreamLSL` (mne_lsl), `IPython`, `InteractiveShell` are never used in the file. `phopylslhelper.easy_time_sync` (EasyTimeSyncParsingMixin, readable_dt_str, from_readable_dt_str) is also imported but never referenced in the script.
  - **NDArray** is not used in the script body (only in import lines); you can keep a single standard import for future type hints or remove it.
- **Run-time behavior:** At module load (lines 415–421) the script sets MNE Qt browser backend, sets log level, and loads Holoviews/hvplot/panel extensions. This runs even when the script is imported (e.g. for `process_XDFs_main`). For standalone CLI use, Qt/Bokeh init can fail in headless environments; consider making this block conditional on `__name__ == "__main__"` or wrapping in try/except so the script still runs when no display is available.

## Recommended changes

### 1. Imports (top of file)

- **NDArray:** Remove `from nptyping import NDArray`. Keep `from numpy.typing import NDArray` as the single source (standard library typing; pyproject already has `nptyping` but `numpy.typing` is preferred for type hints). If you prefer zero type-hint imports and NDArray is unused, remove both lines.
- **Dead third-party imports:** Remove:
  - `from mne.io import read_raw`
  - `from mne_lsl.player import PlayerLSL as Player`
  - `from mne_lsl.stream import StreamLSL as Stream`
  - `import IPython` and `from IPython.core.interactiveshell import InteractiveShell`
- **Dead project import:** Remove the entire line:
  - `from phopylslhelper.easy_time_sync import EasyTimeSyncParsingMixin, readable_dt_str, from_readable_dt_str`
  (Nothing in the script references these symbols.)

No other import paths need changing: `phoofflineeeganalysis` and remaining packages are used and match the project layout (e.g. [EegVisualization.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\EegVisualization.py), [PendingNotebookCode.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\PendingNotebookCode.py)).

### 2. Module-level config (lines 414–424)

- **Optional robustness for standalone/headless:** Move the block that sets MNE backend, log level, and hv/pn extensions (lines 415–421) so it runs only when executing as main (e.g. indent under `if __name__ == "__main__":` at the start of the main block), or wrap in try/except and fall back to a non-GUI MNE backend (e.g. `"matplotlib"`) and skip or no-op extension loading on failure. This avoids import/startup failures when run from CLI or headless environments.
- Remove the now-unused comment about `InteractiveShell.ast_node_interactivity` (line 424) when removing IPython imports.

### 3. Verify run

- After edits, run from project root with the project’s env, e.g.  
`uv run python examples_jupyter/main_analyze_run.py`  
Ensure paths (e.g. `db_root_path`, `lab_recorder_output_path`, `outputs_root_folder`) exist on the machine or the script documents them clearly; the script already uses assertions for key paths.

## Summary of edits


| Location      | Action                                                                                                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Lines 18–19   | Remove `from nptyping import NDArray`; keep `from numpy.typing import NDArray` (or remove both if dropping NDArray entirely).             |
| Lines 24–26   | Remove `read_raw`, `Player`, `Stream` imports.                                                                                            |
| Lines 35–37   | Remove IPython import block.                                                                                                              |
| Line 40       | Remove phopylslhelper `easy_time_sync` import line.                                                                                       |
| Lines 414–424 | Optionally move MNE/hv/pn config into `if __name__ == "__main__"` or guard with try/except for headless; remove InteractiveShell comment. |


No changes to `pyproject.toml` or dependencies are required for the import cleanup; the script will still run with `uv sync --all-extras` as per project rules.