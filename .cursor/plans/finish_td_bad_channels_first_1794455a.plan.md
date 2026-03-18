---
name: Finish TD Bad Channels First
overview: Complete `time_dependent_bad_channels` behavior and ensure it executes first in the `EEGComputations.run_all` pipeline by reordering the function registry.
todos:
  - id: reorder-pipeline
    content: Move `time_dependent_bad_channels` to first entry in `EEGComputations.all_fcns_dict()`.
    status: completed
  - id: harden-time-dependent-method
    content: Finalize `time_dependent_bad_channels` edge-case handling and keep return structure consistent.
    status: completed
  - id: verify-run-and-lints
    content: Validate execution order and output shape via `run_all`, then check lints for edited files.
    status: completed
isProject: false
---

# Finish `time_dependent_bad_channels` and run first

## Target files

- [C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoPyMNEHelper/src/phopymnehelper/EEG_data.py](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoPyMNEHelper/src/phopymnehelper/EEG_data.py)

## What I will change

- Update `EEGComputations.all_fcns_dict()` so `time_dependent_bad_channels` is the first entry, which makes it run first in `run_all()`.
- Finalize `time_dependent_bad_channels()` by tightening edge-case handling and output consistency while preserving current return contract:
  - Validate/normalize `picks` and window sizing.
  - Ensure LOF neighbor count is safely bounded for current channel count.
  - Keep stable output keys (`window_sec`, `intervals`, `bad_channels_per_interval`, `df`, optional `scores_per_interval`).
  - Ensure per-window score output is shape-stable when `return_scores=True`.
- Keep edits minimal and local to this method and registry ordering only.

## Implementation notes

- Pipeline order is currently controlled only by insertion order in `all_fcns_dict()`.
- `run_all()` already forwards `**kwargs` to this method, so no caller changes are required for this scope.

## Verification

- Run a quick local invocation path through `EEGComputations.run_all(raw=...)` to confirm:
  - `time_dependent_bad_channels` executes first.
  - Return object includes expected keys and dataframe columns.
- Run lint diagnostics for edited file(s) and fix any introduced issues.

