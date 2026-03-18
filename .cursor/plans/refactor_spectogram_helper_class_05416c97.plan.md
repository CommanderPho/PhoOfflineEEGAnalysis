---
name: Refactor Spectogram Helper Class
overview: Introduce a new `SpectogramPlottingHelper` class in `main_analyze_run.py` and move the spectogram-related top-level functions into it as classmethods while preserving existing behavior and call sites.
todos:
  - id: add-helper-class
    content: Create `SpectogramPlottingHelper` and move target spectogram functions into class as classmethods.
    status: completed
  - id: update-internal-calls
    content: Replace internal helper references with `cls.` calls where methods call one another.
    status: completed
  - id: update-call-sites
    content: Switch existing top-level invocations to `SpectogramPlottingHelper.<method>(...)`.
    status: completed
  - id: validate-refactor
    content: Run lints/sanity checks on `main_analyze_run.py` to ensure no introduced errors.
    status: completed
isProject: false
---

# Refactor Spectogram Functions Into Class

## Scope

Refactor the spectogram-related top-level functions in [C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/main_analyze_run.py](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/main_analyze_run.py) (the block currently around lines 299-1050) into a new `SpectogramPlottingHelper` class, with each moved function implemented as a `@classmethod`.

## Implementation Plan

- Add a new class `SpectogramPlottingHelper` in [C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/main_analyze_run.py](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/main_analyze_run.py), placed where the current spectogram helper/export functions live.
- Move these functions into the class and convert each to `@classmethod` (minimal body changes):
  - `_style_spectrogram_bokeh_plots`
  - `export_session_spectrograms_html`
  - `export_combined_spectrograms_html`
  - `_safe_meas_date_sec`
  - `_extract_raw_export_payload`
  - `_compute_nearest_spectrogram_bin_idx`
  - `_build_alignment_export_payload`
  - `export_spectrograms_for_rerun`
  - `export_spectrograms_hdf5`
  - `export_spectrograms_netcdf`
  - `export_spectrograms_parquet`
- Update intra-helper calls to use `cls.` (for example, `cls._extract_raw_export_payload(...)`, `cls._build_alignment_export_payload(...)`, `cls._style_spectrogram_bokeh_plots(...)`) so method dispatch remains internal to the class.
- Keep `_get_channel_bad_intervals` as-is unless needed for cohesion; classmethods can still call it directly with no behavior change.
- Update all direct call sites in the processing section to use the class API:
  - `SpectogramPlottingHelper.export_session_spectrograms_html(...)`
  - `SpectogramPlottingHelper.export_spectrograms_for_rerun(...)`
  - `SpectogramPlottingHelper.export_spectrograms_hdf5(...)`
  - `SpectogramPlottingHelper.export_spectrograms_netcdf(...)`
  - `SpectogramPlottingHelper.export_spectrograms_parquet(...)`
  - (and the commented combined-export call for consistency)
- Preserve signatures and defaults unless required for classmethod conversion (`cls` first argument), and keep formatting conventions (single-line signatures where feasible; two blank lines between class methods).

## Validation

- Run a quick static sanity pass for unresolved names from refactor (especially former top-level helper references).
- Run lints for [C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/main_analyze_run.py](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/main_analyze_run.py) and fix any introduced issues.
- Confirm no functional behavior change in export flow (HTML/NPZ/HDF5/NetCDF/Parquet paths still execute from `__main`__).

