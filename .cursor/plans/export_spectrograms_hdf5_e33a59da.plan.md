---
name: Export spectrograms HDF5
overview: Add a new `export_spectrograms_hdf5` function in `main_analyze_run.py` that writes spectrograms, timestamps, and recording metadata to a single HDF5 file for broad tooling compatibility (Python, MATLAB, R, Julia, etc.), reusing existing spectrogram extraction logic and optional stream info.
todos: []
isProject: false
---

# Export spectrograms and recording info to HDF5

## Context

- [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py) already has several export helpers that take `active_only_out_eeg_raws`, `results`, and an output path: `export_session_spectrograms_html`, `export_combined_spectrograms_html`, and `export_spectrograms_for_rerun` (which writes `.npz`).
- Spectrogram data lives in `a_result["spectogram"]["spectogram_result_dict"]`: dict of `channel_name -> (f, t, Sxx)` with `Sxx` shape `(n_freqs, n_times)`. Recording/session info comes from `a_raw.info` (e.g. `meas_date`, `sfreq`, `ch_names`, duration) and optionally from `stream_infos_df` (e.g. `xdf_filename`, `xdf_dataset_idx`) as in `compute_session_summary_metrics`.
- The project already depends on **h5py** ([pyproject.toml](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\pyproject.toml) line 40); no new dependency is required.

## Goal

Add one new function that exports the same spectrogram data plus timestamps and recording metadata into a single **HDF5** file so that other programs (MATLAB, R, Julia, etc.) can import and analyze it without custom Python code.

## Design

### Function

- **Name**: `export_spectrograms_hdf5`
- **Location**: In [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py), immediately after `export_spectrograms_for_rerun` (around line 462), keeping two blank lines between functions per user preference.
- **Signature** (single line where possible):
  - `active_only_out_eeg_raws`, `results`, `output_path: Path`
  - `freq_min: float = 1.0`, `freq_max: float = 40.0`
  - `stream_infos_df: Optional[pd.DataFrame] = None` (optional; used to attach `xdf_filename` and any other per-session columns to each session group)

### HDF5 layout (interchange-friendly)

- **Root attributes**: `format_version` (e.g. `"1.0"`), `freq_min`, `freq_max`, `n_sessions`.
- **Per-session group**: e.g. `/sessions/session_000`, `/sessions/session_001`, …
  - **Datasets** (numeric/string only for portability):
    - `freqs`: 1D float (Hz)
    - `times`: 1D float (seconds, relative to session start)
    - `channel_names`: 1D array of strings (variable-length UTF-8 or fixed-length; h5py supports both; prefer variable-length for clarity)
    - `spectrogram`: 3D float, shape `(n_channels, n_freqs, n_times)` (same as Rerun export)
  - **Group attributes** (scalar / short strings): `meas_date_iso`, `meas_date_sec`, `sfreq_hz`, `duration_s`, `n_channels`; and if `stream_infos_df` is provided and has a row for this session: `xdf_filename` (and optionally other safe scalar columns). Avoid storing complex Python objects; stick to numbers and strings so MATLAB/R/Julia can read them.

### Implementation details

- Reuse the same extraction loop as `export_spectrograms_for_rerun`: skip when `a_result is None` or `"spectogram"` missing; get `meas_date`, normalize to UTC and compute `meas_date_sec` and ISO string; build `channel_names`, apply `freq_mask` to get `freqs` and filter `Sxx` per channel; stack into `Sxx_stack` (n_channels x n_freqs x n_times).
- Use `h5py` to create the file and write root attributes, then a `sessions` group and one subgroup per exported session. Write datasets with simple dtypes; for `channel_names`, use `h5py.special_dtype(vlen=str)` (or equivalent) so other tools can read string arrays.
- Derive per-session recording metadata from `a_raw.info` (duration from `a_raw.times`, sfreq, ch count) and, when `stream_infos_df` is provided, the same `dataset_to_filename`-style lookup by `xdf_dataset_idx` as in `compute_session_summary_metrics` (group by `xdf_dataset_idx`, take first `xdf_filename` per dataset index).
- Docstring: state that the output is HDF5 for interchange (Python, MATLAB, R, Julia, etc.), list what is stored (spectrograms, time/freq axes, channel names, session timestamps and recording info), and mention optional `stream_infos_df` for filenames.

### Call site

- In the `if __name__ == "__main__"` block, after the existing `export_spectrograms_for_rerun` call (around line 861), add a call to `export_spectrograms_hdf5` with:
  - `output_path = outputs_root_folder / "spectrograms_export.h5"`
  - Same `freq_min`/`freq_max` as other exports
  - `stream_infos_df=_out_xdf_stream_infos_df`
- Optionally add a single print line in the final “Processing Complete” summary that mentions the HDF5 export path.

## Files to change


| File                                                                                                            | Change                                                                                                                                                                                                                              |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py) | Add `import h5py` at top (if not already present). Add `export_spectrograms_hdf5` after `export_spectrograms_for_rerun`. In `if __name__ == "__main__"`, call the new function and include the HDF5 path in the completion summary. |


## Out of scope

- No new dependency (h5py already in project).
- No change to Rerun or .npz export; HDF5 is an additional format.
- Reading/validation script for the HDF5 file is not required unless you ask for it later.

