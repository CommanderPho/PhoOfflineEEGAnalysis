---
name: Time-dependent bad channel detection
overview: Add time-dependent low-quality EEG channel detection in 3-second windows to `EEGComputations` in PhoPyMNEHelper, using MNE's `find_bad_channels_lof` on each windowed segment and returning a structured result suitable for overlays and HDF export.
todos: []
isProject: false
---

# Time-dependent bad channel detection (3s increments)

## Goal

Implement detection of bad (low-quality) EEG channels in **3-second time windows** inside [EEG_data.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\src\phopymnehelper\EEG_data.py), using MNE helpers where applicable.

## Key findings

- **MNE** provides `mne.preprocessing.find_bad_channels_lof(raw, n_neighbors=20, picks=None, threshold=1.5, return_scores=False)` (MNE 1.7+). It expects a `Raw` instance and returns a list of bad channel names (and optionally LOF scores). It uses the full time series per channel (so running it on a **cropped** Raw gives per-window bad channels).
- **PhoPyMNEHelper** already depends on `mne>=1.8.0` ([pyproject.toml](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\pyproject.toml)); LOF requires **scikit-learn** (MNE uses `_soft_import("sklearn", ...)` in `_lof.py`). Add `scikit-learn` to dependencies if not already pulled in by MNE.
- **Raw.crop** in MNE is **in-place**; use `raw.copy().crop(tmin, tmax)` to get a 3s segment without altering the original.
- Existing pattern in the same file: [EEG_data.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\src\phopymnehelper\EEG_data.py) uses `EEGComputations.all_fcns_dict()` and `run_all(raw, **kwargs)` to run several computations; each returns a dict. HDF export in `to_hdf` recurses into dicts and writes DataFrames/arrays; list-of-lists or list-of-tuples are not currently written, so the new result should include a **DataFrame** for HDF compatibility.

## Implementation plan

### 1. New method: `EEGComputations.time_dependent_bad_channels`

- **Location**: [EEG_data.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\src\phopymnehelper\EEG_data.py) in class `EEGComputations` (after existing computation methods, before `perform_write_to_hdf`).
- **Signature** (single line per project rules):
`def time_dependent_bad_channels(cls, raw: mne.io.Raw, window_sec: float = 3.0, picks=None, n_neighbors: int = 20, threshold: float = 1.5, return_scores: bool = False, **kwargs) -> dict:`
- **Logic**:
  - If `picks is None`, use `mne.pick_types(raw.info, eeg=True, meg=False)`.
  - Get total duration: `duration_sec = raw.times[-1] - raw.times[0]` (or equivalent from `raw.n_times` and `raw.info['sfreq']`).
  - Build non-overlapping windows: `t_start = 0, window_sec, 2*window_sec, ...` until `t_start < duration_sec`. For each window: `t_end = min(t_start + window_sec, duration_sec)` (last window may be shorter than 3s).
  - For each window:
    - `raw_seg = raw.copy().crop(tmin=t_start, tmax=t_end)` (no in-place change to original).
    - Call `mne.preprocessing.find_bad_channels_lof(raw_seg, n_neighbors=n_neighbors, threshold=threshold, picks=picks, return_scores=return_scores, **kwargs)`.
    - Collect: `(t_start, t_end, list_of_bad_ch_names)` and optionally scores.
  - Build return dict (see below). Use a **try/import** around `find_bad_channels_lof` and document that `scikit-learn` is required for this method (or add sklearn to pyproject.toml).
- **Return structure** (dict):
  - `window_sec`: float (e.g. 3.0).
  - `intervals`: list of `(t_start, t_end)` in seconds.
  - `bad_channels_per_interval`: list of lists of channel names (same length as `intervals`).
  - `df`: `pd.DataFrame` with columns `t_start`, `t_end`, `n_bad`, `bad_channels` (each cell a list of str). This allows existing `to_hdf` recursion to write the result (DataFrame branch).
  - If `return_scores`: `scores_per_interval`: list of 1d arrays (one per window), same order as `intervals`.
- **Edge cases**:
  - Recording shorter than 3s: one window with actual duration.
  - Empty picks / no EEG: handle gracefully (e.g. return empty intervals and empty lists).
  - LOF can fail on very few channels; consider a minimum number of channels (e.g. < 3) and skip or return empty bads for that window with a log message.

### 2. Register in pipeline

- In `all_fcns_dict()` add entry: `'time_dependent_bad_channels': cls.time_dependent_bad_channels`.
- Then `EEGComputations.run_all(raw)` will include this computation. Callers that want to skip it can be given an option later (e.g. exclude list) if needed; for now adding it keeps the API consistent with other computations.

### 3. Dependencies

- Add **scikit-learn** to [pyproject.toml](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\pyproject.toml) dependencies (e.g. `scikit-learn>=1.0`) so that `find_bad_channels_lof` runs without optional-import failures. Confirm MNE does not already bring it in as a hard dependency for this code path.

### 4. HDF export

- No change required to `to_hdf` / `_perform_write_dict_recurrsively`: the new result is a dict containing a `df` key (DataFrame). The recursive writer will write that DataFrame. Optionally store `intervals` and `bad_channels_per_interval` as well; if those are written, add a branch for “list of tuples” / “list of lists” only if you want them in HDF (e.g. as arrays or encoded in a table). Minimal approach: rely on `df` for HDF.

### 5. Optional downstream use

- [main_analyze_run.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py) already has `_get_channel_bad_intervals(raw, channel_names)` for **annotation-based** bad segments (BAD_*). The new function is **quality-based** (LOF per window). If you later want to overlay time-dependent LOF bad channels on spectrograms, you can derive per-channel bad intervals from `time_dependent_bad_channels`: for each channel, collect `(t_start, t_end)` for every window where that channel is in `bad_channels_per_interval`, then merge overlapping intervals. This can be a small helper in PhoOfflineEEGAnalysis or inside the same module; **out of scope** for this plan unless you want it included.

## Summary


| Item       | Action                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| New method | `EEGComputations.time_dependent_bad_channels(raw, window_sec=3.0, picks=None, n_neighbors=20, threshold=1.5, return_scores=False, **kwargs)` |
| Algorithm  | Sliding 3s windows; for each, `raw.copy().crop(...)` then `find_bad_channels_lof(raw_seg, ...)`                                              |
| Output     | Dict with `window_sec`, `intervals`, `bad_channels_per_interval`, `df` (DataFrame), optionally `scores_per_interval`                         |
| Pipeline   | Add to `all_fcns_dict()` as `'time_dependent_bad_channels'`                                                                                  |
| Deps       | Add `scikit-learn` in PhoPyMNEHelper if not already present                                                                                  |


No changes to PhoOfflineEEGAnalysis or notebooks in this plan; they can start using the new key from `EEGComputations.run_all(raw)` once implemented.