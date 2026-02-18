---
name: Export spectrograms to Rerun
overview: Export computed EEG spectrograms and session datetime from main_analyze_run.py to a file, then add a small viewer script that loads that file and displays the data in a Rerun viewer (run as a separate process).
todos: []
isProject: false
---

# Export spectrograms with datetime and view in Rerun

## Current data shape (from [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py))

- **Per session**: `a_result["spectogram"]` has:
  - `spectogram_result_dict`: `dict[channel_name, (f, t, Sxx)]` where `f` = freq 1D, `t` = time 1D (seconds from recording start), `Sxx` = 2D `(n_freqs, n_times)`.
  - `fs`: sampling-related.
- **Session datetime**: `a_raw.info.get("meas_date")` (datetime). Absolute time for each time bin = `meas_date + timedelta(seconds=t[i])`.

Existing HTML export in `export_session_spectrograms_html` already iterates over `(active_only_out_eeg_raws, results)` and uses `spectogram_result_dict` and `meas_date`; the same data will drive the new export and Rerun logging.

---

## 1. Export spectrograms + datetime to a file (from main_analyze_run)

**Goal**: Write a single file that stores, for each session, session start datetime and per-channel spectrogram arrays (freq, time, Sxx) so another process can open it without re-running the pipeline.

**Recommended format: NumPy `.npz**` (or a ZIP of numpy arrays). No extra dependency; easy to load in the viewer script.

- **Structure** (one option):
  - `session_indices`: 1D array of session indices.
  - For each session `i`:  
  `meas_date_sec_i` (float, Unix timestamp), `channel_names_i` (1D str), `freqs_i` (1D), `times_i` (1D), `Sxx_i` (3D array `n_channels x n_freqs x n_times`), optionally `freq_min`/`freq_max` if you restrict range.
- **Simpler alternative**: One group per session, e.g. `s0_meas_date`, `s0_channels`, `s0_freqs`, `s0_times`, `s0_Sxx`, `s1_...`, etc. Or store a list of dicts in a single pickle/npz (e.g. one array for meas_date_sec and one list of dicts for the rest).

**Concrete steps**:

- Add a function in [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py), e.g. `export_spectrograms_for_rerun(active_only_out_eeg_raws, results, output_path: Path, freq_min=1.0, freq_max=40.0)` that:
  - Loops over sessions; for each, gets `meas_date` and converts to Unix seconds (for Rerun and for storing).
  - Reads `a_result["spectogram"]["spectogram_result_dict"]`, optionally restricts to `[freq_min, freq_max]` (same as HTML export), builds arrays `freqs`, `times`, and stacked `Sxx` (n_channels, n_freqs, n_times).
  - Saves to one `.npz` with keys like `session_*` or a structured layout as above (including `meas_date_sec` and channel names so the viewer can label entities).
- Call this from the `if __name__ == "__main__"` block after `process_XDFs_main` and the existing HTML export (reuse the same `active_only_out_eeg_raws` and `results`).

**Optional**: Also support exporting directly to a Rerun `.rrd` file from this script (see below). That would add an optional dependency on `rerun-sdk` and a flag like `export_rerun_rrd=False`. Then “another process” can be simply running the Rerun viewer on that `.rrd`.

---

## 2. Display in Rerun viewer (other process)

Two ways to “display in a Rerun viewer in another process”:

### Option A – Viewer script (recommended): load exported file and log to Rerun

- **New script** (e.g. `examples_jupyter/view_spectrograms_rerun.py` or `scripts/view_spectrograms_rerun.py`) that:
  - Takes the exported file path as argument (e.g. the `.npz`).
  - Uses `rerun` only in this script (so the main analysis script stays free of `rerun-sdk` if you prefer).
  - Loads the npz; for each session:
    - Sets time so the session is at the correct datetime: `rr.set_time_seconds("session_time", meas_date_sec)` (or `rr.set_time(timestamp=...)` with a datetime if the API accepts it). This makes the Rerun timeline show real time.
    - For each channel, logs the 2D spectrogram as an image:
      - Convert `Sxx` to a displayable 2D (e.g. dB: `10 * np.log10(Sxx + 1e-12)`), then normalize to 0–1 or 0–255 for grayscale.
      - Log with `rr.log(f"session_{i}/channel_{ch_name}", rr.Image(...))` (or use `rr.Tensor` for heatmap-style; Rerun supports 2D grayscale and tensor heatmaps).
  - Then either:
    - **Save to `.rrd**`: `rr.save("spectrograms.rrd")` and exit; user runs `rerun spectrograms.rrd` in another process, or
    - **Spawn viewer**: `rr.spawn()` (or connect to a running viewer) so the user runs this script and sees the viewer in the same run.
- **Dependency**: Add `rerun-sdk` only for this script (e.g. optional extra or in a separate env). Use `uv add rerun-sdk` in the project (or optional dependency).

### Option B – Export `.rrd` directly from main_analyze_run

- In [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py), add an optional branch (e.g. behind `export_rerun_rrd=True`) that:
  - Imports `rerun as rr`, initializes with `rr.init("pho_eeg_spectrograms")`, then for each session/channel logs the spectrogram image and sets time via `rr.set_time_seconds("session_time", meas_date_sec)`.
  - Calls `rr.save(outputs_root_folder / "spectrograms.rrd")` (or a path passed in).
- “Another process” = run the Rerun viewer on the saved file: `rerun path/to/spectrograms.rrd`.
- **Downside**: Main script (or its environment) must depend on `rerun-sdk`. Prefer Option A if you want to keep the main pipeline dependency-light.

---

## 3. Rerun time and image details

- **Time**: Use one timeline, e.g. `"session_time"`, in seconds since Unix epoch. Convert `meas_date` with `meas_date.timestamp()` (and ensure timezone-aware if needed). Then `rr.set_time_seconds("session_time", t)` before logging each session’s data so the Rerun time panel shows datetimes and sessions are ordered correctly.
- **Image**: For each channel, log the 2D spectrogram (freq × time) as one image. Use `rr.Image(..., color_model="l")` for grayscale, or `rr.Tensor(..., dim_names=["freq","time"])` for heatmap-style. Normalize dB values to 0–1 (or 0–255) so the viewer shows a sensible range.
- **Entity path**: Use a hierarchy like `sessions/session_{i}/{ch_name}` or `sessions/session_{i}/spectrograms/{ch_name}` so multiple sessions and channels are easy to toggle in the viewer.

---

## 4. Suggested implementation order

1. Add **export function** in `main_analyze_run.py`: write spectrograms + `meas_date_sec` + channel names + freq/time axes to one `.npz` (with optional freq_min/freq_max filtering).
2. Add **viewer script** that loads the `.npz`, initializes Rerun, sets time per session, logs each channel’s spectrogram as Image (or Tensor), then saves to `.rrd` and/or spawns the viewer.
3. Document: “Run analysis → export to `spectrograms_export.npz` → run `python view_spectrograms_rerun.py spectrograms_export.npz` (or open the generated `.rrd` with `rerun …`) in another process.”
4. Optional: add a second path that writes `.rrd` directly from `main_analyze_run.py` if you want a one-step export without a separate viewer script.

---

## 5. File summary


| File                                                                                                            | Change                                                                                                          |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [main_analyze_run.py](c:\Users\pho\repos\ACTIVE_DEV\PhoOfflineEEGAnalysis\examples_jupyter\main_analyze_run.py) | Add `export_spectrograms_for_rerun(...)` and call it from `__main__` (and optionally add direct `.rrd` export). |
| New script (e.g. `view_spectrograms_rerun.py`)                                                                  | Load npz, loop sessions/channels, set time, log images, `rr.save()` and/or `rr.spawn()`.                        |
| Project deps                                                                                                    | Add `rerun-sdk` (e.g. via `uv add rerun-sdk`), at least for the viewer script.                                  |


This gives you a clear “export to file” step and a separate “view in Rerun” process, with the option to keep the main pipeline free of Rerun by using the viewer script as the only Rerun consumer.