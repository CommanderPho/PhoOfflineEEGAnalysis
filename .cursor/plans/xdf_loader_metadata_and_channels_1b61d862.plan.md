---
name: XDF loader metadata and channels
overview: "Align rerun_loader_xdf.py with main XDF loading: use the same session/recording datetime from the XDF header for Rerun's recording_id by default, and derive channel names the same way as LabRecorderXDF (including unwrapping list-like labels) so the viewer shows correct channel names. No new dependencies; no rerun-specific logic from main_analyze_run.py."
todos: []
isProject: false
---

# Align rerun_loader_xdf with main XDF metadata and channel names

## Current state

- **[rerun/rerun_loader_xdf.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\rerun\rerun_loader_xdf.py)** loads XDF via `pyxdf.load_xdf`, then logs streams with `rr.send_columns` and `rr.Scalars.columns(scalars=time_series)`. It does not set `recording_id` from the file (only from CLI). Channel labels come from `_channel_labels_from_stream`, which reads `stream["info"]["desc"][0]["channels"][0]["channel"]` and uses `c.get("label", c)` without unwrapping list-like values (XDF often has `label: ["AF3"]`).
- **main_analyze_run.py** uses [LabRecorderXDF](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\xdf_files.py) from [xdf_files.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\xdf_files.py). Session datetime comes from the XDF **header**: `header['info']['datetime'][0]` parsed as `"%Y-%m-%dT%H:%M:%S%z"`, then converted to UTC (`file_datetime`). Channel names for fixed-rate streams come from `stream['info']['desc'][0]['channels'][0]` via benedict flatten → DataFrame → `channels_df['label'].to_list()`, with values unwrapped from single-element lists via `unwrap_single_element_listlike_if_needed` (so `['AF3']` → `'AF3'`).

## Goal

By default, the loader should:

1. **Session/recording datetime**: Set Rerun’s `recording_id` from the XDF file’s session datetime (same source as `main_analyze_run`), so the same file is identified consistently.
2. **Channel names**: Derive channel labels exactly like LabRecorderXDF (same desc structure + unwrap), and expose them in the Rerun viewer (named series).

No rerun-specific logic from `main_analyze_run.py` (e.g. caching, spectrograms, MNE) should be added. The rerun package must stay self-contained (no dependency on `phoofflineeeganalysis` or `phopylslhelper`).

---

## 1. Session/recording datetime (default `recording_id`)

- **Source**: Same as [LabRecorderXDF.init_basic_from_lab_recorder_xdf_file](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\xdf_files.py) (lines 302–305):
  - `header['info']['datetime'][0]`
  - Parse with `datetime.strptime(..., "%Y-%m-%dT%H:%M:%S%z")`
  - Convert to UTC (e.g. `.astimezone(timezone.utc)`).
- **Use**: After loading XDF, if the Viewer did not pass a `recording_id` (i.e. `args.opened_recording_id` and `args.recording_id` are both None), set `recording_id` to a stable string from that datetime (e.g. `file_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")`).
- **Fallback**: If `header['info']['datetime']` is missing or parsing fails, leave `recording_id` as None (current behavior).
- **Implementation**: Add a small helper (e.g. `_file_datetime_from_header(header) -> datetime | None`) in `rerun_loader_xdf.py`; call it after `pyxdf.load_xdf` and use it only to compute default `recording_id` when CLI did not provide one.

---

## 2. Channel names (same as main pipeline)

- **Source**: Match [xdf_files.py perform_load_xdf_streams](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\xdf_files.py) (lines 454–458):
  - `stream['info']['desc'][0]['channels'][0]` → list of channel dicts (each with `label`, `type`, etc.).
  - For each channel: get `label` and **unwrap** single-element listlike (e.g. `['AF3']` → `'AF3'`). In the main code this is done with `unwrap_single_element_listlike_if_needed` and building a DataFrame from the flattened structure.
- **In the loader**: Keep logic local (no new deps). Update `_channel_labels_from_stream(stream, n_channels)` to:
  - Use the same desc path: `desc[0]['channels'][0]` and obtain the list of channel items (handling both list-of-dicts and benedict-style nested dict; if needed, replicate the minimal unwrap behavior for `label` only).
  - For each item, take `label` and unwrap: if the value is a list/tuple of length 1, use the single element; otherwise use the value as-is (string or fallback to `ch_{i}`).
  - Return exactly `n_channels` labels; if the desc yields fewer, pad with `ch_{i}`; if more, truncate to `n_channels`.
- **Expose in Rerun**: So the viewer shows these names (e.g. in the time-series legend):
  - After `rr.send_columns(..., columns=rr.Scalars.columns(scalars=time_series))`, log series names at the same entity path: `rr.log(path, rr.SeriesLines(names=channel_labels), static=True)` (Rerun docs: SeriesLines at the same path as Scalars names the series). Use the labels returned by the updated `_channel_labels_from_stream`.
- **Fallback**: If desc is missing or malformed, keep current fallback `ch_0`, `ch_1`, ... and still call `rr.SeriesLines(names=channel_labels)` with that list so behavior is consistent.

---

## 3. Implementation notes

- **Load order**: Parse header for `file_datetime` immediately after `pyxdf.load_xdf`; use it for default `recording_id` before `rr.init`. Then in `_log_xdf_streams_imu_style`, pass `header` only if needed for future use; channel labels come from each `stream` only.
- **API**: `rr.init(application_id, recording_id=recording_id)` with `recording_id` = CLI value or derived ISO UTC string or None. Do not change how `application_id` is chosen.
- **Dependencies**: No new packages. Implement datetime parsing and label unwrapping with the stdlib (+ existing `pyxdf`, `rerun`, `numpy`).
- **Tests**: Manually verify with a LabRecorder XDF that (1) recording shows a recognizable session datetime as recording id when opened without `--recording-id`, and (2) EEG (and other) streams show correct channel names (e.g. AF3, AF4) in the time-series view.

---

## Summary


| Area                  | Change                                                                                                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Session/recording** | Parse XDF header `info.datetime` like LabRecorderXDF; use UTC ISO string as default `recording_id` when Viewer does not supply one.                                                                                        |
| **Channel names**     | Derive labels from `stream['info']['desc'][0]['channels'][0]` with single-element list unwrap; pass same list to `_log_xdf_streams_imu_style` and log it via `rr.SeriesLines(names=channel_labels)` at each stream’s path. |
| **Scope**             | Metadata and channel naming only; no rerun/analysis logic from main_analyze_run; rerun loader remains standalone.                                                                                                          |


