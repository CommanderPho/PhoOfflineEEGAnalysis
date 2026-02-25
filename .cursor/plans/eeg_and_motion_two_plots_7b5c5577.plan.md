---
name: EEG and MOTION two plots
overview: Change the Rerun XDF loader so EEG channels and MOTION channels appear on two separate time-series plots by classifying streams by name and logging one entity for EEG and one for MOTION (under distinct paths), keeping the loader standalone.
todos: []
isProject: false
---

# Plot EEG and MOTION on two separate plots

## Context

- **[rerun/rerun_loader_xdf.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/rerun/rerun_loader_xdf.py)** currently logs every XDF stream under a flat path `{prefix}xdf/{safe_stream_name}` (e.g. `xdf/Epoc X`, `xdf/Epoc X Motion`). The Rerun viewer shows **one time-series panel per entity**, so you get one panel per stream.
- The main pipeline in [xdf_files.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/xdf_files.py) classifies streams by **name** via `stream_name_to_modality_dict`: e.g. `"Epoc X"` → EEG, `"Epoc X Motion"` → MOTION (see line 242).
- The loader must stay **standalone** (no dependency on `phoofflineeeganalysis` or `phopylslhelper`), so stream-type classification will be duplicated locally.

## Goal

Exactly **two** time-series plots in the Rerun viewer:

1. **EEG plot** – all channels from stream(s) classified as EEG.
2. **MOTION plot** – all channels from stream(s) classified as MOTION.

## Approach

1. **Classify streams by name** (inside the loader, no new deps). Use a local mapping aligned with the main pipeline, e.g.:
  - `"Epoc X"` → EEG  
  - `"Epoc X Motion"` → MOTION  
   Other names (e.g. `"Epoc X eQuality"`, `"TextLogger"`) can be treated as "other" and either skipped or logged under `xdf/other/{name}`.
2. **One entity per type for the two plots.**
  Log at most two time-series entities:
  - `{prefix}xdf/EEG` – one entity containing the first EEG stream’s channels (same format as now: `rr.send_columns` + `rr.Scalars.columns` + `rr.SeriesLines`).
  - `{prefix}xdf/MOTION` – one entity for the first MOTION stream’s channels.
   If there are multiple EEG or multiple MOTION streams, use the **first** of each (by iteration order). No time alignment or merging of multiple streams; that keeps the change simple and avoids dependency on sync logic.
3. **Optional: other streams.**
  Streams that are neither EEG nor MOTION can be:
  - **A)** Not logged (simplest), or  
  - **B)** Logged under `xdf/other/{safe_name}` so they still appear as extra panels.

Recommendation: **(A)** for minimal scope; **(B)** if you want to see e.g. TextLogger in the same recording.

## Implementation outline

- **File:** [rerun/rerun_loader_xdf.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/rerun/rerun_loader_xdf.py)
- **Add a small helper** (e.g. `_stream_modality(stream) -> str | None`) that returns `"EEG"`, `"MOTION"`, or `None` using `_stream_name(stream)` and a local dict:
  - `{"Epoc X": "EEG", "Epoc X Motion": "MOTION"}` (and optionally more entries for other modalities if you log them).
- **Change `_log_xdf_streams_imu_style`** so that:
  - It first iterates streams and groups them by modality (`EEG` / `MOTION` / other).
  - For modality in `("EEG", "MOTION")`: take the first stream in that group (if any) and log it at `{prefix}xdf/EEG` and `{prefix}xdf/MOTION` respectively. Reuse the existing per-stream logic (time_sec, channel labels, `send_columns`, `SeriesLines`); only the path and the selection of which stream to log change.
  - If you choose option (B) for other streams: log each "other" stream under `{prefix}xdf/other/{safe_name}` as today.
- **Paths:**  
  - EEG: `{prefix}xdf/EEG` (single entity).  
  - MOTION: `{prefix}xdf/MOTION` (single entity).  
  So the viewer shows exactly two time-series panels when both types exist.
- **README:** Update [rerun/README.md](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/rerun/README.md) to state that the loader shows EEG and MOTION on two separate plots (and, if applicable, that only the first stream of each type is shown when multiple exist).

## Summary


| Item                  | Action                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Stream classification | Add `_stream_modality(stream)` and a local name→EEG/MOTION map (no new deps).                                |
| Logging               | Log first EEG stream at `xdf/EEG`, first MOTION at `xdf/MOTION`; same `send_columns` + `SeriesLines` as now. |
| Other streams         | Either skip or log under `xdf/other/...`.                                                                    |
| Docs                  | README: two plots (EEG, MOTION); mention “first stream per type” if multiple.                                |


No new dependencies; no changes to `main_analyze_run.py` or `xdf_files.py`. All edits are in `rerun/rerun_loader_xdf.py` and `rerun/README.md`.