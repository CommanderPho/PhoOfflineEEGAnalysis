---
name: EEG and MOTION two plots
overview: Change the Rerun XDF loader so EEG and MOTION appear on two separate time-series plots and all Text Logger streams are merged into one Text panel, by classifying streams by name and logging accordingly, keeping the loader standalone.
todos: []
isProject: false
---

# Plot EEG and MOTION on two separate plots; merge Text Logger into one Text panel

## Context

- **[rerun/rerun_loader_xdf.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/rerun/rerun_loader_xdf.py)** currently logs every XDF stream under a flat path `{prefix}xdf/{safe_stream_name}` (e.g. `xdf/Epoc X`, `xdf/Epoc X Motion`). The Rerun viewer shows **one time-series panel per entity**, so you get one panel per stream. Non-numeric streams (e.g. TextLogger) are skipped today because the code only logs numeric `time_series`.
- The main pipeline in [xdf_files.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/xdf_files.py) classifies streams by **name** via `stream_name_to_modality_dict`: e.g. `"Epoc X"` → EEG, `"Epoc X Motion"` → MOTION, `"TextLogger"` (and `"EventBoard"`) → PHO_LOG (see line 242).
- TextLogger streams in XDF are **irregular marker streams**: `time_series` is a list of strings, one per event; `time_stamps` gives the timestamp for each. The phoRerunTesting converter logs these with `rr.log(entity, rr.TextLog(text))` at each event time ([xdf_to_rerun.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/phoRerunTesting/converters/xdf_to_rerun.py) `_stream_markers_to_rerun`).
- The loader must stay **standalone** (no dependency on `phoofflineeeganalysis` or `phopylslhelper`), so stream-type classification will be duplicated locally.

## Goal

1. **EEG plot** – one time-series panel with channels from the first EEG stream.
2. **MOTION plot** – one time-series panel with channels from the first MOTION stream.
3. **Text panel** – one Text panel with **all** Text Logger (and optionally EventBoard) streams **merged**: collect every (timestamp, text) from those streams, sort by time, and log all entries to a single entity so the viewer shows one combined log.

## Approach

1. **Classify streams by name** (inside the loader, no new deps). Use a local mapping aligned with the main pipeline, e.g.:
  - `"Epoc X"` → EEG  
  - `"Epoc X Motion"` → MOTION  
  - `"TextLogger"` and `"EventBoard"` → TEXT (for the merged Text panel)  
   Other names (e.g. `"Epoc X eQuality"`) can be treated as "other" and skipped or logged under `xdf/other/{name}`.
2. **One entity per type for the two time-series plots.**
  Log at most two time-series entities:
  - `{prefix}xdf/EEG` – first EEG stream’s channels (same format as now: `rr.send_columns` + `rr.Scalars.columns` + `rr.SeriesLines`).
  - `{prefix}xdf/MOTION` – first MOTION stream’s channels.
   If there are multiple EEG or MOTION streams, use the **first** of each (by iteration order).
3. **One entity for merged Text.**
  - Collect all streams classified as TEXT (TextLogger, EventBoard).
  - For each: read `time_stamps` and `time_series` (string/marker data; may need to handle list-of-lists or ravel to one string per sample). Build a list of `(t_sec, text)` using a common time base (e.g. global t0 = min of first timestamp across all streams, or use first stream’s t0 and offset others if needed for consistency with EEG/MOTION).
  - Sort the combined list by `t_sec`.
  - Log to a **single** entity path `{prefix}xdf/Text`: for each `(t_sec, text)` set the timeline to `t_sec` (duration since global t0), then `rr.log(path, rr.TextLog(text))`. Use the same timeline name as the numeric streams (e.g. `time_sec`) so the Text panel is aligned with the time-series views.
  - Result: one Text panel in the viewer showing all log entries from all Text Logger (and EventBoard) streams in time order.
4. **Other streams.** Streams that are neither EEG, MOTION, nor TEXT: skip or log under `xdf/other/...` (recommend skip for minimal scope).

## Implementation outline

- **File:** [rerun/rerun_loader_xdf.py](c:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/PhoOfflineEEGAnalysis/rerun/rerun_loader_xdf.py)
- **Stream classification:** Add `_stream_modality(stream) -> str | None)` using `_stream_name(stream)` and a local dict, e.g. `{"Epoc X": "EEG", "Epoc X Motion": "MOTION", "TextLogger": "TEXT", "EventBoard": "TEXT"}`.
- **Numeric streams (EEG / MOTION):**
  - In `_log_xdf_streams_imu_style` (or a refactored flow): group numeric streams by modality; log first EEG at `{prefix}xdf/EEG`, first MOTION at `{prefix}xdf/MOTION` with existing `send_columns` + `SeriesLines` logic. Use a **global t0** (e.g. minimum of first timestamps across streams) for `time_sec` so EEG, MOTION, and Text share the same timeline.
- **Text streams (merged):**
  - Add a helper e.g. `_log_xdf_text_streams_merged(streams, text_modality_names, entity_path_prefix, t0)` that:
    - Filters streams to those whose name is in `text_modality_names` (e.g. `["TextLogger", "EventBoard"]`).
    - For each such stream: ensure `time_series` is present and not numeric; get `time_stamps` and convert to relative seconds (e.g. `time_stamps - t0`). Build `(t_sec, text)` per sample (handle `time_series` as 1D string array or list; unwrap/ravel as in phoRerunTesting).
    - Concatenate all `(t_sec, text)` from all TEXT streams, sort by `t_sec`.
    - Log to a single path `{prefix}xdf/Text`: for each entry call `rr.set_time_sec(t_sec)` (or equivalent) then `rr.log(path, rr.TextLog(text))`. Use the same timeline key as the numeric streams so the Text panel is time-aligned.
  - Call this after logging EEG/MOTION, reusing the same `t0` (e.g. from the first numeric stream or from the earliest timestamp across all streams).
- **Paths:** `xdf/EEG`, `xdf/MOTION`, `xdf/Text` (each at most one entity). Other streams: skip.
- **README:** State that the loader shows EEG and MOTION on two separate time-series plots and all Text Logger (and EventBoard) streams merged in one Text panel; mention “first stream per type” for EEG/MOTION when multiple exist.

## Summary


| Item                  | Action                                                                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Stream classification | Add `_stream_modality(stream)` with EEG / MOTION / TEXT; map "TextLogger" and "EventBoard" to TEXT.                                      |
| EEG / MOTION          | Log first EEG at `xdf/EEG`, first MOTION at `xdf/MOTION`; shared `t0` for timeline.                                                      |
| Text panel            | Collect all TEXT streams; build merged (t_sec, text) sorted by time; log all to single entity `xdf/Text` with `rr.TextLog` at each time. |
| Other streams         | Skip (no `xdf/other/...`).                                                                                                               |
| Docs                  | README: two time-series plots + one merged Text panel.                                                                                   |


No new dependencies; no changes to `main_analyze_run.py` or `xdf_files.py`. All edits are in `rerun/rerun_loader_xdf.py` and `rerun/README.md`.