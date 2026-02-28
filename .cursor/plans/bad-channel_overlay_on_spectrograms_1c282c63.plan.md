---
name: Bad-channel overlay on spectrograms
overview: Add a subtle visual indication of per-channel BAD annotation periods on the HTML spectrogram exports (both per-session and combined) so bad segments are visible without obscuring the spectrogram.
todos: []
isProject: false
---

# Indicate bad-channel periods on output histograms (spectrograms)

## Context

- The "output histograms" are **HTML spectrograms** (time–frequency heatmaps) produced by `[export_session_spectrograms_html](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py)` (and the combined variant `[export_combined_spectrograms_html](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py)`).
- Each EEG channel gets one subplot: x = time (s), y = frequency (Hz), built with HoloViews/Bokeh via `da.hvplot.image(...)`.
- Bad segments are stored in **MNE `raw.annotations`**: descriptions starting with `BAD_` (e.g. `BAD_motion`, `BAD_JAW`, `BAD_peak`). Annotations can be **per-channel** via `annotations.ch_names` (list of lists; empty/None = all channels).
- Spectrogram time axis `t` from `[EEG_data.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\EEG_data.py)` is in the same time base as the raw (seconds from start), so annotation onset/duration align with `t`.

## Approach

1. **Helper: per-channel bad intervals**
  Add a small function (e.g. in `[main_analyze_run.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py)`) that, given an `mne.io.Raw` and the list of channel names to consider:
  - Iterate over `raw.annotations` and select segments where `description` starts with `"BAD_"` (case-insensitive).
  - Use `annotations.ch_names`: if `None` or the segment’s list is empty, treat as “all channels”; otherwise include the segment only for channels listed for that segment.
  - Return a dict `{ch_name: [(start, end), ...]}` with times in seconds (onset, onset+duration).
2. **Overlay on each channel plot**
  For each channel’s spectrogram image:
  - Get `bad_intervals = channel_bad_intervals.get(ch_name, [])`.
  - Clip/intersect each interval with the plot’s time range `[t.min(), t.max()]` to avoid drawing outside the visible range.
  - For each (start, end), create a **semi-transparent vertical band** (full frequency span for that subplot) using HoloViews:
    - Use `hv.Rect((start, f_min), (end, f_max))` (or equivalent) with `f_min`/`f_max` from the filtered frequency coords.
    - Style with **low alpha** (e.g. 0.25–0.4) and a distinct color (e.g. red or gray) so the spectrogram remains clearly visible (“don’t completely obscure it”).
  - Overlay rectangles on the image: `img * hv.Overlay(rects)` (or one `hv.Rectangles` if preferred), and append this overlay to `channel_plots`; if there are no bad intervals, keep appending `img` as today.
3. **Apply in both export paths**
  - `**export_session_spectrograms_html`** (loop starting ~line 344): Before the channel loop, compute `channel_bad_intervals` from `a_raw` and the keys of `spectogram_result_dict`. When building each channel’s plot, add the bad-interval overlay as above.
  - `**export_combined_spectrograms_html`** (loop starting ~line 459): Same logic so the combined HTML also shows bad segments consistently.

## Implementation details

- **HoloViews**: Use `hv.Rect((x0, y0), (x1, y1))` for each (start, end) with y0 = `f_filtered.min()`, y1 = `f_filtered.max()`. Apply `.opts(alpha=0.3, color='red', line_alpha=0)` (or similar) so the band is visible but transparent. Combine with `img * hv.Overlay(rect_list)` so the image is drawn first and rects on top.
- **Time alignment**: Use annotation onset/duration as-is; they are in the same time base as spectrogram `t`. Optionally clip intervals to `[t.min(), t.max()]` for cleanliness.
- **Edge cases**: If `raw.annotations` is None or empty, `channel_bad_intervals` is empty and behavior is unchanged. If `ch_names` for an annotation is missing or not a list, treat as “all channels” for that segment.

## Files to change

- **[main_analyze_run.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py)**  
  - Add helper `_get_channel_bad_intervals(raw, channel_names)` returning `Dict[str, List[Tuple[float, float]]]`.  
  - In `export_session_spectrograms_html`: compute bad intervals once per session; in the channel loop, add rect overlay when non-empty.  
  - In `export_combined_spectrograms_html`: same (compute once per session, overlay per channel).

No changes to `[EEG_data.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\EEG_data.py)` or to the spectrogram computation; only the HTML export and a small helper in `main_analyze_run.py` are modified.