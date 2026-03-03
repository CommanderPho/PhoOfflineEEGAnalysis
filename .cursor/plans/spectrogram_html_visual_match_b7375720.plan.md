---
name: Spectrogram HTML visual match
overview: "Update `export_session_spectrograms_html` in main_analyze_run.py so each channel spectrogram matches the preferred look: colormap and limits, axis labels, no colorbar, title in a white box with black border (with optional session_label), and light background/grid. Layout stays per-channel; no averaged spectrogram."
todos: []
isProject: false
---

# Spectrogram HTML Visual Match Plan

## Goal

Update [main_analyze_run.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py) `export_session_spectrograms_html` (lines 292–416) so the exported HTML spectrograms match the preferred appearance from your reference image, while keeping the existing per-channel stacked layout.

## Visual changes to apply


| Aspect                | Current                                                             | Target (from reference)                                                                                                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Colormap**          | `viridis`                                                           | Keep `viridis` (already yellow → green → blue/purple); optionally allow override via parameter.                                                                                                                                                               |
| **Colorbar**          | `colorbar=True`                                                     | `colorbar=False`.                                                                                                                                                                                                                                             |
| **Y-axis label**      | `'Frequency (Hz)'`                                                  | `'Frequency (Hz)'` per channel (reference uses “All N Channels Average” for a single plot; for per-channel we keep “Frequency (Hz)”).                                                                                                                         |
| **X-axis label**      | `'Time (s)'`                                                        | Remove or empty (no visible x-axis label in reference).                                                                                                                                                                                                       |
| **Title format**      | `session_title` / `ch_name`                                         | Session-level: `YYYY-MM-DD/HH-MM-SS` plus optional `- {session_label}` (e.g. `2025-11-11/22-22-22 - cog_bad`). Per-plot title in a **white box with thin black border** (top-left).                                                                           |
| **Title content**     | Per plot: channel name; layout: “EEG Spectrogram - {session_title}” | Use the new session title string (date/time + optional label) for the overall layout/title; per-subplot titles can remain channel name in the same box style, or be omitted if layout title is sufficient (preference: one clear session title in box style). |
| **Background / grid** | Default                                                             | Light blue/grey plot background and horizontal grid lines (subtle).                                                                                                                                                                                           |


## Implementation approach

1. **Function signature**
  Add optional parameter, e.g. `session_label: Optional[str] = None`. Keep all other parameters unchanged.
2. **Session title string**
  Build a session title for the session (and for any title box):
  - Date/time in reference format: `meas_date.strftime('%Y-%m-%d/%H-%M-%S')` (slash, not space).
  - If `session_label` is provided, append `f" - {session_label}"`.
3. **Per-channel `hvplot.image` options**
  - Set `colorbar=False`.  
  - Set `xlabel=''` (or omit so no x-axis label).  
  - Keep `ylabel='Frequency (Hz)'`.  
  - Use the same `cmap='viridis'` and keep `clim=(Sxx_filtered.min(), Sxx_filtered.max())` unless you prefer a shared clim across channels (optional later).  
  - Per-subplot title: either the new session title string (so every row shows the same session title in the box) or channel name; recommend **session title in box** for consistency with reference, with channel name in axis or a small subtitle if needed.
4. **Title box styling (white background, black border)**
  HoloViews/Bokeh does not expose title box styling directly in hvplot; use one of:
  - **Preferred:** After building the layout, use a **hook** when saving: get the Bokeh figure via `hv.render()` (or the plot handle from the layout), then set on each subplot’s title:
    - `plot.title.background_fill_color = 'white'`
    - `plot.title.border_line_color = 'black'`
    - `plot.title.border_line_width = 1`
  - Apply the same in a **backend_opts** dict if HoloViews supports `plot.title.`* (e.g. `backend_opts={"plot.title.background_fill_color": "white", ...}`) for the Image elements or the layout.  
   Use whichever is reliable in your HoloViews/Bokeh version (hook is most portable).
5. **Layout title**
  Set the overall layout title to the new session title string (date/time + optional label) so the page has one clear session identifier. Ensure the per-plot title (in box) matches or duplicates this as in the reference.
6. **Background and grid**
  Use Bokeh plot options so the plot area has a light blue/grey background and horizontal grid lines:
  - `plot.background_fill_color` to a light grey/blue (e.g. `#e8eef2` or similar).
  - Enable y-axis grid and set grid line color/alpha to a subtle shade (e.g. same as background but slightly darker).  
   Apply via the same hook or `backend_opts` used for the title (e.g. `plot.background_fill_color`, `plot.ygrid.grid_line_color` / `plot.ygrid.grid_line_alpha`).
7. **Bad intervals**
  No change: keep drawing red transparent rectangles over bad intervals as today.
8. **Docstring**
  Update the docstring to document `session_label` and the new title format.

## Files to modify

- [main_analyze_run.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis\main_analyze_run.py): only `export_session_spectrograms_html` and its docstring.

## Call sites

- If any call site has a “cog_bad”-style label available (e.g. from pipeline config or metadata), pass it as `session_label` when calling `export_session_spectrograms_html`. No change required for call sites that don’t have a label.

## Testing

- Run the existing pipeline that produces spectrogram HTML and confirm:
  - No colorbar, “Frequency (Hz)” on y-axis, no x-axis label.
  - Session title in form `YYYY-MM-DD/HH-MM-SS` or `YYYY-MM-DD/HH-MM-SS - cog_bad` in a white box with black border.
  - Light background and horizontal grid.
  - Bad intervals still visible when present.

