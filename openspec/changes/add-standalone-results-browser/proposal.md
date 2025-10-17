## Why
Analysts need an easy way to share exploration outcomes without requiring Python or a live server. A self-contained browser-based artifact enables quick review, collaboration, and archival of EEG sessions, spectrograms, annotations, and derived metrics.

## What Changes
- Add a standalone results/data browser export that writes a single self-contained `.html` file.
- Support browsing multiple sessions and modalities (EEG, MOTION, transcripts) with filters and timelines.
- Embed datasets (xarray/netCDF/Zarr slices) and thumbnails with lazy loading where possible.
- Provide linked views: channel list, spectrograms, bandpower trends, annotations/events, and metadata panels.
- Add Python API and CLI command to generate the artifact from saved analysis outputs.
- Include lightweight theming and responsive layout for desktop and tablet.

- Define a four-panel layout:
  - Left sidebar: session pager (list + prev/next) for browsing sessions.
  - Right sidebar: display/plot options (channel toggles, plot types, overlays, color maps).
  - Bottom panel: comments viewer (from provided comments file) with quick filters and selection highlight.
  - Main panel: time-aligned timeline showing raw EEG for the active session, chosen result plots, and a synchronized comment track.

- Interactions and synchronization:
  - Zoom, pan, and brush in the main panel synchronize across raw EEG, result plots, and the comment track.
  - Selecting a comment in the bottom panel seeks the main timeline; clicking a comment marker focuses the corresponding comment.

## Impact
- Affected specs: `results-browser`
- Affected code: `src/phoofflineeeganalysis/analysis/UI/spectrogram_gui.py`, export utilities under `analysis/`, new `export_results_browser.py` module.
- No breaking changes expected; feature is additive.


