## 1. Implementation
- [ ] 1.1 Add exporter module `export_results_browser.py` with API function
- [ ] 1.2 Add CLI command `export-results-browser` wiring to API
- [ ] 1.3 Build Panel/Bokeh app layout with linked views
- [ ] 1.3.1 Compose four-panel layout (left sessions, right options, bottom comments, main timeline)
- [ ] 1.4 Implement data loaders for NetCDF/xarray/Zarr inputs
- [ ] 1.5 Implement downsampling/LOD for spectrograms and trends
- [ ] 1.6 Implement annotation/events and metadata panels
- [ ] 1.7 Implement multi-session selector and modality filter
- [ ] 1.8 Implement embedding strategy (inline small, reference large) and offline bundling
- [ ] 1.9 Save as standalone HTML and verify no CDN dependencies
- [ ] 1.10 Add unit/integration tests and notebook smoke test
- [ ] 1.11 Document API and CLI in README

## 2. UI Details
- [ ] 2.1 Left sidebar: session pager (list, search, prev/next)
- [ ] 2.2 Right sidebar: display options (channels, plot types, overlays, colormaps)
- [ ] 2.3 Bottom panel: comments viewer (load from file, filter, highlight selections)
- [ ] 2.4 Main panel: timeline with raw EEG, chosen result plots, and comment track
- [ ] 2.5 Cross-panel time sync (zoom/pan/brush/seek; comment ↔ timeline linkage)


