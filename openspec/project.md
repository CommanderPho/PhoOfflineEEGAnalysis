# Project Context

## Purpose
Offline EEG analysis toolkit and utilities for Emotiv Epoc/Epoc X recordings. The project focuses on:
- Loading raw EEG and auxiliary modalities (MOTION, text logs, Whisper transcripts) from MNE FIF and LabRecorder XDF
- Cleaning and annotating data (motion artefacts, jaw clench, event logs)
- Spectral analysis (PSD, bandpower, spectrograms) and trend exploration
- Merging/aligning multi-session data by absolute time and exporting to interoperable formats (FIF, EDF+, MAT, HDF5)
- Interactive visualization via PyQt and browser-based dashboards

## Tech Stack
- Python (>=3.9,<3.13), Hatchling packaging
- Core scientific: NumPy, SciPy, pandas, scikit-learn
- EEG/MEG: MNE, mne-qt-browser, autoreject, mne-connectivity, mne-bids
- Streaming/IO: pylsl, mne-lsl, pyedflib (EDF+), pyxdf (XDF), h5py, tables, netCDF4, xarray, zarr
- Spectra/metrics: fooof, hmmlearn, tsdownsample
- Viz/UI: Matplotlib, mplcursors, Plotly, HoloViews + hvPlot + Panel + Bokeh, JupyterLab/ipywidgets, Napari (optional)
- Data engineering: Dask, Coiled
- Utilities: lxml[html-clean], python-benedict, dill, Pillow, Selenium, PhantomJS
- App/CLI: `phoofflineeeganalysis:main` entrypoint; PyQt5/qtpy GUI components

## Project Conventions

### Code Style
- PEP 8 with type hints for public APIs; prefer explicit return and argument types
- Descriptive names (no 1–2 letter vars); avoid deep nesting; prefer early returns
- Guard clauses over broad try/except; only catch and handle known error cases
- Maintain existing indentation style per file; new Python code uses 4-space indents
- Keep module-level functions pure when possible; isolate IO/side effects
- Docstrings: short summary line + key params/returns; include units where relevant (e.g., Hz, seconds)

### Architecture Patterns
- Package root: `src/phoofflineeeganalysis`
  - `analysis/` domain modules
    - `SavedSessionsProcessor`: orchestrates multi-modality discovery, preprocessing, and batch post-processing; exposes export helpers (EDF/FIF/MAT/HDF5)
    - `MNE_helpers`: MNE utilities (datetime alignment, annotation merging, dataframe conversions, Raw wrappers)
    - `EEG_data`, `motion_data`, `event_data`, `historical_data`, `flutter_data`: modality-specific loaders and preprocessors
    - `UI/`: PyQt5 widgets; `spectrogram_gui.py` for interactive spectrogram exploration; `CustomCalendarWidget.py`
  - `EegProcessing.py`: spectral transforms, PSD/bandpower utilities, artefact detection
  - `EegVisualization.py`: Matplotlib helpers (linear/circular plots, heatmaps)
  - `helpers/`: small utilities (e.g., indexing)
  - `resources/`: electrode layout TSVs and static assets
- Time handling: Prefer absolute times via `raw.info['meas_date']`; convert relative `time`/`onset` columns to absolute datetimes for alignment; ensure annotation `orig_time` is respected or normalized
- Filenaming: FIF files typically `YYYYMMDD-HHMMSS-<Stream>-raw.fif`; helper extracts datetime from filename when metadata missing
- Data model: `DataModalityType` enumerates modalities; `SessionModality` encapsulates per-modality dataframes, datasets, and analysis products

### Testing Strategy
- Unit tests (pytest) for helpers: datetime parsing, annotation merging, PSD/bandpower functions (deterministic seeds)
- Integration tests on small representative FIF/XDF samples (redacted/minimized) to validate end-to-end preprocessing and exports
- Notebook smoke tests for visualization components (nbconvert headless execution) and Panel/Bokeh/HoloViews rendering
- Golden-file checks for exported formats (EDF header fields, MAT struct keys, HDF5 group paths)
- CI can skip large-data tests by default; run extended tests behind flag

### Git Workflow
- Main branch is stable; feature branches per change
- Verb-led, kebab-case OpenSpec change IDs referenced in commits when applicable (e.g., `add-labrecorder-xdf-export`)
- Conventional, present-tense commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Large data and generated artifacts are not checked in; prefer references or small fixtures under `examples_jupyter/`

## Domain Context
- Device: Emotiv Epoc/Epoc X; typical EEG sampling ~128 Hz; multi-stream recordings via LabRecorder (EEG, motion, text logs)
- Modalities: `EEG`, `MOTION`, `PHO_LOG_TO_LSL` (text logger), `WHISPER` (transcripts)
- Preprocessing: filtering (1–40 Hz typical), artefact detection (jaw clench, motion), fixed-length epochs for bandpower
- Alignment: annotations may have differing `orig_time`; utilities reconcile to Raw meas time or normalize to None-based relative for safe concatenation
- Exports: EDF+ (for viewer compatibility), MAT (FieldTrip-style), FIF (canonical), HDF5 (analysis results, raw frames, flattened annotations)
- Visualization: interactive spectrograms and dashboards; saved HTML artifacts under `examples_jupyter/`

## Important Constraints
- Python version constraint: `>=3.9,<3.13`
- Large recordings; memory-aware processing preferred; batch/parallel operations use ThreadPool where safe
- MNE concatenation caveat: only first file's measurement info persists; avoid saving concatenated raws unless accepted trade-offs
- Windows paths are common in local workflows; keep path handling cross-platform and avoid hard-coded drive letters in committed code
- Some analysis code references `statsmodels` and `yasa` at runtime; ensure environment includes them when those paths are exercised

## External Dependencies
- File formats and tools: MNE FIF, LabRecorder XDF, EDF+, FieldTrip MAT, HDF5, NetCDF, Zarr
- Libraries/services:
  - MNE ecosystem (MNE, mne-lsl, mne-qt-browser)
  - PyQt5/qtpy for desktop UI
  - HoloViews/hvPlot/Panel/Bokeh, Plotly for web viz
  - Coiled/Dask for scalable compute (optional)
  - pyedflib/pyxdf for IO; lxml for HTML cleaning
- Data locations (developer local): `E:/Dropbox (Personal)/Databases/...` — treat as example paths; parameterize in code and configs

