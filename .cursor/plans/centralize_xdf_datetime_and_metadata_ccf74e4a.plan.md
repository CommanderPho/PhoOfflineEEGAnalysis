---
name: Centralize XDF datetime and metadata
overview: Analyze duplicate datetime/parsing logic across PhoOfflineEEGAnalysis (historical_data.py, xdf_files.py) and pyPhoTimeline (BaseFileMetadataParser), then recommend whether to introduce an XDFMetadataParser subclass and how to centralize without breaking existing call sites.
todos: []
isProject: false
---

# Centralized XDF / datetime parsing: analysis and options

## Current implementations

### 1. pyPhoTimeline – BaseFileMetadataParser ([file_metadata.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\file_metadata.py))

- **extract_datetime_from_filename(filename)** – Returns `Optional[datetime]`. Same regex and format list as HistoricalData; **returns None** if no match (caller skips file).
- **parse_filesystem_folder(...)** – Folder loop: datetime from **filename only**, then `extract_file_metadata(file_path)` for type-specific metadata (must include `duration_metadata_key`). Builds DataFrame with path_column, start_datetime_column, end_datetime_column, cache columns. Caching by path, size, mtime.
- **extract_file_metadata(file_path)** – Base returns `None`; subclasses (e.g. VideoMetadataParser) override to return a dict including duration.

### 2. PhoOfflineEEGAnalysis – HistoricalData ([historical_data.py](c:\Users\pho\repos\EmotivEpoc\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\historical_data.py))

- **extract_datetime_from_filename(filename)** – Same regex/formats as base but **raises ValueError** if no match. Used for .fif, .csv, .xdf **filenames**.
- **get_or_parse_datetime_from_raw(raw, override_filepath, ...)** – MNE-specific: uses `raw.info['meas_date']` or falls back to filename; can set `raw.info` from parsed filename.
- **_get_xdf_datetime_cached(a_file)** – **XDF only**: calls `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file(a_file)` to get **file_datetime from XDF header**; fallback to `extract_datetime_from_filename(a_file.name)`. Module-level cache keyed by `(path, mtime, size)`.
- **build_file_comparison_df(recording_files, max_workers)** – Mixed FIF/EDF/XDF: for .xdf uses `_get_xdf_datetime_cached`; for others uses `read_raw` + `get_or_parse_datetime_from_raw`. Returns DataFrame with **different schema**: `src_file_name`, `start_t`, `src_file`, `meas_datetime`, `ctime`, `size`, `mtime`. Used for comparison/sorting, not “folder metadata table” like VideoMetadataParser.

### 3. PhoOfflineEEGAnalysis – LabRecorderXDF ([xdf_files.py](c:\Users\pho\repos\EmotivEpoc\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\xdf_files.py))

- **file_datetime** – Set in `init_basic_from_lab_recorder_xdf_file`: from **header['info']['datetime'][0]** (e.g. `'2025-09-11T17:04:20-0400'`), then converted to UTC. **Authoritative source is inside the file**, not the filename.
- **load_and_process_all(lab_recorder_output_path, ...)** – Globs `*.xdf`, optionally filters by `included_xdf_file_names`, then for each file calls `init_from_lab_recorder_xdf_file` (full load). No “lightweight folder metadata DataFrame” like VideoMetadataParser.

### 4. PhoOfflineEEGAnalysis – video_metadata ([video_metadata.py](c:\Users\pho\repos\EmotivEpoc\PhoOfflineEEGAnalysis\src\phoofflineeeganalysis\analysis\video_metadata.py))

- Standalone **copy** of VideoMetadataParser (same logic as pyPhoTimeline’s refactored VideoMetadataParser). Does **not** use `BaseFileMetadataParser` from pyPhoTimeline.

---

## Tension: where does “start datetime” come from?


| Context                                        | Source of start datetime                                              |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| BaseFileMetadataParser.parse_filesystem_folder | **Filename only** (extract_datetime_from_filename)                    |
| XDF (LabRecorderXDF)                           | **File header** (header['info']['datetime'][0]); filename is fallback |
| HistoricalData.build_file_comparison_df (XDF)  | **Header** via _get_xdf_datetime_cached, then filename fallback       |


So for XDF, the canonical start time is **in the file**. A subclass that only overrides `extract_file_metadata` would still have the base use **filename** for start in `parse_filesystem_folder`. To centralize properly you either:

- Extend the base so that when `extract_file_metadata` returns a start datetime (e.g. `start_datetime` key), the folder parser uses it instead of filename, or  
- Override `parse_filesystem_folder` in the XDF subclass so that after calling `extract_file_metadata` you build the row using **header datetime when present**, filename as fallback.

---

## Options

### Option A: XDFMetadataParser in PhoOfflineEEGAnalysis (subclass of BaseFileMetadataParser)

- **Dependency**: PhoOfflineEEGAnalysis already depends on `py-pho-timeline`; use `from pypho_timeline.utils.file_metadata import BaseFileMetadataParser`.
- **New class** (e.g. in `xdf_metadata.py` or alongside `xdf_files.py`):
  - **extract_datetime_from_filename** – Override to support LabRecorder-style names (optional fractional seconds / Z); keep `Optional[datetime]` (return None if no match) so base folder logic can skip bad names.
  - **extract_file_metadata(file_path)** – Lightweight XDF read: call `pyxdf.load_xdf` (or only read header if the API allows) to get `header['info']['datetime'][0]` and, if available, first/last timestamp from streams/footer for duration. Return dict with e.g. `xdf_start_datetime`, `xdf_duration_seconds` (0 if not computable), `xdf_file_path`, etc. Use filename parsing as fallback when header datetime is missing.
  - **parse_filesystem_folder** – Override so that for each file you call `extract_file_metadata` first; if the returned dict contains a start datetime (e.g. `xdf_start_datetime`), use it for the row instead of `extract_datetime_from_filename`. Otherwise use base behavior (filename then extract_file_metadata). Column names: e.g. `xdf_file_path`, `xdf_start_datetime`, `xdf_end_datetime`, `xdf_duration_seconds`, cache columns.
  - **parse_xdf_folder(folder_path, xdf_extensions=[".xdf"], ...)** – Thin wrapper calling `parse_filesystem_folder` with `cache_filename="_xdf_metadata_cache.csv"`, `path_column="xdf_file_path"`, `start_datetime_column="xdf_start_datetime"`, `end_datetime_column="xdf_end_datetime"`, `duration_metadata_key="xdf_duration_seconds"`.
- **HistoricalData**:
  - **_get_xdf_datetime_cached** – Delegate to XDFMetadataParser: e.g. a classmethod `get_xdf_datetime(file_path)` that does header + filename fallback and returns datetime (with optional internal caching), so you don’t duplicate header parsing.
  - **extract_datetime_from_filename** – Can stay as-is (raise on no match) for FIF/CSV; or call shared implementation (e.g. base or a small helper) and wrap with “raise if None” for backward compatibility.
- **Pros**: One place for “XDF folder → metadata DataFrame” and for “XDF file → start datetime”; aligns with VideoMetadataParser pattern; caching in base. **Cons**: Need to implement lightweight header-only (or minimal) XDF read for duration if desired; override of parse_filesystem_folder required for header-first datetime.

### Option B: Only centralize “datetime from filename” (no XDF folder parser)

- **Shared datetime parsing**: Use BaseFileMetadataParser.extract_datetime_from_filename (or a small shared helper in pyPhoTimeline) as the single implementation of regex + formats.
- **HistoricalData**: Call that shared implementation; wrap with “if None then raise ValueError” to keep current “strict” behavior for FIF/CSV/XDF filenames.
- **XDF**: Keep _get_xdf_datetime_cached and LabRecorderXDF as-is (header + filename fallback). No new XDFMetadataParser; no folder DataFrame for XDF from the base.
- **Pros**: Minimal change; no new class; no dependency of XDF logic on folder-parsing design. **Cons**: No single “parse XDF folder” API; build_file_comparison_df and load_and_process_all stay as the only ways to get XDF file lists/datetimes.

### Option C: XDFMetadataParser only for “single-file” datetime (no folder DataFrame)

- New class (can subclass BaseFileMetadataParser or be standalone) that provides:
  - **extract_datetime_from_filename** – For XDF-style names; Optional[datetime].
  - **get_xdf_datetime(file_path)** – Header (via init_basic_from_lab_recorder_xdf_file or a slimmer header-only read) + filename fallback; return datetime; optional caching.
- HistoricalData._get_xdf_datetime_cached and build_file_comparison_df call this. No `parse_xdf_folder`; no reuse of parse_filesystem_folder for XDF.
- **Pros**: Centralizes “XDF start datetime” only; no need to touch base or implement duration/end for XDF. **Cons**: No unified “folder metadata DataFrame” for XDF; two patterns (video folder vs XDF folder).

---

## Recommendation

- **If the goal is a single “track parsing / datetime” story and you want XDF to have the same shape as video (folder → DataFrame with start/end/duration/path and caching):** use **Option A** (XDFMetadataParser subclass, with override so start datetime comes from header when present).
- **If the goal is only to remove duplicate datetime-from-filename and XDF datetime logic:** **Option B** (centralize filename parsing) plus **Option C** (single-file XDF datetime helper used by _get_xdf_datetime_cached) gives the least structural change while still centralizing.

**Suggested next steps if choosing Option A**

1. Add **XDFMetadataParser** in PhoOfflineEEGAnalysis (new module or in `xdf_files.py`), subclassing `BaseFileMetadataParser` from pyPhoTimeline.
2. Implement **extract_file_metadata** to do a minimal XDF load (reuse `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file` or a header-only path), return dict with `xdf_start_datetime`, `xdf_duration_seconds` (from stream footer first/last if available), and any other columns you want.
3. Override **parse_filesystem_folder** (or add a small hook in base) so the row uses start datetime from that dict when present, else filename.
4. Add **parse_xdf_folder** that calls `parse_filesystem_folder` with XDF column names and cache filename.
5. Refactor **HistoricalData._get_xdf_datetime_cached** to use a method on XDFMetadataParser (or a shared helper) that returns datetime from header + filename fallback, keeping the same (path, mtime, size) cache key behavior.
6. Optionally, make **HistoricalData.extract_datetime_from_filename** call the base implementation and raise if None, so regex/formats live in one place.

**Video in PhoOfflineEEGAnalysis:** You can later switch the local `VideoMetadataParser` copy to inherit from `BaseFileMetadataParser` and delegate to the base (like pyPhoTimeline’s VideoMetadataParser) so video and XDF both go through the same base pattern.