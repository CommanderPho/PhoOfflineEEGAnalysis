---
name: DataFileMetadataParser Refactoring
overview: Factor out XDF and FIF file metadata caching from `historical_data.py` into a new `DataFileMetadataParser` class that inherits from `BaseFileMetadataParser`, providing disk-persisted caching similar to `VideoMetadataParser`.
todos:
  - id: create_data_file_metadata_parser
    content: Create new file `data_file_metadata.py` with `DataFileMetadataParser` class inheriting from `BaseFileMetadataParser`
    status: completed
  - id: implement_datetime_extraction
    content: Implement `extract_datetime_from_filename` method (reuse logic from HistoricalData)
    status: completed
    dependencies:
      - create_data_file_metadata_parser
  - id: implement_file_metadata_extraction
    content: Implement `extract_file_metadata` method to handle both .xdf (via LabRecorderXDF) and .fif (via read_raw) files, including duration extraction
    status: completed
    dependencies:
      - create_data_file_metadata_parser
  - id: implement_parse_data_folder
    content: Implement `parse_data_folder` method using base class `parse_filesystem_folder` with appropriate configuration
    status: completed
    dependencies:
      - implement_file_metadata_extraction
  - id: investigate_xdf_duration
    content: Investigate how to extract duration from XDF files without full file load (check LabRecorderXDF header/stream info)
    status: completed
    dependencies:
      - implement_file_metadata_extraction
  - id: refactor_historical_data
    content: Remove module-level cache and `_get_xdf_datetime_cached` method from historical_data.py
    status: completed
    dependencies:
      - implement_parse_data_folder
  - id: update_build_file_comparison_df
    content: Refactor `build_file_comparison_df` to use DataFileMetadataParser while maintaining backward compatibility
    status: completed
    dependencies:
      - refactor_historical_data
  - id: test_cache_persistence
    content: Test that cache persists to disk and loads correctly across sessions
    status: pending
    dependencies:
      - update_build_file_comparison_df
  - id: test_backward_compatibility
    content: Verify existing code using build_file_comparison_df still works with new implementation
    status: pending
    dependencies:
      - update_build_file_comparison_df
---

# Refactor XDF/FIF File Cache into DataFileMetadataParser

## Overview

Create a new `DataFileMetadataParser` class that inherits from `BaseFileMetadataParser` to handle `.xdf` and `.fif` file metadata extraction with disk-persisted caching. This will replace the current in-memory-only cache implementation in `historical_data.py`.

**Note**: `BaseFileMetadataParser` and `VideoMetadataParser` have been moved from `pyPhoTimeline` into `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/` as `file_metadata.py` and `video_metadata.py` respectively.

## Current Implementation Analysis

### Components to Factor Out:

1. **Module-level in-memory cache** (lines 43-47):

- `_xdf_datetime_cache: Dict[Tuple[str, float, int], datetime] = {}`
- `_xdf_cache_lock = threading.Lock()`
- Currently only persists within a single Python session

2. **`_get_xdf_datetime_cached` method** (lines 757-793):

- Extracts datetime from XDF files using `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file`
- Falls back to filename parsing if `file_datetime` is None
- Uses in-memory cache keyed by `(file_path, mtime, size)`
- Thread-safe with lock

3. **FIF file handling** (lines 844-846 in `build_file_comparison_df`):

- Uses `read_raw()` from MNE
- Extracts datetime via `get_or_parse_datetime_from_raw()`
- No caching currently

4. **`build_file_comparison_df` method** (lines 797-885):

- Processes both .xdf and .fif files
- Uses parallel processing with ThreadPoolExecutor
- Returns DataFrame with file metadata

5. **`get_or_parse_datetime_from_raw` method** (lines 123-146):

- Extracts datetime from MNE raw object's `meas_date`
- Falls back to filename parsing
- Optionally sets `meas_date` on the raw object

## Implementation Plan

### Step 1: Create New File

Create `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/data_file_metadata.py` with:

- Import `BaseFileMetadataParser` from `phoofflineeeganalysis.analysis.file_metadata`
- Import required dependencies (mne, LabRecorderXDF, etc.)

### Step 2: Implement DataFileMetadataParser Class

The class should inherit from `BaseFileMetadataParser` and implement:

1. **`extract_datetime_from_filename`** (override):

- Reuse existing logic from `HistoricalData.extract_datetime_from_filename`
- Same datetime parsing patterns

2. **`extract_file_metadata`** (override):

- Handle both `.xdf` and `.fif` file types
- For `.xdf`: Use `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file` (lightweight, no full load)
- For `.fif`: Use `read_raw(file, preload=False)` and extract from `raw.info['meas_date']`
- Extract duration:
- `.fif`: `raw.times[-1]` (last timestamp)
- `.xdf`: May need stream info or header metadata (TBD based on available data)
- Return dict with: `start_datetime`, `duration`, and any other metadata (file_size, channels, sampling_rate, etc.)
- Handle errors gracefully (return None)

3. **`get_file_metadata`** (inherit or override):

- Use base class implementation for file_size and file_mtime

4. **`is_file_changed`** (inherit):

- Use base class implementation

5. **`load_cache`** (override):

- Parse datetime columns: `start_datetime`, `end_datetime`
- Use base class pattern

6. **`save_cache`** (inherit):

- Use base class implementation

7. **`parse_data_folder`** (new method, similar to `parse_video_folder`):

- Use `parse_filesystem_folder` from base class
- Configure for data files:
- `cache_filename="_data_file_metadata_cache.csv"`
- `path_column="data_file_path"`
- `start_datetime_column="start_datetime"`
- `end_datetime_column="end_datetime"`
- `duration_metadata_key="duration"`
- `included_file_extensions=['.xdf', '.fif']`

### Step 3: Update historical_data.py

1. **Remove module-level cache** (lines 43-47):

- Delete `_xdf_datetime_cache` and `_xdf_cache_lock`

2. **Remove `_get_xdf_datetime_cached` method** (lines 757-793):

- Functionality moved to `DataFileMetadataParser.extract_file_metadata`

3. **Refactor `build_file_comparison_df`** (lines 797-885):

- Option A: Replace with call to `DataFileMetadataParser.parse_data_folder()` and adapt output format
- Option B: Keep method but use `DataFileMetadataParser` internally for metadata extraction
- Maintain parallel processing if needed (base class doesn't do parallel, but we can add it)
- Preserve existing return format for backward compatibility

4. **Update `get_or_parse_datetime_from_raw`** (lines 123-146):

- Keep as-is (still needed for other uses)
- Or consider moving to `DataFileMetadataParser` as a helper

5. **Update imports**:

- Add import for `DataFileMetadataParser`

### Step 4: Handle Duration Extraction

For `.fif` files:

- Duration = `raw.times[-1]` (last timestamp in seconds)

For `.xdf` files:

- May need to examine `LabRecorderXDF` to see if duration is available in header or requires stream processing
- If not easily available, may need to use a lightweight approach or mark as TBD

### Step 5: Testing Considerations

- Ensure backward compatibility with existing code using `build_file_comparison_df`
- Test cache persistence across sessions
- Test cache invalidation when files change
- Test both .xdf and .fif file types
- Test parallel processing if maintained

## Key Design Decisions

1. **Cache Location**: Follow `VideoMetadataParser` pattern - cache CSV in the same folder as data files
2. **Cache Key**: Use file path + file_size + file_mtime (same as base class)
3. **Thread Safety**: Base class doesn't use locks, but we may need to add if parallel processing is maintained
4. **Duration for XDF**: Need to investigate if available without full file load
5. **Backward Compatibility**: `build_file_comparison_df` should maintain its current interface

## Files to Modify

1. **New file**: `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/data_file_metadata.py`
2. **Modify**: `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/historical_data.py`
3. **Dependencies**: `BaseFileMetadataParser` is now available in `phoofflineeeganalysis.analysis.file_metadata` (moved from `pyPhoTimeline`)

## Open Questions

1. Should `build_file_comparison_df` be completely replaced or kept as a wrapper?
2. How to extract duration from XDF files without full load?
3. Should parallel processing be maintained in the new parser or rely on base class sequential processing?
4. What additional metadata should be extracted (channels, sampling rate, etc.)?