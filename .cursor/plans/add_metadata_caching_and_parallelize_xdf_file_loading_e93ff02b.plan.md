---
name: Add metadata caching and parallelize XDF file loading
overview: Implement metadata caching for XDF files and parallelize the file processing in `build_file_comparison_df` using ThreadPoolExecutor with max 3 workers. The cache will store XDF datetime metadata keyed by file path + mtime/size for automatic invalidation.
todos: []
isProject: false
---

## Implementation Plan

### 1. Add Module-Level Cache and Thread Safety

**Location:** [`src/phoofflineeeganalysis/analysis/historical_data.py`](src/phoofflineeeganalysis/analysis/historical_data.py) after line 39

- Add module-level cache dictionary: `_xdf_datetime_cache: Dict[Tuple[str, float, int], datetime] = {}`
  - Cache key: `(file_path_str, mtime, size)` for automatic invalidation when files change
- Add threading lock: `_xdf_cache_lock = threading.Lock()` for thread-safe cache access
- Import `threading` and `concurrent.futures` at the top of the file

### 2. Create Cached Helper Method

**Location:** [`src/phoofflineeeganalysis/analysis/historical_data.py`](src/phoofflineeeganalysis/analysis/historical_data.py) before `build_file_comparison_df` (after line 740)

Add `_get_xdf_datetime_cached` classmethod:

- Takes `a_file: Path` as input
- Gets file stat (mtime, size) for cache key
- Checks cache first (with lock)
- If cache miss, calls `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file()` to load datetime
- Stores result in cache (with lock)
- Falls back to filename parsing if datetime is None
- Returns `datetime` object in UTC timezone

### 3. Create Parallel Processing Worker Function

**Location:** Inside `build_file_comparison_df` method

Add `_process_single_file` nested function:

- Takes `(file_idx, a_file)` tuple as input
- Handles both XDF and non-XDF files
- For XDF: uses `_get_xdf_datetime_cached()`
- For non-XDF: uses existing `read_raw()` logic
- Returns `(file_idx, result_dict)` or `(file_idx, None)` on error
- Result dict contains: `{'src_file_name', 'start_t', 'src_file', 'meas_datetime', **metadata_dict}`

### 4. Refactor build_file_comparison_df for Parallel Processing

**Location:** [`src/phoofflineeeganalysis/analysis/historical_data.py`](src/phoofflineeeganalysis/analysis/historical_data.py) lines 741-804

Changes:

- Replace sequential `for a_file in recording_files:` loop with parallel processing
- Use `ThreadPoolExecutor(max_workers=3)` context manager
- Submit all files using `executor.submit(_process_single_file, (idx, file))`
- Collect results using `as_completed()` to process as they finish
- Maintain result order using index-based result list: `results = [None] * len(recording_files)`
- Handle exceptions per file without stopping entire process
- Keep existing DataFrame construction and sorting logic unchanged

### 5. Error Handling

- Wrap each file processing in try-except within worker function
- Log errors per file: `print(f'failed to load file: "{a_file}" with error: {e}. Skipping.')`
- Continue processing other files even if one fails
- Return None for failed files, filter them out before DataFrame construction

### 6. Cache Invalidation Strategy

- Cache key includes `(file_path, mtime, size)` 
- On cache lookup, check if current file stats match cached key
- If stats differ, treat as cache miss and reload
- This ensures cache automatically invalidates when files are modified

## Implementation Details

### Thread Safety

- Use `threading.Lock()` for all cache read/write operations
- Cache structure: `Dict[Tuple[str, float, int], datetime] `where tuple is `(file_path, mtime, size)`

### Parallel Processing Pattern

Following the existing pattern in `main_analyze_run.py`:

```python
with ThreadPoolExecutor(max_workers=3) as executor:
    future_to_idx = {executor.submit(_process_single_file, (idx, file)): idx 
                     for idx, file in enumerate(recording_files)}
    results = [None] * len(recording_files)
    for future in as_completed(future_to_idx):
        idx, result = future.result()
        results[idx] = result
```

### Performance Benefits

- **Caching**: Avoids reloading XDF files that haven't changed (significant speedup for repeated calls)
- **Parallelization**: Processes up to 3 files simultaneously instead of sequentially
- **Combined**: For directories with many XDF files, expect 2-3x speedup on first run, 5-10x+ on subsequent runs with cache hits

## Files to Modify

- [`src/phoofflineeeganalysis/analysis/historical_data.py`](src/phoofflineeeganalysis/analysis/historical_data.py)
  - Add imports: `threading`, `concurrent.futures`
  - Add module-level cache and lock (after line 39)
  - Add `_get_xdf_datetime_cached` helper method (before line 741)
  - Refactor `build_file_comparison_df` method (lines 741-804)