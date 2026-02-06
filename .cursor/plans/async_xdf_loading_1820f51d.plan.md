---
name: Async XDF Loading
overview: Convert synchronous XDF file loading in `build_from_xdf_files` to asynchronous background loading using Qt's QThreadPool, following the existing `AsyncDetailFetcher` pattern. This will prevent UI blocking during file loading.
todos:
  - id: create-xdf-worker
    content: Create XdfLoadWorker class (QRunnable) that loads a single XDF file in background thread and puts results in queue
    status: pending
  - id: modify-build-method
    content: Replace synchronous XDF loading loop in build_from_xdf_files with async implementation using QThreadPool
    status: pending
    dependencies:
      - create-xdf-worker
  - id: add-result-processing
    content: Add QTimer-based result processing to collect worker results on main thread while keeping UI responsive
    status: pending
    dependencies:
      - create-xdf-worker
  - id: add-progress-updates
    content: Add progress print statements showing loading status for each file
    status: pending
    dependencies:
      - modify-build-method
  - id: test-async-loading
    content: Test async loading with single file, multiple files, and error cases
    status: pending
    dependencies:
      - modify-build-method
      - add-result-processing
---

# Make XDF File Loading Asynchronous

## Problem

The XDF file loading in `build_from_xdf_files` (lines 162-167) is synchronous and blocks the UI thread, making it the slowest part of timeline building. The user expects this to be done asynchronously in the background.

## Solution

Implement asynchronous XDF loading using Qt's `QThreadPool` and `QRunnable`, following the existing async pattern in `AsyncDetailFetcher`.

## Implementation Plan

### 1. Create XDF Loading Worker Class

- **File**: `pyPhoTimeline/pypho_timeline/timeline_builder.py` (add new class)
- Create `XdfLoadWorker(QtCore.QRunnable)` class similar to `DetailFetchWorker`
- Worker will:
- Load a single XDF file using `pyxdf.load_xdf()` in background thread
- Put results in a thread-safe queue
- Handle cancellation and errors

### 2. Modify `build_from_xdf_files` Method

- **File**: `pyPhoTimeline/pypho_timeline/timeline_builder.py` (lines 122-203)
- Replace synchronous loop (lines 162-167) with:
- Create workers for each XDF file
- Submit all workers to `QThreadPool.globalInstance()`
- Use `QEventLoop` with `QTimer` to process results from queue while allowing Qt event loop to run
- Wait for all files to complete loading
- Collect results in correct order
- Show progress updates via print statements

### 3. Thread-Safe Result Collection

- Use a `queue.Queue` for worker results (file index, streams, file_header, error)
- Process queue on main thread using `QTimer` (similar to `AsyncDetailFetcher._process_result_queue`)
- Maintain order of results to match input file order

### 4. Error Handling

- Handle individual file load failures gracefully
- Continue loading other files even if one fails
- Report errors clearly to user

### 5. Progress Feedback

- Update print statements to show progress (e.g., "Loading XDF file 1/3: path/to/file.xdf")
- Show completion status for each file

## Key Design Decisions

1. **Use QEventLoop instead of blocking wait**: This allows Qt's event loop to process while waiting, keeping UI responsive
2. **Maintain file order**: Results are collected in the same order as input files
3. **Parallel loading**: Multiple files load simultaneously using thread pool
4. **Follow existing pattern**: Reuse the `AsyncDetailFetcher` architecture for consistency

## Files to Modify

- `pyPhoTimeline/pypho_timeline/timeline_builder.py`
- Add `XdfLoadWorker` class
- Modify `build_from_xdf_files` method to use async loading
- Add necessary imports (`queue`, `QtCore`)

## Testing Considerations

- Test with single file (backward compatibility)
- Test with multiple files
- Test with large files to verify UI remains responsive
- Test error handling (missing file, corrupted file)