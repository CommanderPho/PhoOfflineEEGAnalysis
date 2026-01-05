---
name: Fix timeline track stream duration display
overview: Fix XDFStreamTrack to accept DataFrames (like other track classes) so that tracks created from DataFrames in TimelineWidget.add_tracks_from_xdf_streams() can properly display stream start/stop durations. The issue is that XDFStreamTrack only accepts BaseDatasource, causing tracks to fail creation when DataFrames are passed.
todos:
  - id: update-xdf-stream-track-init
    content: Update XDFStreamTrack.__init__() to accept both BaseDatasource and DataFrame, wrapping DataFrames in IntervalDataframeDatasource
    status: completed
  - id: ensure-interval-caching
    content: Ensure _cache_intervals() and update_display() are called after datasource setup in XDFStreamTrack
    status: completed
    dependencies:
      - update-xdf-stream-track-init
  - id: verify-display
    content: Verify that stream start/stop durations are now displayed correctly for all tracks
    status: completed
    dependencies:
      - ensure-interval-caching
---

# Fix Timeline Track Stream Duration Display

## Problem Analysis

After the recent refactoring (commit 8b40977), timeline tracks by default don't display their stream start/stop durations. Investigation reveals:

1. **Root Cause**: `XDFStreamTrack` only accepts `BaseDatasource` objects, but `TimelineWidget.add_tracks_from_xdf_streams()` passes DataFrames directly to track constructors.
2. **Impact**: When `XDFStreamTrack` is used as a fallback track class (when no specific mapping is found), it fails to create tracks because it receives a DataFrame instead of a `BaseDatasource`.
3. **Comparison**: Other track classes like `EEGRecordingTrack`, `MotionRecordingTrack`, and `VideoMetadataTrack` accept both DataFrames and `BaseDatasource` objects, wrapping DataFrames in `IntervalDataframeDatasource` when needed.

## Solution

Update `XDFStreamTrack.__init__()` to accept both DataFrames and `BaseDatasource` objects, following the same pattern used by other track classes:

1. Check if the input is a `BaseDatasource` - if so, use it directly
2. If it's a DataFrame, wrap it in an `IntervalDataframeDatasource` (like `EEGRecordingTrack` does)
3. Ensure intervals are cached and displayed properly

## Implementation

### File: `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/XDFStreamTrack.py`

**Changes needed:**

- Modify `__init__()` to accept both `BaseDatasource` and `pd.DataFrame`
- Add logic to wrap DataFrames in `IntervalDataframeDatasource` (similar to `EEGRecordingTrack`)
- Ensure `_cache_intervals()` is called after datasource setup
- Ensure `update_display()` is called to render intervals

**Pattern to follow** (from `EEGRecordingTrack.__init__()`):

```python
if isinstance(stream_source, BaseDatasource):
    self.set_datasource(stream_source)
    df = self._get_full_dataframe()
    self._stream_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
else:
    stream_df = stream_source
    self._stream_df = stream_df.copy()
    interval_ds = IntervalDataframeDatasource(self._stream_df, time_column_name='recording_datetime', datasource_name=name)
    self.set_datasource(interval_ds)

# Cache intervals immediately
self._cache_intervals()

# Initial display update (show all)
self.update_display()


```