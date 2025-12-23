---
name: Convert Tracks to Use Datasources
overview: Refactor all track widgets to use datasources from the datasource module instead of directly accessing DataFrames. This will provide flexibility for different data sources (DataFrames, XDF files, etc.) and enable features like lazy loading and windowed data access.
todos:
  - id: base_track_datasource
    content: "Add datasource support to BaseTrackWidget: _datasource property, _ensure_datasource() method, _get_full_dataframe() helper, and connect to source_data_changed_signal"
    status: completed
  - id: eeg_track_datasource
    content: Update EEGRecordingTrack to accept datasource or DataFrame, use datasource in _get_recording_intervals_vectorized()
    status: completed
    dependencies:
      - base_track_datasource
  - id: motion_track_datasource
    content: Update MotionRecordingTrack to accept datasource or DataFrame, use datasource in _get_recording_intervals_vectorized() and consider for _load_motion_timeseries()
    status: completed
    dependencies:
      - base_track_datasource
  - id: xdf_track_datasource
    content: Update XDFStreamTrack to accept datasource or DataFrame, use datasource in _get_recording_intervals_vectorized()
    status: completed
    dependencies:
      - base_track_datasource
  - id: video_track_datasource
    content: Update VideoMetadataTrack to accept datasource or DataFrame, use datasource in _get_recording_intervals_vectorized()
    status: completed
    dependencies:
      - base_track_datasource
  - id: string_track_datasource
    content: Update StringDataTrack to accept datasource or DataFrame, use datasource in _get_recording_intervals_vectorized()
    status: completed
    dependencies:
      - base_track_datasource
  - id: timeline_widget_review
    content: Review TimelineWidget.add_tracks_from_xdf_streams() to determine if datasources should be created there
    status: completed
    dependencies:
      - base_track_datasource
---

# Convert Timeline Tracks to Use Datasource System

## Overview

Currently, all track widgets (`EEGRecordingTrack`, `MotionRecordingTrack`, `XDFStreamTrack`, `VideoMetadataTrack`, `StringDataTrack`) directly store and access DataFrames (e.g., `self.motion_df`, `self.eeg_df`). This plan converts them to use the datasource abstraction layer (`BaseDatasource`, `DataframeDatasource`, `XDFDatasource`) for more flexible data access.

## Current Architecture

- Tracks store DataFrames directly: `self.motion_df`, `self.eeg_df`, `self.stream_df`, `self.video_df`, `self._df`
- Data access happens in `_get_recording_intervals_vectorized()` and `_get_metadata_for_interval()` methods
- Datasource system exists but is unused by tracks

## Target Architecture

- Tracks accept either a `BaseDatasource` or a DataFrame (for backward compatibility)
- If DataFrame is provided, automatically wrap it in `DataframeDatasource`
- Tracks use datasource methods: `get_updated_data_window()`, `total_datasource_start_end_times`
- Data access is abstracted through the datasource interface

## Implementation Plan

### 1. Update BaseTrackWidget

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/BaseTrackWidget.py`

- Add `_datasource: Optional[BaseDatasource]` property
- Add method `_ensure_datasource(df_or_datasource)` that:
- If `BaseDatasource` instance: use directly
- If DataFrame: wrap in `DataframeDatasource`
- Store in `self._datasource`
- Add helper method `_get_full_dataframe()` that returns the full DataFrame from datasource (for interval extraction)
- Connect to datasource's `source_data_changed_signal` to trigger updates

### 2. Update EEGRecordingTrack

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/EEGRecordingTrack.py`

- Modify `__init__()` to accept `eeg_df_or_datasource` (Union[pd.DataFrame, BaseDatasource])
- Call `_ensure_datasource()` in constructor
- Update `_get_recording_intervals_vectorized()` to:
- Get full DataFrame via `_get_full_dataframe()` or datasource's `df` property
- Use datasource's time column name if available
- Keep `_display_df` for metadata access (populated from datasource data)

### 3. Update MotionRecordingTrack

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/MotionRecordingTrack.py`

- Modify `__init__()` to accept `motion_df_or_datasource` (Union[pd.DataFrame, BaseDatasource])
- Call `_ensure_datasource()` in constructor
- Update `_get_recording_intervals_vectorized()` to use datasource
- Update `_load_motion_timeseries()` to potentially use datasource's `get_updated_data_window()` for windowed data access

### 4. Update XDFStreamTrack

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/XDFStreamTrack.py`

- Modify `__init__()` to accept `stream_df_or_datasource` (Union[pd.DataFrame, BaseDatasource])
- Call `_ensure_datasource()` in constructor
- Update `_get_recording_intervals_vectorized()` to use datasource
- Consider using `XDFDatasource` when XDF file path is available

### 5. Update VideoMetadataTrack

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/VideoMetadataTrack.py`

- Modify `__init__()` to accept `video_df_or_datasource` (Union[pd.DataFrame, BaseDatasource])
- Call `_ensure_datasource()` in constructor
- Update `_get_recording_intervals_vectorized()` to use datasource

### 6. Update StringDataTrack

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/tracks/StringDataTrack.py`

- Modify `__init__()` to accept `df_or_datasource` (Union[pd.DataFrame, BaseDatasource])
- Call `_ensure_datasource()` in constructor
- Update `_get_recording_intervals_vectorized()` to use datasource

### 7. Update TimelineWidget (if needed)

**File:** `src/phoofflineeeganalysis/analysis/UI/timeline/TimelineWidget.py`

- Review `add_tracks_from_xdf_streams()` to see if it should create datasources
- Consider creating `XDFDatasource` instances when appropriate
- Maintain backward compatibility with DataFrame-based track creation

## Data Access Pattern Changes

### Before:

```python
def _get_recording_intervals_vectorized(self):
    df = self.motion_df.copy()  # Direct DataFrame access
    start_dt = df['recording_datetime']
    # ... process DataFrame directly
```



### After:

```python
def _get_recording_intervals_vectorized(self):
    df = self._get_full_dataframe()  # Get from datasource
    start_dt = df[self._datasource.time_column_name]  # Use datasource's time column
    # ... process DataFrame from datasource
```



## Backward Compatibility

- All track constructors will accept DataFrames (existing code continues to work)
- DataFrames are automatically wrapped in `DataframeDatasource`
- No breaking changes to existing API

## Benefits

1. **Flexibility**: Tracks can work with any datasource type (DataFrame, XDF, future sources)
2. **Lazy Loading**: Datasources can implement lazy loading for large datasets
3. **Windowed Access**: Use `get_updated_data_window()` for efficient time-range queries
4. **Unified Interface**: Consistent data access pattern across all tracks
5. **Future Extensibility**: Easy to add new datasource types without changing tracks

## Testing Considerations

- Verify existing code using DataFrame constructors still works