---
name: Create Working Track Datasources
overview: Create working datasources for pyPhoTimeline that derive from BaseTrackDatasource for each modality (Video, EEG, Motion, PhoLog, Whisper, XDFStream, StringData), converting from the timeline track data formats to the BaseTrackDatasource interface.
todos:
  - id: create_modality_datasources_file
    content: Create modality_datasources.py file with all datasource implementations
    status: completed
  - id: implement_helper_utilities
    content: Implement helper functions for datetime conversion, duration parsing, and visualization properties
    status: completed
  - id: implement_video_datasource
    content: Implement VideoMetadataTrackDatasource with video interval handling
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: implement_eeg_datasource
    content: Implement EEGRecordingTrackDatasource with EEG interval handling
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: implement_motion_datasource
    content: Implement MotionRecordingTrackDatasource with motion interval handling
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: implement_pholog_datasource
    content: Implement PhoLogTrackDatasource with point marker support (0 duration)
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: implement_whisper_datasource
    content: Implement WhisperTrackDatasource with transcript interval handling
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: implement_xdf_datasource
    content: Implement XDFStreamTrackDatasource with flexible datetime column handling
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: implement_string_datasource
    content: Implement StringDataTrackDatasource as base class for string/comment data
    status: completed
    dependencies:
      - implement_helper_utilities
  - id: update_init_exports
    content: Update __init__.py to export all new datasources
    status: completed
    dependencies:
      - create_modality_datasources_file
---

# Create Working Track Datasources for pyPhoTimeline

## Overview

Create working datasource implementations for `pyPhoTimeline` that derive from `BaseTrackDatasource`. These will replace the non-working implementations in `PhoOfflineEEGAnalysis` timeline tracks. Each datasource will convert from the timeline track data formats (datetime-based) to the `BaseTrackDatasource` interface (float timestamp-based).

## Modalities to Implement

Based on the timeline tracks, create datasources for:

1. **VideoMetadataTrackDatasource** - Video recordings
2. **EEGRecordingTrackDatasource** - EEG recordings  
3. **MotionRecordingTrackDatasource** - Motion recordings
4. **PhoLogTrackDatasource** - PHO_LOG annotations
5. **WhisperTrackDatasource** - Whisper transcripts
6. **XDFStreamTrackDatasource** - Generic XDF streams
7. **StringDataTrackDatasource** - Generic string/comment data (base class)

## Implementation Details

### File Structure

Create a new file: `pypho_timeline/rendering/datasources/modality_datasources.py`This file will contain all modality-specific datasources.

### Common Pattern

Each datasource will:

1. Accept data in the format expected by the corresponding timeline track (DataFrame with datetime columns)
2. Convert datetime columns to float timestamps (`t_start`, `t_duration`)
3. Create visualization columns (`series_vertical_offset`, `series_height`, `pen`, `brush`)
4. Implement all required `BaseTrackDatasource` abstract methods
5. Provide appropriate detail renderers (or None for tracks without detail views)

### Key Conversions

- **Datetime to Timestamp**: Convert pandas datetime columns to float timestamps using `.timestamp()` or `.astype('datetime64[ns]').astype(np.float64) / 1e9`
- **Duration Handling**: Parse `duration_sec` (can be Timedelta, float, or string) to float seconds
- **Interval Filtering**: Filter intervals that overlap with time windows using `(t_start + t_duration >= new_start) & (t_start <= new_end)`

### Specific Implementations

#### 1. VideoMetadataTrackDatasource

- **Input**: DataFrame with `video_start_datetime`, `video_end_datetime` (or `video_start_datetime` + `video_duration`)
- **Output**: Intervals with `t_start`, `t_duration`
- **Detail Renderer**: `VideoThumbnailDetailRenderer` (if video frames available)
- **Color**: Blue theme (matching VideoMetadataTrack)

#### 2. EEGRecordingTrackDatasource

- **Input**: DataFrame with `recording_datetime`, `duration_sec` (or `duration_sec_check`)
- **Output**: Intervals with `t_start`, `t_duration`
- **Detail Renderer**: None (overview only, matching EEGRecordingTrack)
- **Color**: Green/blue theme (matching EEGRecordingTrack)

#### 3. MotionRecordingTrackDatasource

- **Input**: DataFrame with `recording_datetime`, `duration_sec`
- **Output**: Intervals with `t_start`, `t_duration`
- **Detail Renderer**: `GenericPlotDetailRenderer` or `PositionPlotDetailRenderer` (if position data available)
- **Color**: Orange/red theme (matching MotionRecordingTrack)

#### 4. PhoLogTrackDatasource

- **Input**: DataFrame with `onset` (or `time`), `duration` (optional, defaults to 0 for point markers)
- **Output**: Intervals with `t_start`, `t_duration` (0 duration for point markers)
- **Detail Renderer**: None (text rendering handled by track, not datasource)
- **Color**: Purple theme (matching PhoLogTrack)

#### 5. WhisperTrackDatasource

- **Input**: DataFrame with `onset`, `duration`
- **Output**: Intervals with `t_start`, `t_duration`
- **Detail Renderer**: None (overview only, matching WhisperTrack)
- **Color**: Cyan/teal theme (matching WhisperTrack)

#### 6. XDFStreamTrackDatasource

- **Input**: DataFrame with `recording_datetime` (or `first_timestamp_dt`), `duration_sec` (or `last_timestamp_dt`)
- **Output**: Intervals with `t_start`, `t_duration`
- **Detail Renderer**: None (overview only, matching XDFStreamTrack)
- **Color**: Gray theme (matching XDFStreamTrack)

#### 7. StringDataTrackDatasource

- **Input**: DataFrame with `onset`, `duration`
- **Output**: Intervals with `t_start`, `t_duration`
- **Detail Renderer**: None (base class, overview only)
- **Color**: Default (subclasses set their own)

### Required Methods Implementation

Each datasource must implement:

1. **`df` property**: Returns DataFrame with `t_start`, `t_duration`, and visualization columns
2. **`time_column_names` property**: Returns `['t_start', 't_duration', 't_end']`
3. **`total_df_start_end_times` property**: Returns `(min_t_start, max_t_end)` tuple
4. **`get_updated_data_window(new_start, new_end)`**: Returns overlapping intervals
5. **`fetch_detailed_data(interval)`**: Returns detailed data (DataFrame, dict, or None)
6. **`get_detail_renderer()`**: Returns `DetailRenderer` instance or None
7. **`update_visualization_properties(function)`**: Updates visualization columns

### Helper Utilities

Create helper functions for:

- Parsing duration to seconds (handle Timedelta, float, string)
- Converting datetime to timestamp
- Normalizing datetime columns to UTC-naive
- Creating default pens/brushes with colors and alpha

### Testing Considerations

Each datasource should:

- Handle empty DataFrames gracefully
- Handle missing columns with sensible defaults
- Convert datetime columns correctly (timezone-aware and naive)
- Filter intervals correctly for time windows
- Generate unique cache keys for intervals

## Files to Create

- `pypho_timeline/rendering/datasources/modality_datasources.py` - All modality datasources

## Files to Update

- `pypho_timeline/rendering/datasources/__init__.py` - Export new datasources

## Notes