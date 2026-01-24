---
name: TimelineBuilder from EEG Raw and Stream Info
overview: Implement a TimelineBuilder method that creates timeline tracks from MNE Raw objects and XDF stream info DataFrame, with proper datetime alignment for all data types (EEG, Motion, Annotations).
todos:
  - id: "1"
    content: Add build_from_eeg_raw_and_stream_info method to TimelineBuilder class
    status: completed
  - id: "2"
    content: Implement _extract_datasources_from_eeg_raw helper method to extract data from MNE Raw objects
    status: completed
  - id: "3"
    content: Implement datetime alignment logic using reference_datetime from stream_infos_df
    status: completed
  - id: "4"
    content: Create EEGTrackDatasource instances from EEG channel data
    status: completed
  - id: "5"
    content: Create MotionTrackDatasource instances if motion channels are present
    status: completed
  - id: "6"
    content: Create annotation/event datasources from Raw annotations
    status: completed
  - id: "7"
    content: "Handle edge cases: missing indices, empty Raw objects, missing datetime columns"
    status: completed
isProject: false
---

# TimelineBuilder Implementation for EEG Raw and Stream Info

## Overview

Implement `build_from_eeg_raw_and_stream_info` method in `TimelineBuilder` that takes MNE Raw objects and XDF stream info DataFrame, extracts all available data types (EEG, Motion, Annotations), and creates properly datetime-aligned timeline tracks.

## Key Components

### 1. New Method: `build_from_eeg_raw_and_stream_info`

**Location**: `pypho_timeline/timeline_builder.py`

**Signature**:

```python
def build_from_eeg_raw_and_stream_info(
    self,
    eeg_raws: List[mne.io.BaseRaw],
    stream_infos_df: pd.DataFrame,
    window_duration: Optional[float] = None,
    window_start_time: Optional[float] = None,
    add_example_tracks: bool = False,
    window_title: Optional[str] = None,
    window_size: Tuple[int, int] = (1000, 800)
) -> Optional[SimpleTimelineWidget]
```

**Key Steps**:

1. Extract reference datetime from `stream_infos_df` (use earliest `recording_datetime` or `first_timestamp_dt`)
2. Match Raw objects to stream info rows using `xdf_dataset_idx` column
3. For each matched pair:

   - Extract EEG channel data and timestamps
   - Extract Motion data (if present in Raw object)
   - Extract Annotations/Events
   - Convert timestamps to relative seconds from reference datetime
   - Create appropriate datasources (EEGTrackDatasource, MotionTrackDatasource, etc.)

4. Build timeline using existing `build_from_datasources` method

### 2. Helper Function: Extract Data from MNE Raw

**Location**: `pypho_timeline/timeline_builder.py` (as private method)

**Function**: `_extract_datasources_from_eeg_raw`

**Responsibilities**:

- Extract EEG channel data: `raw.get_data()` with channel names
- Extract timestamps: `raw.times` (relative to recording start)
- Extract Motion data: Check for motion channels (AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
- Extract Annotations: `raw.annotations.to_data_frame()`
- Convert timestamps to relative seconds from reference datetime
- Create intervals_df and detailed_df for each data type
- Return list of TrackDatasource instances

### 3. Datetime Alignment Logic

**Location**: `pypho_timeline/timeline_builder.py`

**Key Points**:

- Use `stream_infos_df['recording_datetime']` or `stream_infos_df['first_timestamp_dt']` as reference
- MNE Raw timestamps (`raw.times`) are relative to `raw.info['meas_date']`
- Convert to relative seconds: `(meas_date + timedelta(seconds=raw.times[i])) - reference_datetime`
- Use `datetime_to_float` helper from `datetime_helpers.py` for conversion

### 4. Datasource Creation

**Location**: `pypho_timeline/timeline_builder.py`

**For each data type**:

- **EEG**: Create `EEGTrackDatasource` with channel names from `raw.ch_names`
- **Motion**: Create `MotionTrackDatasource` if motion channels detected
- **Annotations**: Create `IntervalProvidingTrackDatasource` with log renderer for text annotations

### 5. Stream Info DataFrame Structure

**Expected columns in `stream_infos_df`**:

- `xdf_dataset_idx`: Index to match with Raw objects
- `recording_datetime`: Reference datetime for the recording
- `first_timestamp_dt`, `last_timestamp_dt`: Datetime boundaries
- `name`: Stream name
- `type`: Stream type (EEG, MOTION, etc.)

## Implementation Details

### Matching Strategy

- Iterate through `stream_infos_df` rows
- For each row, find corresponding Raw object: `eeg_raws[row['xdf_dataset_idx']]`
- Handle missing indices gracefully (skip or warn)

### Timestamp Conversion

```python
# Get reference datetime from stream info
reference_datetime = stream_info_row['recording_datetime']  # or first_timestamp_dt

# Get meas_date from Raw object
meas_date = raw.info.get('meas_date')

# Convert raw.times to absolute datetimes
absolute_times = [meas_date + timedelta(seconds=t) for t in raw.times]

# Convert to relative seconds from reference
relative_times = [datetime_to_float(dt, reference_datetime) for dt in absolute_times]
```

### Data Extraction Pattern

1. Check if Raw object has data: `len(raw.times) > 0`
2. Extract channel data: `raw.get_data(picks='eeg')` or `raw.get_data(picks=['AccX', 'AccY', ...])`
3. Create DataFrame with 't' column and channel columns
4. Create intervals_df with `t_start`, `t_duration`, `t_end`
5. Instantiate appropriate TrackDatasource

## Files to Modify

1. **`pypho_timeline/timeline_builder.py`**

   - Add `build_from_eeg_raw_and_stream_info` method
   - Add `_extract_datasources_from_eeg_raw` helper method
   - Import MNE types and EEG/Motion datasources

2. **Dependencies** (already available):

   - `EEGTrackDatasource` from `pypho_timeline.rendering.datasources.specific.eeg`
   - `MotionTrackDatasource` from `pypho_timeline.rendering.datasources.specific`
   - `datetime_helpers` utilities for timestamp conversion

## Testing Considerations

- Handle empty Raw objects gracefully
- Handle missing `xdf_dataset_idx` matches
- Handle missing datetime columns in stream_infos_df
- Test with single and multiple Raw objects
- Verify datetime alignment across multiple tracks