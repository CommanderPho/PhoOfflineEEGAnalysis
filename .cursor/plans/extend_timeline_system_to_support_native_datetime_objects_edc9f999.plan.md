---
name: Extend Timeline System to Support Native Datetime Objects
overview: Extend the timeline system to support datetime objects (pd.Timestamp/datetime) natively throughout, instead of converting to float timestamps. This requires changes to datasource interfaces, timeline widget, and all time-based operations.
todos:
  - id: "1"
    content: Update BaseTrackDatasource.total_df_start_end_times to detect and return datetime objects
    status: completed
  - id: "2"
    content: Update BaseTrackDatasource.get_updated_data_window to handle datetime comparisons
    status: completed
  - id: "3"
    content: Update SimpleTimelineWidget to accept and store datetime objects for time properties
    status: completed
  - id: "4"
    content: Modify _extract_datasources_from_eeg_raw to store datetime objects directly in DataFrames
    status: completed
  - id: "5"
    content: Update build_from_datasources to calculate time ranges using datetime operations
    status: completed
  - id: "6"
    content: Update _add_tracks_to_timeline to handle datetime objects when setting plot ranges
    status: completed
  - id: "7"
    content: Update detail renderers to convert datetime t columns to Unix timestamps for plotting
    status: completed
  - id: "8"
    content: Test backward compatibility with float timestamps
    status: completed
isProject: false
---

# Extend Timeline System to Support Native Datetime Objects

## Overview

The timeline system currently uses float timestamps throughout. We need to extend it to support datetime objects (pd.Timestamp or datetime) natively, eliminating the need for relative timestamp calculations.

## Architecture Changes

### Current State

- Datasources store `t_start`, `t_duration` as floats
- `total_df_start_end_times` returns `Tuple[float, float]`
- `get_updated_data_window(new_start: float, new_end: float)` expects floats
- Timeline widget stores time properties as floats
- All comparisons and arithmetic use float operations

### Target State

- Datasources store `t_start`, `t` as datetime objects (pd.Timestamp)
- `total_df_start_end_times` returns `Tuple[datetime, datetime]` or `Tuple[pd.Timestamp, pd.Timestamp]`
- `get_updated_data_window` accepts datetime objects
- Timeline widget stores time properties as datetime objects
- All comparisons and arithmetic use datetime operations

## Implementation Plan

### 1. Update Datasource Interface

**Location**: `pypho_timeline/rendering/datasources/track_datasource.py` and `interval_datasource.py`

**Changes**:

- Change `total_df_start_end_times` return type from `Tuple[float, float]` to `Union[Tuple[float, float], Tuple[datetime, datetime], Tuple[pd.Timestamp, pd.Timestamp]]`
- Change `get_updated_data_window` signature to accept `Union[float, datetime, pd.Timestamp]`
- Update type hints to support both float and datetime

**Key Methods to Update**:

```python
@property
def total_df_start_end_times(self) -> Union[Tuple[float, float], Tuple[datetime, datetime]]:
    """Returns (earliest_time, latest_time) for the entire dataset"""
    # Support both float and datetime
    ...

def get_updated_data_window(self, new_start: Union[float, datetime], new_end: Union[float, datetime]) -> pd.DataFrame:
    """Returns intervals overlapping with time window"""
    # Handle both float and datetime comparisons
    ...
```

### 2. Update BaseTrackDatasource Implementation

**Location**: `pypho_timeline/rendering/datasources/track_datasource.py`

**Changes**:

- Modify `total_df_start_end_times` to detect datetime columns and return datetime objects
- Modify `get_updated_data_window` to handle datetime comparisons
- Ensure pandas datetime arithmetic works correctly

**Implementation**:

```python
@property
def total_df_start_end_times(self) -> tuple:
    if len(self.intervals_df) == 0:
        # Return appropriate default based on expected type
        return (pd.Timestamp('1970-01-01'), pd.Timestamp('1970-01-02'))
    
    t_start = self.intervals_df['t_start'].min()
    t_end = (self.intervals_df['t_start'] + pd.to_timedelta(self.intervals_df['t_duration'], unit='s')).max()
    
    # If t_start is datetime-like, return as-is; otherwise convert
    if pd.api.types.is_datetime64_any_dtype(self.intervals_df['t_start']):
        return (t_start, t_end)
    else:
        return (float(t_start), float(t_end))

def get_updated_data_window(self, new_start: Union[float, datetime], new_end: Union[float, datetime]) -> pd.DataFrame:
    """Get intervals overlapping with time window."""
    # Convert inputs to match DataFrame dtype
    if pd.api.types.is_datetime64_any_dtype(self.intervals_df['t_start']):
        if isinstance(new_start, (int, float)):
            new_start = pd.Timestamp.fromtimestamp(new_start, tz='UTC')
        if isinstance(new_end, (int, float)):
            new_end = pd.Timestamp.fromtimestamp(new_end, tz='UTC')
    
    # Calculate t_end for each interval
    if pd.api.types.is_datetime64_any_dtype(self.intervals_df['t_start']):
        t_end_col = self.intervals_df['t_start'] + pd.to_timedelta(self.intervals_df['t_duration'], unit='s')
    else:
        t_end_col = self.intervals_df['t_start'] + self.intervals_df['t_duration']
    
    mask = (t_end_col >= new_start) & (self.intervals_df['t_start'] <= new_end)
    return self.intervals_df[mask].copy()
```

### 3. Update Timeline Widget

**Location**: `pypho_timeline/widgets/simple_timeline_widget.py`

**Changes**:

- Change `total_data_start_time`, `total_data_end_time`, `active_window_start_time`, `active_window_end_time` to support datetime objects
- Update `__init__` to accept datetime objects
- Update time range calculations to work with datetime objects
- Update plot axis setting to handle datetime objects directly

**Key Updates**:

```python
def __init__(self, total_start_time: Union[float, datetime] = 0.0, 
             total_end_time: Union[float, datetime] = 100.0, 
             window_duration: Union[float, timedelta] = 10.0, 
             window_start_time: Union[float, datetime] = 30.0, 
             add_example_tracks=False, 
             reference_datetime: Optional[datetime] = None, 
             parent=None):
    # Store as datetime if provided, or convert from float
    if isinstance(total_start_time, (int, float)):
        if reference_datetime:
            self.total_data_start_time = reference_datetime + timedelta(seconds=total_start_time)
        else:
            self.total_data_start_time = pd.Timestamp.fromtimestamp(total_start_time, tz='UTC')
    else:
        self.total_data_start_time = pd.Timestamp(total_start_time)
    
    # Similar for other time properties...
```

### 4. Update TimelineBuilder

**Location**: `pypho_timeline/timeline_builder.py`

**Changes**:

- Modify `_extract_datasources_from_eeg_raw` to store datetime objects directly in DataFrames
- Update `build_from_datasources` to calculate time ranges using datetime operations
- Ensure reference_datetime is used appropriately (for display, not conversion)

**Key Changes in `_extract_datasources_from_eeg_raw`**:

```python
# Instead of converting to relative timestamps:
# relative_times = [datetime_to_float(dt, reference_datetime) for dt in absolute_times]

# Store datetime objects directly:
absolute_times = [meas_date + timedelta(seconds=float(t)) for t in raw.times]
# Convert to pd.Timestamp for consistency
datetime_times = [pd.Timestamp(dt) for dt in absolute_times]

# Store in DataFrame
eeg_df['t'] = datetime_times  # pd.Series with datetime dtype

# Create intervals_df with datetime
base_intervals_df = pd.DataFrame({
    't_start': [pd.Timestamp(absolute_times[0])],
    't_duration': [pd.Timedelta(seconds=t_duration)],
    't_end': [pd.Timestamp(absolute_times[-1])]
})
```

### 5. Update Plot Rendering

**Location**: `pypho_timeline/timeline_builder.py` and `pypho_timeline/widgets/simple_timeline_widget.py`

**Changes**:

- When setting plot X ranges, if timestamps are datetime objects, convert directly to Unix timestamps for PyQtGraph
- Remove the intermediate relative→absolute conversion step

**Update in `_add_tracks_to_timeline`**:

```python
# If timeline uses datetime objects
if isinstance(timeline.total_data_start_time, (datetime, pd.Timestamp)):
    unix_start = datetime_to_unix_timestamp(timeline.total_data_start_time)
    unix_end = datetime_to_unix_timestamp(timeline.total_data_end_time)
    a_plot_item.setXRange(unix_start, unix_end, padding=0)
else:
    # Fallback to float conversion (backward compatibility)
    if timeline.reference_datetime is not None:
        dt_start = float_to_datetime(timeline.total_data_start_time, timeline.reference_datetime)
        dt_end = float_to_datetime(timeline.total_data_end_time, timeline.reference_datetime)
        unix_start = datetime_to_unix_timestamp(dt_start)
        unix_end = datetime_to_unix_timestamp(dt_end)
        a_plot_item.setXRange(unix_start, unix_end, padding=0)
```

### 6. Update Detail Renderers

**Location**: `pypho_timeline/rendering/detail_renderers/`

**Changes**:

- Ensure detail renderers can handle datetime objects in the 't' column
- Update bounds calculations to work with datetime objects

**Key Consideration**:

- PyQtGraph PlotDataItem expects numeric values for plotting
- Convert datetime 't' column to Unix timestamps when rendering: `eeg_df['t'].apply(lambda x: x.timestamp() if isinstance(x, (datetime, pd.Timestamp)) else x)`

## Backward Compatibility

To maintain backward compatibility:

- Support both float and datetime inputs throughout
- Detect dtype of DataFrame columns and handle accordingly
- Convert between formats only when necessary (at interface boundaries)

## Files to Modify

1. **`pypho_timeline/rendering/datasources/track_datasource.py`**

   - Update `BaseTrackDatasource.total_df_start_end_times`
   - Update `BaseTrackDatasource.get_updated_data_window`

2. **`pypho_timeline/rendering/datasources/interval_datasource.py`**

   - Update protocol type hints

3. **`pypho_timeline/widgets/simple_timeline_widget.py`**

   - Update `SimpleTimelineWidget.__init__`
   - Update time property handling
   - Update plot range setting

4. **`pypho_timeline/timeline_builder.py`**

   - Update `_extract_datasources_from_eeg_raw` to use datetime objects
   - Update `build_from_datasources` time range calculations
   - Update `_add_tracks_to_timeline` plot range setting

5. **`pypho_timeline/rendering/detail_renderers/generic_plot_renderer.py`** (if needed)

   - Ensure datetime 't' columns are converted to Unix timestamps for plotting

## Testing Considerations

- Test with datetime objects in DataFrames
- Test with float timestamps (backward compatibility)
- Test mixed scenarios (some datasources with datetime, some with float)
- Verify plot rendering works correctly
- Verify time window scrolling works with datetime objects
- Test with multiple recordings spanning different time periods