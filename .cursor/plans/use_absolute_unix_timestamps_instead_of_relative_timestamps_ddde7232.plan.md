---
name: Use Absolute Unix Timestamps Instead of Relative Timestamps
overview: Modify the `build_from_eeg_raw_and_stream_info` implementation to use absolute Unix timestamps (seconds since epoch) instead of relative timestamps. This ensures all timestamps are absolute datetimes, eliminating the need for relative time calculations.
todos:
  - id: "1"
    content: Replace datetime_to_float with datetime_to_unix_timestamp in _extract_datasources_from_eeg_raw
    status: pending
  - id: "2"
    content: Update EEG datasource timestamp assignment to use Unix timestamps
    status: pending
  - id: "3"
    content: Update Motion datasource timestamp assignment to use Unix timestamps
    status: pending
  - id: "4"
    content: Update annotation timestamp conversion to use Unix timestamps
    status: pending
  - id: "5"
    content: Update base_intervals_df to use Unix timestamps for t_start, t_end, t_duration
    status: pending
isProject: false
---

# Use Absolute Unix Timestamps Instead of Relative Timestamps

## Problem

The current implementation converts absolute datetimes to relative seconds using `datetime_to_float(dt, reference_datetime)`, which produces relative timestamps like `(0.0, 5485.875)`. The user wants absolute datetimes everywhere, not relative offsets.

## Solution

Instead of converting to relative timestamps, use absolute Unix timestamps (seconds since 1970-01-01 UTC). This makes all timestamps absolute and allows direct conversion to datetimes without needing a reference point.

## Changes Required

### 1. Modify `_extract_datasources_from_eeg_raw` method

**Location**: `pypho_timeline/timeline_builder.py`

**Current approach** (lines 626-633):

```python
# Convert raw.times to relative seconds from reference_datetime
absolute_times = [meas_date + timedelta(seconds=float(t)) for t in raw.times]
relative_times = [datetime_to_float(dt, reference_datetime) for dt in absolute_times]
```

**New approach**:

```python
# Convert raw.times to absolute Unix timestamps
absolute_times = [meas_date + timedelta(seconds=float(t)) for t in raw.times]
unix_times = [datetime_to_unix_timestamp(dt) for dt in absolute_times]
```

**Key changes**:

- Replace `datetime_to_float(dt, reference_datetime)` with `datetime_to_unix_timestamp(dt)`
- Store Unix timestamps in `eeg_df['t']`, `motion_df['t']`, and annotation intervals
- Update `base_intervals_df` to use Unix timestamps for `t_start`, `t_end`, `t_duration`

### 2. Update reference_datetime usage

**Location**: `pypho_timeline/timeline_builder.py`

The `reference_datetime` should still be set (for display purposes), but it's no longer needed for timestamp conversion. However, we should set it to the earliest absolute datetime from the data, not use it for relative calculations.

**Change in `build_from_eeg_raw_and_stream_info`**:

- Keep the reference_datetime extraction logic (for timeline display)
- But don't use it for timestamp conversion - use Unix timestamps directly

### 3. Update annotation timestamp conversion

**Location**: `pypho_timeline/timeline_builder.py` (lines ~700-730)

**Current approach**:

```python
annotation_relative_times = []
for onset in annotations_df['onset']:
    absolute_time = meas_date + timedelta(seconds=float(onset))
    relative_time = datetime_to_float(absolute_time, reference_datetime)
    annotation_relative_times.append(relative_time)
```

**New approach**:

```python
annotation_unix_times = []
for onset in annotations_df['onset']:
    absolute_time = meas_date + timedelta(seconds=float(onset))
    unix_time = datetime_to_unix_timestamp(absolute_time)
    annotation_unix_times.append(unix_time)
```

### 4. Import datetime_to_unix_timestamp

**Location**: `pypho_timeline/timeline_builder.py` (line 20)

Already imported, but ensure it's used instead of `datetime_to_float`.

## Implementation Details

### Timestamp Flow

1. **MNE Raw timestamps** (`raw.times`) are relative to `meas_date`
2. **Convert to absolute datetime**: `meas_date + timedelta(seconds=raw.times[i])`
3. **Convert to Unix timestamp**: `datetime_to_unix_timestamp(absolute_datetime)`
4. **Store Unix timestamps** in datasources (no relative conversion)

### Benefits

- All timestamps are absolute (Unix epoch)
- No need for reference_datetime for data conversion (only for display)
- Timestamps can be directly converted to datetimes: `datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)`
- Timeline can still use reference_datetime for display formatting, but data is absolute

### Compatibility

The timeline system already handles Unix timestamps when `reference_datetime` is set - it converts relative timestamps to datetimes, then to Unix timestamps for display. With absolute Unix timestamps, we can skip the relative→absolute conversion step.

## Files to Modify

1. **`pypho_timeline/timeline_builder.py`**

   - Replace `datetime_to_float` calls with `datetime_to_unix_timestamp` in `_extract_datasources_from_eeg_raw`
   - Update all timestamp assignments to use Unix timestamps
   - Update intervals_df creation to use Unix timestamps

## Testing Considerations

- Verify timestamps are absolute (large numbers, not starting from 0)
- Verify timeline displays correct datetimes
- Verify multiple recordings from different times are correctly aligned
- Check that reference_datetime is still used for display formatting