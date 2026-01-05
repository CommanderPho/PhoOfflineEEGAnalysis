---
name: Fix XDF example to show actual stream data
overview: Fix the XDF example to disable example tracks and properly display XDF stream intervals like PhoOfflineEEGAnalysis does, instead of showing generic example tracks.
todos: []
---

# Fix XDF Example to Show Ac

tual Stream Data

## Problem

1. `SimpleTimelineWidget.setupUI()` always calls `add_example_tracks()`, so example tracks appear even in the XDF example
2. The XDF example creates incorrect intervals (1-second intervals for every timestamp) instead of proper stream recording intervals
3. The intervals don't represent the actual stream data properly

## Solution

1. Make `add_example_tracks()` optional in `SimpleTimelineWidget`
2. Fix XDF stream interval extraction to create proper recording intervals (one interval per stream with correct start/end times)
3. Ensure XDF example doesn't call `add_example_tracks()`

## Changes

### File: `pypho_timeline/__main__.py`

**1. Make `add_example_tracks()` optional in `SimpleTimelineWidget.__init__`**Add a parameter to control whether example tracks are added:

```python
def __init__(self, total_start_time=0.0, total_end_time=100.0, window_duration=10.0, window_start_time=30.0, add_example_tracks=True, parent=None):
```

Then in `setupUI()`:

```python
# Add some example tracks (only if requested)
if add_example_tracks:
    self.add_example_tracks()
```

**2. Fix XDF stream interval extraction (lines 603-622)**Replace the incorrect interval creation with proper stream intervals:

```python
# --- 3. Build interval DataFrame for each EEG stream ---
eeg_datasources = []
for i, s in enumerate(eeg_streams):
    timestamps = s['time_stamps']
    stream_name = s['info']['name'][0]
    n_channels = int(s['info']['channel_count'][0])
    print(f"Stream {i}: {stream_name}, channels: {n_channels}, samples: {len(timestamps)}")

    # Create a single interval representing the entire stream recording
    if len(timestamps) == 0:
        continue
    
    stream_start = float(timestamps[0])
    stream_end = float(timestamps[-1])
    stream_duration = stream_end - stream_start
    
    # Create interval DataFrame with proper structure
    intervals_df = pd.DataFrame({
        't_start': [stream_start],
        't_duration': [stream_duration],
        't_end': [stream_end]
    })
    
    # Add visualization columns
    intervals_df['series_vertical_offset'] = 0.0
    intervals_df['series_height'] = 1.0
    
    # Create pens and brushes
    color = pg.mkColor('blue')
    color.setAlphaF(0.3)
    pen = pg.mkPen(color, width=1)
    brush = pg.mkBrush(color)
    intervals_df['pen'] = [pen]
    intervals_df['brush'] = [brush]
    
    # Create datasource
    datasource = PositionTrackDatasource(position_df=None, intervals_df=intervals_df)
    datasource.custom_datasource_name = f"EEG_{stream_name}"
    eeg_datasources.append(datasource)
```

**3. Disable example tracks in XDF example (line 625)**

```python
timeline = SimpleTimelineWidget(
    total_start_time=min([ds.total_df_start_end_times[0] for ds in eeg_datasources]),
    total_end_time=max([ds.total_df_start_end_times[1] for ds in eeg_datasources]),
    window_duration=10.0,
    window_start_time=min([ds.total_df_start_end_times[0] for ds in eeg_datasources]),
    add_example_tracks=False  # Don't add example tracks for XDF data
)
```



## Notes