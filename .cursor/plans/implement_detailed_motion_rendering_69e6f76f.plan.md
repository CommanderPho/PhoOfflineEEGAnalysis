---
name: Implement detailed motion rendering
overview: Implement detailed display methods for MotionRecordingTrack to render position data as line plots. Each position column (AccX, AccY, AccZ, GyroX, GyroY, GyroZ) will be displayed as a separate 1D line plot within the visible interval time range.
todos:
  - id: add_position_datasource
    content: Add position_datasource parameter to __init__ and store as self._position_datasource
    status: pending
  - id: implement_ensure_detailed_items
    content: Implement _ensure_detailed_items() to create 6 PlotDataItems (one per position column) with distinct colors
    status: pending
  - id: implement_clear_detailed_items
    content: Implement _clear_detailed_items() to hide/clear all detailed plot items
    status: pending
    dependencies:
      - implement_ensure_detailed_items
  - id: implement_render_detailed
    content: Implement _render_detailed() to query position data, update PlotDataItems, and handle y-axis scaling
    status: pending
    dependencies:
      - implement_ensure_detailed_items
      - add_position_datasource
  - id: update_render_overview
    content: Update _render_overview() to call _clear_detailed_items() and ensure bar graph is visible
    status: pending
    dependencies:
      - implement_clear_detailed_items
---

# Implement Detailed Motion Position Rendering

## Overview

Implement detailed rendering methods in `MotionRecordingTrack` to display position data as line plots. The position dataframe has columns: `time`, `AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, `GyroZ`. Each position column will be rendered as a separate line plot when zoomed into an interval.

## Architecture

The implementation will:

1. Store PlotDataItems for each position column (6 total)
2. Access position data from a datasource for the visible time range
3. Render lines with proper time-to-timestamp conversion
4. Handle y-axis scaling (normalize/stack channels for visibility)

## Implementation Details

### Files to Modify

- [`PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/UI/timeline/tracks/MotionRecordingTrack.py`](PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/UI/timeline/tracks/MotionRecordingTrack.py)

### Key Changes

1. **Add position datasource storage** (in `__init__`):

- Add optional `position_datasource` parameter to accept a `DataframeDatasource` with position data
- Store it as `self._position_datasource`
- If not provided, attempt to access position data from the existing datasource

2. **Implement `_ensure_detailed_items()`**:

- Create 6 `pg.PlotDataItem` objects (one per position column)
- Store them in `self._detailed_plot_items: Dict[str, pg.PlotDataItem]`
- Add each to `self.plot_widget`
- Use distinct colors for each channel (AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
- Initially hide them (set data to empty arrays)

3. **Implement `_clear_detailed_items()`**:

- Hide all detailed plot items by setting their data to empty arrays
- Or remove them from the plot widget (prefer hiding for performance)

4. **Implement `_render_detailed()`**:

- Hide overview bar graph: `self.bar_graph_item.setVisible(False)`
- Find visible intervals that overlap with `time_range`
- For each visible interval:
    - Convert interval start/end datetime to timestamps (float seconds)
    - Query position datasource using `get_updated_data_window(start_ts, end_ts)`
    - Filter position dataframe to the interval time range
- Combine position data from all visible intervals
- For each position column:
    - Extract time and value arrays
    - Convert time column to timestamps (if needed)
    - Update corresponding PlotDataItem with `setData(x=times, y=values)`
- Handle y-axis scaling:
    - Option A: Normalize each channel to [0, 1] and stack vertically
    - Option B: Use auto-range per channel with offset
    - Option C: Use a single y-range that encompasses all channels
- Show detailed plot items: `item.setVisible(True)`

5. **Update `_render_overview()`** (if needed):

- Ensure detailed items are cleared when in overview mode
- Call `_clear_detailed_items()` at the start

### Data Access Strategy

The position dataframe structure:

- Time column: `time` (float timestamps in seconds)
- Position columns: `AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, `GyroZ`

Access pattern:

- Use `position_datasource.get_updated_data_window(start_ts, end_ts)` to query data
- Filter by `time` column: `df[df['time'].between(start_ts, end_ts)]`
- Convert datetime to timestamp: `self._safe_datetime_to_timestamp(dt)`

### Color Scheme

- AccX, AccY, AccZ: Shades of red/orange (matching track theme)
- GyroX, GyroY, GyroZ: Shades of blue/cyan (complementary)

### Y-Axis Scaling

- Use `plot_widget.setYRange()` to set appropriate range
- Consider normalizing channels to [0, 1] and stacking with offsets (0, 1, 2, 3, 4, 5)
- Or use auto-range to fit all data

### Error Handling

- Handle missing position datasource gracefully (fall back to overview)
- Handle empty position data (hide detailed items)
- Handle invalid time ranges
- Handle missing columns in position dataframe

## Testing Considerations

- Test with single interval
- Test with multiple overlapping intervals