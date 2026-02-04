---
name: Enhanced PhoLogTrack with Text Rendering
overview: Enhance PhoLogTrack to support both ['time', 'text'] and ['onset', 'duration'] data formats, and add intelligent text label rendering in detailed mode with overlap prevention, elliding/wrapping, and vertical staggering.
todos:
  - id: "1"
    content: Update PhoLogTrack.__init__ to detect and support both ['time', 'text'] and ['onset', 'duration'] data formats
    status: completed
  - id: "2"
    content: "Add text rendering infrastructure: override _render_detailed, add text item caching, enable detailed mode threshold"
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Implement _layout_text_labels method for overlap detection and vertical staggering
    status: completed
    dependencies:
      - "2"
  - id: "4"
    content: Implement _prepare_text_for_display method for elliding/wrapping text based on available width
    status: completed
    dependencies:
      - "2"
  - id: "5"
    content: Add point marker rendering for zero-duration entries using pg.ScatterPlotItem or vertical lines
    status: completed
    dependencies:
      - "2"
  - id: "6"
    content: Integrate text layout and rendering into _render_detailed method
    status: completed
    dependencies:
      - "3"
      - "4"
      - "5"
  - id: "7"
    content: "Add performance optimizations: viewport culling, text item pooling, render limits"
    status: completed
    dependencies:
      - "6"
---

# Enhanced PhoLogTrack with Text Rendering

## Overview

Enhance `PhoLogTrack` to support multiple data formats and add intelligent text label rendering that prevents overlap by using vertical staggering, elliding, and wrapping.

## Implementation Plan

### 1. Update PhoLogTrack Data Format Support

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- Modify `__init__` to detect and handle both formats:
- If `'time'` column exists: use `'time'` as onset, check for optional `'duration'` column
- If `'onset'` column exists: use existing behavior (backward compatible)
- If `'text'` column exists: store for text rendering
- When `'duration'` is missing and using `'time'` format: set duration to 0 (point markers)
- Pass appropriate column names to `StringDataTrack` base class

### 2. Add Text Rendering Infrastructure

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- Override `_render_detailed` method to render text labels
- Add instance variables:
- `_text_items: List[pg.TextItem]` - cache for text items
- `_max_text_height: int` - maximum vertical space for text (default: track height)
- Implement text label management:
- Clear old text items before rendering new ones
- Create new `pg.TextItem` objects for visible intervals with text data

### 3. Implement Overlap Detection and Layout Algorithm

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- Create `_layout_text_labels` method that:
- Takes visible intervals with their text and x-positions
- Detects horizontal overlaps (intervals that would overlap on x-axis)
- Groups overlapping intervals into "rows"
- Assigns y-positions to each text label (stagger vertically)
- Uses full track height (0 to track_height) for vertical distribution
- Returns list of (x, y, text, width) tuples for rendering
- Overlap detection algorithm:
  ```python
                # Sort intervals by start time
                # For each interval, check if it overlaps with any existing row
                # If overlaps, assign to next available row
                # Distribute rows evenly across available height
  ```




### 4. Implement Text Elliding and Wrapping

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- Create `_prepare_text_for_display` method:
- Calculate available width based on interval duration and zoom level
- If text fits: use as-is
- If text too long: ellide with "..." (prefer elliding over wrapping for performance)
- Optional: Support multi-line wrapping if text is very long and height allows
- Return processed text string
- Text width calculation:
- Get interval width in pixels: `(end_ts - start_ts) * pixels_per_second`
- Account for padding/margins
- Use QFontMetrics to measure text width

### 5. Render Point Markers for Zero-Duration Entries

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- In `_render_detailed`:
- Separate intervals into: (1) those with duration > 0 (bars), (2) those with duration = 0 (points)
- For point markers: use `pg.ScatterPlotItem` or small vertical lines
- Position point markers at appropriate y-level based on overlap layout
- Render bars for intervals with duration using existing bar rendering

### 6. Enable Detailed Mode Threshold

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- In `__init__`, set `self.detailed_mode_timespan_threshold_sec = 60.0` (or configurable)
- This enables detailed text rendering when zoomed in to show < 60 seconds of data

### 7. Performance Optimizations

**File**: [`src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py`](src/phoofflineeeganalysis/analysis/UI/timeline/tracks/PhoLogTrack.py)

- Only render text labels for visible intervals (viewport culling)
- Reuse `pg.TextItem` objects when possible (pooling)
- Cache text layout calculations until view range changes significantly
- Limit maximum number of text items rendered simultaneously (e.g., 100)

## Technical Details

### Data Format Detection Logic

```python
if 'time' in df.columns:
    onset_col = 'time'
    duration_col = df.get('duration', None)  # Optional
elif 'onset' in df.columns:
    onset_col = 'onset'
    duration_col = 'duration'
else:
    raise ValueError("DataFrame must have 'time' or 'onset' column")
```



### Text Layout Algorithm Pseudocode

```javascript
1. Get visible intervals sorted by start time
2. Initialize rows = []
3. For each interval:
   a. Find first row where interval doesn't overlap existing items
   b. If no such row, create new row
   c. Add interval to that row
4. Distribute rows evenly across y-range [0, track_height]
5. For each interval, calculate x-position and assigned y-position
6. Prepare text (ellide/wrap) based on available width
7. Create TextItem at (x, y) with processed text
```



### Dependencies

- PyQt5 (already used)
- pyqtgraph (already used)
- pandas (already used)
- numpy (already used)

## Testing Considerations

- Test with both `['time', 'text']` and `['onset', 'duration']` formats
- Test with missing duration column (point markers)
- Test overlap scenarios (many overlapping intervals)
- Test with very long text strings (elliding)
- Test performance with large datasets (1000+ intervals)
- Test zoom in/out behavior (overview vs detailed mode switching)