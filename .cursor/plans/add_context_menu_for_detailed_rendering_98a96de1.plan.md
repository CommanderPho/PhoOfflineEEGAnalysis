---
name: Add context menu for detailed rendering
overview: Add a right-click context menu to interval rectangles that allows enabling detailed rendering for a specific (track, interval) pair. The menu will appear when right-clicking on an interval rectangle and will include a "Render detailed" option.
todos:
  - id: "1"
    content: Add detail_render_callback parameter to IntervalRectsItem.__init__ and store as instance variable
    status: completed
  - id: "2"
    content: Modify getContextMenus() to add 'Render detailed' action that uses the callback
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Add method to handle right-click event and identify clicked rectangle index
    status: completed
    dependencies:
      - "1"
  - id: "4"
    content: Update TrackRenderer._update_overview() to create detail rendering callback
    status: completed
  - id: "5"
    content: Pass callback to build_IntervalRectsItem_from_interval_datasource() when creating overview_rects_item
    status: completed
    dependencies:
      - "4"
  - id: "6"
    content: Update render_rectangles_helper.py to accept and pass through callback parameter if needed
    status: completed
---

# Add Context Menu for Detailed Rendering

## Overview

Add a right-click context menu to `IntervalRectsItem` that enables detailed rendering for a specific interval. When a user right-clicks on an interval rectangle, they'll see a "Render detailed" option that triggers detailed rendering for that specific interval.

## Implementation Plan

### 1. Modify `IntervalRectsItem` to accept a detail rendering callback

- **File**: `pypho_timeline/rendering/graphics/interval_rects_item.py`
- Add optional `detail_render_callback` parameter to `__init__`
- The callback should accept: `(rect_index: int, rect_data: IntervalRectsItemData) -> None`
- Store the callback as an instance variable

### 2. Update context menu to include "Render detailed" option

- **File**: `pypho_timeline/rendering/graphics/interval_rects_item.py`
- Modify `getContextMenus()` to add a "Render detailed" action
- Only show this option if `detail_render_callback` is provided
- Connect the action to a new method that:
 - Gets the clicked rectangle index from the mouse event position
 - Calls the callback with the rectangle index and data

### 3. Update `TrackRenderer` to provide the callback

- **File**: `pypho_timeline/rendering/graphics/track_renderer.py`
- Modify `_update_overview()` to create a callback function that:
 - Takes `rect_index` and `rect_data`
 - Maps the rectangle index to the corresponding interval in the datasource
 - Calls `_render_detail()` for that specific interval
- Pass this callback to `build_IntervalRectsItem_from_interval_datasource()` via `format_tooltip_fn` or a new parameter

### 4. Handle interval-to-rectangle mapping

- **File**: `pypho_timeline/rendering/graphics/track_renderer.py`
- Store a mapping from rectangle indices to interval cache keys or interval data
- Use this mapping in the callback to identify which interval to render
- Ensure the interval data is properly formatted for `_render_detail()`

### 5. Update helper function if needed

- **File**: `pypho_timeline/rendering/helpers/render_rectangles_helper.py`
- If `build_IntervalRectsItem_from_interval_datasource()` needs to accept the callback, add it as an optional parameter and pass it through to `IntervalRectsItem`

## Key Considerations

1. **Event handling**: The context menu is triggered on right-click, but we need to identify which rectangle was clicked. Use `_get_rect_at_position()` method that already exists in `IntervalRectsItem`.

2. **Interval identification**: Map the rectangle index to the actual interval data from the datasource. The `overview_df` used in `_update_overview()` should match the order of rectangles in `IntervalRectsItem.data`.

3. **Detail rendering**: The `_render_detail()` method expects:

- `interval`: A DataFrame with a single row
- `cache_key`: A string cache key
- `detail_data`: The detail data to render

The callback should fetch or trigger fetching of the detail data if not already cached.

4. **Error handling**: Handle cases where:

- The interval doesn't have detail data available
- The detail data fetch fails
- The interval is no longer valid

## Files to Modify

1. `pypho_timeline/rendering/graphics/interval_rects_item.py` - Add callback parameter and context menu option
2. `pypho_timeline/rendering/graphics/track_renderer.py` - Create and pass callback to IntervalRectsItem
3. `pypho_timeline/rendering/helpers/render_rectangles_helper.py` - Optionally update to pass callback through

## Testing Considerations

- Right-click on different intervals should show the context menu
- "Render detailed" should only appear if the track supports detailed rendering
- Clicking "Render detailed" should trigger rendering for that specific interval
- Multiple intervals can have detailed rendering enabled simultaneously
- The detailed rendering should appear as an overlay on the interval rectangle