---
name: Complete MotionPlotDetailRenderer with distinct channel colors
overview: Complete the MotionPlotDetailRenderer implementation by generating distinct colors for each channel and fixing related issues in the renderer and motion datasource.
todos: []
---

# Complete MotionPlotDetailRenderer Implementation

## Overview

Complete the `MotionPlotDetailRenderer` class to use distinct colors for each channel in `channel_names`, and fix related issues in the renderer and motion datasource.

## Changes Required

### 1. `motion_plot_renderer.py` - Generate distinct colors for channels

**In `__init__` method:**

- Generate a list of distinct colors for each channel in `channel_names`
- Store as `self.pen_colors` (list matching the order of `channel_names`)
- Use a predefined color palette or `ColorsUtil` to ensure distinct, visible colors
- Default palette suggestion: `['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']` (cycling if more channels)

**In `render_detail` method:**

- Update the loop to use the corresponding color from `self.pen_colors` for each channel
- Map each `found_channel_name` to its index in `channel_names` to get the correct color

**In `get_detail_bounds` method:**

- Fix the method to work with motion data (time vs channel values)
- Remove references to `self.y_column` (doesn't exist in this class)
- Calculate bounds based on time range (x-axis) and the min/max across all channel values (y-axis)
- Handle the case when `channel_names` is None

### 2. `motion.py` - Fix get_detail_renderer method

**In `get_detail_renderer` method:**

- Remove the `y_column` parameter from `MotionPlotDetailRenderer` calls (this parameter doesn't exist)
- The renderer should use `channel_names` instead, which can be passed if needed
- Keep the `pen_color` and `pen_width` parameters, or remove `pen_color` if using per-channel colors

## Implementation Details

### Color Generation Strategy

- Use a predefined list of 6-8 distinct colors that work well for line plots
- If more channels than colors, cycle through the palette
- Colors should be easily distinguishable (avoid similar shades)

### Bounds Calculation

- X-axis: time range from `t_start` to `t_start + t_duration`
- Y-axis: min/max across all channel values found in the data
- Add 10% padding to y-axis range for better visualization

## Files to Modify

1. [`pypho_timeline/rendering/detail_renderers/motion_plot_renderer.py`](pypho_timeline/rendering/detail_renderers/motion_plot_renderer.py)

- Update `__init__` to generate `pen_colors` list
- Update `render_detail` to use per-channel colors