---
name: Refactor TrackDatasources to use IntervalProvidingTrackDatasource
overview: Refactor `PositionTrackDatasource` and `VideoTrackDatasource` to inherit from `IntervalProvidingTrackDatasource` instead of `BaseTrackDatasource`, removing duplicate code and leveraging the existing interval management functionality.
todos:
  - id: refactor-position-datasource
    content: Refactor PositionTrackDatasource to inherit from IntervalProvidingTrackDatasource, remove duplicate methods, override get_detail_renderer and visualization properties
    status: completed
  - id: refactor-video-datasource
    content: Refactor VideoTrackDatasource to inherit from IntervalProvidingTrackDatasource, remove duplicate methods, override fetch_detailed_data, get_detail_renderer and visualization properties
    status: completed
---

# Refactor TrackDatasources to use IntervalProvidingTrackDatasource

## Overview

Both `PositionTrackDatasource` and `VideoTrackDatasource` currently inherit from `BaseTrackDatasource` and duplicate functionality that `IntervalProvidingTrackDatasource` already provides. This refactoring will make them subclasses of `IntervalProvidingTrackDatasource` to reduce code duplication.

## Analysis

### Current State

- Both classes implement interval management methods (`df`, `time_column_names`, `total_df_start_end_times`, `get_updated_data_window`, `update_visualization_properties`, `get_overview_intervals`) that are already provided by `IntervalProvidingTrackDatasource`
- `IntervalProvidingTrackDatasource` (in `pypho_timeline/rendering/datasources/track_datasource.py`) handles:
- Interval DataFrame management
- Time window filtering
- Basic visualization properties (pens, brushes, offsets, heights)
- Optional detailed DataFrame filtering by time window

### Changes Required

#### PositionTrackDatasource

1. Change inheritance from `BaseTrackDatasource` to `IntervalProvidingTrackDatasource`
2. Update `__init__` to call parent with `intervals_df` and `position_df` (as `detailed_df`)
3. Remove duplicate implementations of:

- `df` property
- `time_column_names` property
- `total_df_start_end_times` property
- `get_updated_data_window` method
- `update_visualization_properties` method
- `get_overview_intervals` method
- `fetch_detailed_data` method (parent already handles this)

4. Override `get_detail_renderer()` to use `PositionPlotDetailRenderer` instead of `GenericPlotDetailRenderer`
5. Override visualization properties in `__init__` (color, height) after calling `super().__init__()`
6. Override `get_detail_cache_key()` to use "position_" prefix

#### VideoTrackDatasource

1. Change inheritance from `BaseTrackDatasource` to `IntervalProvidingTrackDatasource`
2. Update `__init__` to call parent with `video_intervals_df` (as `intervals_df`) and `None` (as `detailed_df`)
3. Remove duplicate implementations of:

- `df` property
- `time_column_names` property
- `total_df_start_end_times` property
- `get_updated_data_window` method
- `update_visualization_properties` method
- `get_overview_intervals` method

4. Override `fetch_detailed_data()` to generate synthetic video frames (parent's implementation won't work for video)
5. Override `get_detail_renderer()` to use `VideoThumbnailDetailRenderer`
6. Override visualization properties in `__init__` (color, height) after calling `super().__init__()`
7. Override `get_detail_cache_key()` to use "video_" prefix

## Implementation Details

### Files to Modify

- `pypho_timeline/__main__.py` (lines 45-204): Refactor both classes

### Key Considerations

- `IntervalProvidingTrackDatasource.__init__` expects `(intervals_df, detailed_df=None)`
- Parent class sets default visualization properties (blue color, height=1.0) that need to be overridden
- `PositionTrackDatasource` can use parent's `fetch_detailed_data` since it filters a DataFrame
- `VideoTrackDatasource` must override `fetch_detailed_data` since it generates synthetic data
- Both classes should preserve their existing `custom_datasource_name` values

## Expected Outcome

- Reduced code: ~80 lines removed per class