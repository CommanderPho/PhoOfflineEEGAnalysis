---
name: Standardize Datasource Protocol
overview: Create a concrete base class implementing the TrackDatasource protocol and update example datasources to explicitly inherit from it, making the required interface clear for new datasource implementations.
todos:
  - id: enhance-protocol
    content: Add @runtime_checkable to TrackDatasource protocol and improve documentation
    status: completed
  - id: create-base-class
    content: Create BaseTrackDatasource ABC class with default implementations and abstract methods
    status: completed
    dependencies:
      - enhance-protocol
  - id: update-position-datasource
    content: Update PositionTrackDatasource to inherit from BaseTrackDatasource
    status: completed
    dependencies:
      - create-base-class
  - id: update-video-datasource
    content: Update VideoTrackDatasource to inherit from BaseTrackDatasource
    status: completed
    dependencies:
      - create-base-class
---

# Standardize Datasource Protocol Implementation

## Overview

The example datasources (`PositionTrackDatasource` and `VideoTrackDatasource`) currently implement all required methods but don't explicitly declare conformance to the `TrackDatasource` protocol. This plan will:

1. Make `TrackDatasource` runtime-checkable
2. Create a concrete `BaseTrackDatasource` ABC class that implements the protocol
3. Update example datasources to explicitly inherit from the base class
4. Enhance documentation to clearly show required vs optional methods

## Implementation Details

### 1. Enhance TrackDatasource Protocol (`track_datasource.py`)

- Add `@runtime_checkable` decorator to `TrackDatasource` protocol to enable `isinstance()` checks
- Improve docstring to clearly categorize required vs optional methods
- Add type hints for better IDE support

### 2. Create BaseTrackDatasource ABC (`track_datasource.py`)

- Create `BaseTrackDatasource` abstract base class that implements `TrackDatasource` protocol
- Provide default implementations for common methods where appropriate:
- `get_overview_intervals()` - default to returning `df` property
- `get_detail_cache_key()` - default implementation using `t_start` and `t_duration`
- Mark abstract methods that must be implemented:
- `df` property
- `time_column_names` property  
- `total_df_start_end_times` property
- `get_updated_data_window()`
- `fetch_detailed_data()`
- `get_detail_renderer()`
- Initialize common attributes (`custom_datasource_name`, `source_data_changed_signal`)

### 3. Update Example Datasources (`__main__.py`)

- Make `PositionTrackDatasource` inherit from `BaseTrackDatasource`
- Make `VideoTrackDatasource` inherit from `BaseTrackDatasource`
- Remove redundant implementations that are now provided by base class
- Add type annotations to make protocol conformance explicit

### 4. Documentation

- Add comprehensive docstring to `BaseTrackDatasource` explaining:
- Required methods that must be implemented
- Optional methods with default implementations
- Best practices for implementing new datasources
- Update example datasource docstrings to reference the base class

## Files to Modify

1. [`pypho_timeline/rendering/datasources/track_datasource.py`](pypho_timeline/rendering/datasources/track_datasource.py)

- Add `@runtime_checkable` to `TrackDatasource`
- Create `BaseTrackDatasource` ABC class
- Export `BaseTrackDatasource` in `__all__`

2. [`pypho_timeline/__main__.py`](pypho_timeline/__main__.py)

- Update `PositionTrackDatasource` to inherit from `BaseTrackDatasource`
- Update `VideoTrackDatasource` to inherit from `BaseTrackDatasource`
- Import `BaseTrackDatasource` from track_datasource module
- Simplify implementations where base class provides defaults

## Benefits

- **Explicit conformance**: Example datasources clearly show they implement the protocol
- **Better documentation**: Base class serves as living documentation of required interface
- **Easier implementation**: New datasources can inherit from base class and override only what's needed