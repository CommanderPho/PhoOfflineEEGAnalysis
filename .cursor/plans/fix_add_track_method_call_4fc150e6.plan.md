---
name: Fix add_track method call
overview: Fix the incorrect `add_track()` method call in `main_all_eeg_modalities_from_xdf_file_example()` by changing `label=` to `name=` and removing the invalid `sync_mode` parameter.
todos: []
---

# Fix add_track Method Call Error

## Problem

The `add_track()` method is being called with incorrect parameters:

- Using `label=` instead of `name=` (the method requires `name` as the second positional argument)
- Passing `sync_mode=` which is not a valid parameter for `add_track()`

## Solution

Update the `add_track()` call in [`pypho_timeline/__main__.py`](H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoTimeline\pypho_timeline\__main__.py) at lines 629-633 to use the correct parameter name.

## Changes

### File: `pypho_timeline/__main__.py`

**Location:** Lines 629-633**Current code:**

```python
for datasource in eeg_datasources:
    timeline.add_track(
        datasource,
        label=datasource.custom_datasource_name,
        sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA
    )
```

**Fixed code:**

```python
for datasource in eeg_datasources:
    timeline.add_track(
        datasource,
        name=datasource.custom_datasource_name
    )
```



## Notes