---
name: Fix UTC timezone for MNE compatibility
overview: Fix the timezone issue in `easy_time_sync.py` by converting from `pytz.timezone("UTC")` to `datetime.timezone.utc` so that MNE receives the expected timezone type. This fixes the root cause rather than patching downstream.
todos:
  - id: "1"
    content: Update import statement to include timezone from datetime module
    status: completed
  - id: "2"
    content: "Fix line 65: Replace pytz.timezone('UTC') with datetime.timezone.utc in capture_current_arbitrary_time_sync_point()"
    status: completed
  - id: "3"
    content: "Fix line 129: Replace tz_UTC with timezone.utc in parse_and_add_lsl_outlet_info_from_desc()"
    status: completed
isProject: false
---

## Problem

MNE-Python's `Annotations` class requires `datetime.timezone.utc` specifically, but `EasyTimeSyncParsingMixin.parse_and_add_lsl_outlet_info_from_desc()` returns datetime objects with `pytz.timezone("UTC")` timezone, causing the error:

```
ValueError: Date must be datetime object in UTC: datetime.datetime(2026, 1, 14, 14, 50, 55, tzinfo=<UTC>)
```

## Root Cause

In [`easy_time_sync.py`](C:\Users\pho\repos\EmotivEpoc\PhoPyLSLhelper\src\phopylslhelper\easy_time_sync.py):

- **Line 65**: Creates datetimes using `pytz.timezone('UTC')` (with a comment suggesting `datetime.timezone.utc`)
- **Line 129**: Converts parsed datetimes to UTC using `tz_UTC` (which is `pytz.timezone("UTC")` from `general_helpers`)

Both locations need to use `datetime.timezone.utc` instead.

## Solution

1. **Add `timezone` import**: Update line 2 to import `timezone` from `datetime` module
2. **Fix line 65**: Change `pytz.timezone('UTC')` to `datetime.timezone.utc` in `capture_current_arbitrary_time_sync_point()`
3. **Fix line 129**: Change `tz_UTC` to `datetime.timezone.utc` in `parse_and_add_lsl_outlet_info_from_desc()`

## Changes

### File: `C:\Users\pho\repos\EmotivEpoc\PhoPyLSLhelper\src\phopylslhelper\easy_time_sync.py`

1. **Line 2**: Update import to include `timezone`:
   ```python
   from datetime import datetime, timedelta, timezone
   ```

2. **Line 65**: Replace `pytz.timezone('UTC')` with `datetime.timezone.utc`:
   ```python
   current_datetime = datetime.now(timezone.utc)  # Changed from pytz.timezone('UTC')
   ```

3. **Line 129**: Replace `tz_UTC` with `timezone.utc`:
   ```python
   a_ts_value = a_ts_value.astimezone(timezone.utc)  # Changed from tz_UTC
   ```


## Side Effects Analysis

**Low risk changes:**

- Both `pytz.timezone("UTC")` and `datetime.timezone.utc` represent the same UTC timezone
- The actual datetime values remain identical
- Most code only checks for timezone-awareness, not the specific timezone type
- This fixes the root cause, so downstream code (like `xdf_files.py`) will automatically receive correct timezone objects

**Potential considerations:**

- Any code that explicitly checks for `pytz` timezone types might need updates (unlikely in this codebase)
- The `tz_UTC` import from `general_helpers` is still available for other uses, but we're not using it here anymore

## Testing

After the fix, the datetime objects returned by `parse_and_add_lsl_outlet_info_from_desc()` will have `tzinfo=timezone.utc`, which MNE will accept. The error should no longer occur when creating `mne.Annotations` objects in `xdf_files.py` line 719.