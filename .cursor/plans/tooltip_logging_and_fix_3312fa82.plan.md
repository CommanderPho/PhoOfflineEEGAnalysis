---
name: Tooltip logging and fix
overview: Add logging around the epoch/video rectangle tooltip path so failures are visible, and fix the missing import that is causing tooltips to fail silently.
todos: []
isProject: false
---

# Tooltip logging and bug fix

## Where tooltips are shown

- **File**: [pyPhoTimeline/pypho_timeline/rendering/graphics/interval_rects_item.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\graphics\interval_rects_item.py)
- **Flow**: `hoverMoveEvent` → `_show_tooltip_for_rect(rect_index, global_pos)` → `self._current_hovered_item_tooltip_format_fn(rect_index=..., rect_data_tuple=...)` → `QToolTip.showText(global_pos, tooltip_text)`.
- **Default formatter**: `_default_format_tooltip_for_rect_data` (lines 354–391). It is invoked with no try/except; any exception there propagates and is often swallowed by the Qt event loop, so no tooltip and no log.

## Clear error: missing import

In `_default_format_tooltip_for_rect_data` the code calls:

```377:378:pypho_timeline/rendering/graphics/interval_rects_item.py
        start_t = unix_timestamp_to_datetime(start_t)
        end_t = unix_timestamp_to_datetime(end_t)
```

`unix_timestamp_to_datetime` is **not imported**. Only `format_seconds_as_hhmmss` is imported from `datetime_helpers` (line 19). So when the formatter runs, it raises `NameError: name 'unix_timestamp_to_datetime' is not defined`, which is never logged.

## Plan

### 1. Add logging and defensive handling in `_show_tooltip_for_rect`

In [interval_rects_item.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\graphics\interval_rects_item.py):

- At top of file: add `import logging` and `logger = logging.getLogger(__name__)`.
- In `_show_tooltip_for_rect` (around lines 406–411): wrap the formatter call and `QToolTip.showText` in a try/except:
  - On success: call `QToolTip.showText` as now.
  - On exception: log at `logger.exception(...)` (so full traceback is recorded), then call `QToolTip.showText(global_pos, "Tooltip error: ..." + str(e))` so the user sees a tooltip with the error message and can correlate with logs.

This makes any future formatter bug (wrong type, missing attr, etc.) visible in logs and on hover.

### 2. Fix the missing import

In [interval_rects_item.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\graphics\interval_rects_item.py) line 19, change:

- From: `from pypho_timeline.utils.datetime_helpers import format_seconds_as_hhmmss`
- To: `from pypho_timeline.utils.datetime_helpers import format_seconds_as_hhmmss, unix_timestamp_to_datetime`

After this, the default tooltip formatter will run without `NameError` and tooltips should show again.

### 3. Optional: log in datetime_helpers only if you want to trace from there

You asked to modify "the function" and referenced [datetime_helpers.py:232-247](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\datetime_helpers.py) (`datetime_to_unix_timestamp`). That function is not in the tooltip path; the failure is in the *caller* (interval_rects_item) due to the missing `unix_timestamp_to_datetime` import. No change to `datetime_to_unix_timestamp` is required for tooltip logging or this fix. If you still want logging inside datetime helpers (e.g. for invalid inputs), we can add a small amount of logging there in a follow-up; the critical change is (1) and (2) above.

### Note on semantics

Timeline epoch rects often store **relative** time (seconds from session start), not Unix timestamps. After the import fix, if your data is relative (e.g. 0, 3600), tooltips will show dates like 1970-01-01 00:00:00 and 1970-01-01 01:00:00. If you prefer start/end to be shown as relative HH:mm:ss (like duration), the formatter can be changed to use `format_seconds_as_hhmmss(start_t)` / `format_seconds_as_hhmmss(end_t)` instead of `unix_timestamp_to_datetime` + strftime; that can be a separate change once logging confirms everything else is fine.
