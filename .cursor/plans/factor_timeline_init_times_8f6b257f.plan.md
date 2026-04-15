---
name: Factor timeline init times
overview: Extract `SimpleTimelineWidget.__init__` time/bounds setup (lines 136–194) into a small pure utility module under `pypho_timeline/utils/`, returning a dataclass the widget unpacks—preserving behavior and avoiding new circular imports.
todos:
  - id: add-utils-module
    content: Add `timeline_widget_time_setup.py` with dataclass, coercion helpers, `build_simple_timeline_initial_time_state`, and `window_value_to_signal_float`.
    status: pending
  - id: wire-widget-init
    content: Replace `SimpleTimelineWidget.__init__` block with call + assignments + existing `SimpleTimeWindow(...)` line; set compare/flags as today.
    status: pending
  - id: delegate-signal-float
    content: Make `_window_value_to_signal_float` call the shared `window_value_to_signal_float`.
    status: pending
  - id: smoke-check
    content: Smoke import/instantiate `SimpleTimelineWidget` with float and datetime/reference variants.
    status: pending
isProject: false
---

# Factor out SimpleTimelineWidget initial time setup

## Context

The block in [`simple_timeline_widget.py`](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/widgets/simple_timeline_widget.py) (lines 136–194) does four things:

1. Resolves `reference_datetime` (lazy-import + `get_earliest_reference_datetime([], [])` when `None`).
2. Coerces `total_start_time`, `total_end_time`, and `window_start_time`: numeric + reference → absolute time; numeric without reference → `float`; otherwise `pd.Timestamp` with naive → `tz_localize('UTC')`.
3. Computes `active_window_end_time` from `window_duration` (`timedelta` vs seconds vs float stride).
4. Builds `SimpleTimeWindow`, sets `_last_applied_plot_window_*` via the same float conversion as [`_window_value_to_signal_float`](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/widgets/simple_timeline_widget.py) (lines 575–580), mirrors compare window fields, and sets `_is_updating_compare_window` / `_applying_window_from_signal`.

[`float_to_datetime`](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/utils/datetime_helpers.py) exists but **requires** a non-`None` reference and returns `datetime`, while this block deliberately keeps **float** when there is no reference and uses **`pd.Timestamp`** in the datetime branch—so behavior must be preserved as-is rather than switching callers to `float_to_datetime` without careful alignment.

## Approach

Add a new module **[`pypho_timeline/utils/timeline_widget_time_setup.py`](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/utils/timeline_widget_time_setup.py)** (no dependency on Qt/widgets—safe for import order):

| Piece | Responsibility |
|--------|----------------|
| `window_value_to_signal_float(value)` | Same logic as today: `datetime`/`pd.Timestamp` → `.timestamp()`, else `float(value)`. |
| `_coerce_timeline_boundary(value, reference_datetime)` | Single implementation of the repeated int/float vs datetime rules (used three times). |
| `_compute_active_window_end(active_window_start_time, window_duration)` | Encapsulates the `timedelta` vs `float(window_duration)` branch. |
| `build_simple_timeline_initial_time_state(...)` | Runs reference resolution, three coercions, end time, returns a **`@dataclass`** (e.g. `TimelineWidgetInitialTimeState`) with: `reference_datetime`, `total_data_start_time`, `total_data_end_time`, `active_window_start_time`, `active_window_end_time`, `last_applied_plot_window_x0`, `last_applied_plot_window_x1`. |
| `build_initial_spikes_window(...)` | Thin wrapper that instantiates **`SimpleTimeWindow`** with the four arguments the widget uses today—import `SimpleTimeWindow` from **`simple_timeline_widget`** inside this function **or** pass `SimpleTimeWindow` as a parameter from `__init__` to avoid importing the widget module from utils at module import time. |

**Circular import guard:** `timeline_widget_time_setup` must not import `simple_timeline_widget` at **module load** time. Preferred pattern: `build_simple_timeline_initial_time_state` returns only numeric/datetime state; **`__init__` keeps** `self.spikes_window = SimpleTimeWindow(...)` on the next line (only two lines of widget-local code). That avoids any `utils → widgets` import edge case.

**`__init__` replacement (conceptual):**

```python
state = build_simple_timeline_initial_time_state(
    total_start_time, total_end_time, window_start_time, window_duration, reference_datetime
)
self.reference_datetime = state.reference_datetime
self.total_data_start_time = state.total_data_start_time
# ... assign remaining fields from state ...
self.spikes_window = SimpleTimeWindow(
    self.total_data_start_time, self.total_data_end_time, window_duration, self.active_window_start_time
)
self.compare_window_start_time = self.active_window_start_time
# ... etc.
```

## Small DRY follow-up (same PR)

Change `_window_value_to_signal_float` to **delegate** to `window_value_to_signal_float` from the new module so signal conversion stays one implementation (behavior unchanged).

## What we will not do

- **Do not** add this to [`utils/__init__.py`](C:/Users/pho/repos/EmotivEpoc/ACTIVE_DEV/pyPhoTimeline/pypho_timeline/utils/__init__.py) `__all__` unless you want a public API surface (optional follow-up).
- **Do not** refactor notebooks or `TimelineBuilder` in this change unless you want them to call the new helper later (out of scope for “factor out this code”).

## Verification

- Run existing tests for `pyPhoTimeline` if present, or a minimal import smoke test: `from pypho_timeline.widgets.simple_timeline_widget import SimpleTimelineWidget` and construct with float-only vs `reference_datetime` + float vs `pd.Timestamp` args to ensure types match previous behavior.
