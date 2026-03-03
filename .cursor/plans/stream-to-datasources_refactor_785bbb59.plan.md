---
name: Stream-to-datasources refactor
overview: Move the three stream-processing functions and their helper constants from `simple_timeline_widget.py` into a new module `pypho_timeline/rendering/datasources/stream_to_datasources.py`, then update all imports and usages so the widget and TimelineBuilder use the new module.
todos: []
isProject: false
---

# Refactor stream-processing code to stream_to_datasources.py

## Scope

**Move from** [pypho_timeline/widgets/simple_timeline_widget.py](pypho_timeline/widgets/simple_timeline_widget.py) (lines 294-793):

- **Constants:** `modality_channels_dict`, `modality_sfreq_dict`, `modality_channels_normalization_mode_dict` (depend on `ChannelNormalizationMode` from `rendering.helpers`).
- **Functions:** `merge_streams_by_name`, `perform_process_single_xdf_file_all_streams`, `perform_process_all_streams_multi_xdf` (with their `@function_attributes` decorators).

**Create:** [pypho_timeline/rendering/datasources/stream_to_datasources.py](pypho_timeline/rendering/datasources/stream_to_datasources.py) (new file).

**Update imports/usages in:**

- [pypho_timeline/widgets/simple_timeline_widget.py](pypho_timeline/widgets/simple_timeline_widget.py) — remove moved code; add re-exports from the new module so existing `from pypho_timeline.widgets.simple_timeline_widget import ...` still works (or rely on widgets/**init**.py to import from the new place).
- [pypho_timeline/widgets/**init**.py](pypho_timeline/widgets/__init__.py) — change lazy import to load from `stream_to_datasources` instead of `simple_timeline_widget` for the processing functions and modality dicts.
- [pypho_timeline/timeline_builder.py](pypho_timeline/timeline_builder.py) — import `perform_process_single_xdf_file_all_streams` and `perform_process_all_streams_multi_xdf` from the new module (or keep importing from `widgets` if widgets re-exports them).

## 1. New module: `stream_to_datasources.py`

**Location:** `pypho_timeline/rendering/datasources/stream_to_datasources.py`

**Imports to add** (no circular risk: this module will not be imported by `track_datasource` or `specific/eeg`):

- `numpy`, `pandas`, `Path`, `List`, `Dict`, `Tuple`, `Optional`
- `pyqtgraph as pg`
- `float_to_datetime`, `datetime_to_unix_timestamp`, `unix_timestamp_to_datetime`, `get_reference_datetime_from_xdf_header` from `pypho_timeline.utils.datetime_helpers`
- `IntervalProvidingTrackDatasource` from `pypho_timeline.rendering.datasources.track_datasource`
- `MotionTrackDatasource`, `EEGTrackDatasource` from `pypho_timeline.rendering.datasources.specific` (or `.specific.eeg` / `.specific.motion`)
- `ChannelNormalizationMode` from `pypho_timeline.rendering.helpers`
- `DataframePlotDetailRenderer` from `pypho_timeline.rendering.detail_renderers.generic_plot_renderer` (inside the branch that uses it)
- `LogTextDataFramePlotDetailRenderer` from `pypho_timeline.rendering.detail_renderers.log_text_plot_renderer` (inside the branch that uses it)
- `function_attributes`: use `try: from pyphocorehelpers.function_helpers import function_attributes` / `except ImportError: function_attributes = lambda **kw: lambda f: f` so the decorator is optional.

**Contents (in order):**

1. Docstring for the module.
2. Imports above.
3. The three constants (`modality_channels_dict`, `modality_sfreq_dict`, `modality_channels_normalization_mode_dict`) — same definitions as current, using `ChannelNormalizationMode`.
4. `merge_streams_by_name` (no decorator).
5. `perform_process_single_xdf_file_all_streams` with `@function_attributes(...)` and body unchanged (references the three constants in the same module).
6. `perform_process_all_streams_multi_xdf` with `@function_attributes(...)` and body unchanged; keep the `from phopymnehelper.historical_data import HistoricalData` inside the function.
7. `__all__ = ['modality_channels_dict', 'modality_sfreq_dict', 'modality_channels_normalization_mode_dict', 'merge_streams_by_name', 'perform_process_single_xdf_file_all_streams', 'perform_process_all_streams_multi_xdf']`.

**Copy-paste:** Use the exact code from lines 297-322 (constants) and 325-791 (three functions) of `simple_timeline_widget.py`; only add the new imports and the optional `function_attributes` fallback.

## 2. Widget: remove moved code and re-export

**In** [pypho_timeline/widgets/simple_timeline_widget.py](pypho_timeline/widgets/simple_timeline_widget.py):

- Delete lines 294-793 (the comment block "Begin Testing/Building", the three constants, and the three functions).
- Add at the end of the file (before any other trailing code) re-exports so code that imports from the widget file still works:

```python
# Re-export stream-to-datasources processing for backward compatibility
from pypho_timeline.rendering.datasources.stream_to_datasources import (
    modality_channels_dict,
    modality_sfreq_dict,
    modality_channels_normalization_mode_dict,
    merge_streams_by_name,
    perform_process_single_xdf_file_all_streams,
    perform_process_all_streams_multi_xdf,
)
```

This keeps `simple_timeline_widget` as a single place to get both the widget and the processing helpers without forcing every caller to change its import path.

## 3. Widgets package: optional adjustment

**In** [pypho_timeline/widgets/**init**.py](pypho_timeline/widgets/__init__.py):

- **Option A (recommended):** In `_lazy_import_simple_timeline()`, keep importing `SimpleTimelineWidget` from `simple_timeline_widget`; import `modality_channels_dict`, `modality_sfreq_dict`, `perform_process_single_xdf_file_all_streams`, and `perform_process_all_streams_multi_xdf` from `pypho_timeline.rendering.datasources.stream_to_datasources`. Return the same 5-tuple. This way the lazy loader does not pull in the heavy widget module just to get the processing functions.
- **Option B:** Leave `_lazy_import_simple_timeline()` importing all five from `simple_timeline_widget`; since the widget file will re-export from `stream_to_datasources`, behavior stays the same.

Fix the existing typo in the import list: line 24 currently says `perform_process_single_xdf_file_all_streams_multi_xdf` (wrong name); the return on line 25 correctly uses `perform_process_all_streams_multi_xdf`. Unify to the correct name `perform_process_all_streams_multi_xdf` in the import list.

## 4. TimelineBuilder: import source

**In** [pypho_timeline/timeline_builder.py](pypho_timeline/timeline_builder.py):

- Keep the current line 16: `from pypho_timeline.widgets import SimpleTimelineWidget, perform_process_single_xdf_file_all_streams, perform_process_all_streams_multi_xdf`. No change required if widgets/**init**.py continues to expose these (via Option A or B above).

Alternatively, change to import the two processing functions from the new module directly:

`from pypho_timeline.rendering.datasources.stream_to_datasources import perform_process_single_xdf_file_all_streams, perform_process_all_streams_multi_xdf` and keep `from pypho_timeline.widgets import SimpleTimelineWidget`. This reduces coupling of TimelineBuilder to the widgets package for processing.

**Recommendation:** Use the direct import from `stream_to_datasources` in `timeline_builder.py` for the two functions; keep `SimpleTimelineWidget` from `widgets`. Then `widgets/__init__.py` can still re-export the processing symbols for any other callers.

## 5. Optional: datasources package export

**In** [pypho_timeline/rendering/datasources/**init**.py](pypho_timeline/rendering/datasources/__init__.py):

- Optionally add to `__all_`_ and import: `modality_channels_dict`, `modality_sfreq_dict`, `perform_process_single_xdf_file_all_streams`, `perform_process_all_streams_multi_xdf`, `merge_streams_by_name` from `stream_to_datasources` so that `from pypho_timeline.rendering.datasources import perform_process_all_streams_multi_xdf` works. Not required if all consumers use `stream_to_datasources` or `widgets` explicitly.

## 6. Verification

- After edits, run a quick test that builds a timeline from XDF (e.g. from `__main__.py` or a test that calls `TimelineBuilder().build_from_xdf_files(...)` or `build_from_streams(...)` which calls `perform_process_single_xdf_file_all_streams`). Ensure no `ImportError` or `AttributeError`.
- Confirm no remaining references in the codebase to the processing functions or modality dicts from `simple_timeline_widget` other than the re-exports we added.

## Summary of file changes


| File                                                            | Action                                                                                                               |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `pypho_timeline/rendering/datasources/stream_to_datasources.py` | **Create**: constants + 3 functions with imports and optional `function_attributes`.                                 |
| `pypho_timeline/widgets/simple_timeline_widget.py`              | **Delete** lines 294-793; **add** re-exports from `stream_to_datasources`.                                           |
| `pypho_timeline/widgets/__init__.py`                            | **Fix** typo in import name; **optionally** load processing symbols from `stream_to_datasources` in the lazy loader. |
| `pypho_timeline/timeline_builder.py`                            | **Optionally** import the two process functions from `stream_to_datasources` instead of `widgets`.                   |
| `pypho_timeline/rendering/datasources/__init__.py`              | **Optionally** re-export the new module’s public symbols.                                                            |


