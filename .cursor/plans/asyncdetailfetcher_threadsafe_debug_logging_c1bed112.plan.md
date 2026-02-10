---
name: AsyncDetailFetcher threadsafe debug logging
overview: Add threadsafe, debug-friendly logging to AsyncDetailFetcher and DetailFetchWorker in pyPhoTimeline so you can observe request flow, cache keys, interval types, and thread context after the float-to-datetime change, and avoid crashes when interval values are datetime/timedelta.
todos: []
isProject: false
---

# AsyncDetailFetcher threadsafe debug logging

## Why detail rendering may have broken

After shifting intervals from relative float timestamps to absolute datetimes, several spots can fail or behave wrongly:

- **DetailFetchWorker** ([async_detail_fetcher.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\async_detail_fetcher.py) lines 33–36) formats `t_start` and `t_duration` with `f"{t_start:.3f}"` and `f"{t_duration:.3f}"`. If `t_start` is a `datetime`/`pd.Timestamp` or `t_duration` is a `timedelta`, this raises and can prevent workers from starting or log correctly.
- **BaseTrackDatasource.get_detail_cache_key** ([track_datasource.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\datasources\track_datasource.py) lines 305–306) uses `t_start:.6f` and `t_duration:.6f`. Any datasource using this default (instead of the datetime-aware override in `IntervalProvidingTrackDatasource`) will raise when intervals use datetimes.
- **TrackRenderer** ([track_renderer.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\graphics\track_renderer.py) uses similar `.3f` formatting for `t_start`/`t_duration` in debug logs (e.g. 348–349, 374–378); same risk if those values are datetime.

So the first thing to fix for **debugging** is: **never assume float in log formatting**; use a small helper that supports float, datetime, timedelta, and None.

## Thread-safe logging

Python’s `logging` is already thread-safe. To make AsyncDetailFetcher debugging reliable and clear:

1. **Include thread identity in messages**
  Add the current thread name (e.g. `threading.current_thread().name`) to each log message in AsyncDetailFetcher and DetailFetchWorker so you can see main thread vs worker threads (e.g. `QThreadPoolWorker-*`).
2. **Keep using the existing logger**
  Continue using `get_rendering_logger(__name__)` from [logging_util.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\logging_util.py). No need for a separate queue-based logger unless you add a Qt log widget; the existing `add_qt_log_handler` already uses `QueuedConnection` for thread-safe display.
3. **Optional: formatter with thread name**
  For even clearer logs, you can add a custom `Formatter` (or set `%(threadName)s` in the existing formatter in `configure_logging`) so every record shows thread name without repeating it in every call.

## Implementation plan

### 1. Safe interval formatting helper (async_detail_fetcher.py)

Add a module-level helper used only for logging (no effect on cache keys or data):

- `_format_interval_for_log(interval: pd.Series) -> str`  
Derive `t_start` and `t_duration` from `interval` (e.g. `.get('t_start', None)`, `.get('t_duration', None)`).  
Format each for display in a single string, e.g. `"t_start=..., t_duration=..."`:
  - If value is `None`: output `"?"`.
  - If `datetime`/`pd.Timestamp`: use `str(value)` or `value.isoformat()`.
  - If `timedelta`/`pd.Timedelta`: use `value.total_seconds()` then e.g. `f"{s:.3f}s"`.
  - If numeric: use `f"{float(value):.3f}"`.
  - Else: `repr(value)`.
  Catch any exception and return a fallback like `"<format error: ...>"` so logging never raises.

Use this helper everywhere in this module that currently formats `t_start`/`t_duration` for logs (worker init, and any new log points).

### 2. Add thread name to log messages

In [async_detail_fetcher.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\async_detail_fetcher.py):

- At the start of each `logger.debug`/`logger.info`/`logger.error` call (in both `DetailFetchWorker` and `AsyncDetailFetcher`), include the current thread name, e.g.:
  - `thread_name = threading.current_thread().name`
  - Prepend or append to the message, e.g. `f"[{thread_name}] ..."`.
- Add `import threading` at the top.

This gives you a clear view of which path runs on the main thread vs worker threads (fetch, queue put, process_result_queue, signal/callback).

### 3. Replace fragile formatting in DetailFetchWorker.**init**

In `DetailFetchWorker.__init__` (lines 33–36), remove the direct `f"{t_start:.3f}"` / `f"{t_duration:.3f}"` usage. Call the new helper and log a single line, e.g.:

- `logger.debug(f"DetailFetchWorker[{self.track_id}] __init__ [%s] cache_key='%s' interval=%s", thread_name, self.cache_key, _format_interval_for_log(interval))`

This prevents crashes when intervals use datetime/timedelta and makes logs consistent.

### 4. Add strategic debug log points

Add (or keep) DEBUG logs at these points, using the safe formatter and thread name:

- **fetch_detail_async**: After computing `cache_key`, log thread name, `track_id`, `cache_key`, and `_format_interval_for_log(interval)` (so you see what the main thread is requesting).
- **DetailFetchWorker.run**: Before calling `fetch_detailed_data`, log thread name, `cache_key`, and interval summary (safe format); after return, log whether result is None or type/size (as now, but ensure no float-only formatting on interval).
- **_process_result_queue**: When an item is dequeued, log thread name (should be main), `track_id`, `cache_key`, and whether `error` is set (one line per result).
- **_on_worker_finished**: Log thread name, cache store (success/error), and whether callback vs signal is used.

These points let you verify: request (main) -> worker run -> queue -> process (main) -> callback/signal (main).

### 5. Optional: thread name in formatter

If you want thread name on every record without changing every log call:

- In [logging_util.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\logging_util.py), in `configure_logging`, add `%(threadName)s` to the formatter format string, e.g.  
`'%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'`.

Then you can omit explicit thread name from individual messages in async_detail_fetcher if you prefer.

### 6. How to enable and use the logs

- Set level to DEBUG for the async_detail_fetcher module, e.g.:
  - `logging.getLogger('pypho_timeline.rendering.async_detail_fetcher').setLevel(logging.DEBUG)`, or
  - Call `configure_logging(log_level=logging.DEBUG)` so the whole rendering tree is DEBUG.
- Reproduce the timeline with detail (scroll so intervals enter view). Watch for:
  - **No "fetch_detail_async" / "cache_key" logs**: viewport or visibility not triggering fetch.
  - **Worker "run()" then exception**: crash in `fetch_detailed_data` or in interval handling (check interval type in the new safe log).
  - **"processed result" but no "_on_worker_finished" or no signal**: queue/thread mix-up (thread name will show who processed).
  - **Cache key mismatch**: e.g. request key vs key used in callback/signal (log both in fetch_detail_async and _on_worker_finished).

## Files to change


| File                                                                                                                                                        | Changes                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [pypho_timeline/rendering/async_detail_fetcher.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\rendering\async_detail_fetcher.py) | Add `threading` import; add `_format_interval_for_log(interval)`; replace float-only formatting in DetailFetchWorker.**init**; add thread name to all existing and new debug/info/error logs; add log points in fetch_detail_async, run, _process_result_queue, _on_worker_finished as above. |
| [pypho_timeline/utils/logging_util.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\logging_util.py)                         | Optional: add `%(threadName)s` to the formatter in `configure_logging` so thread is visible on every line.                                                                                                                                                                                    |


## Out of scope (but useful follow-ups)

- Fixing **BaseTrackDatasource.get_detail_cache_key** to support datetime (same logic as IntervalProvidingTrackDatasource) so datasources that don’t override it don’t raise.
- Applying the same safe-interval formatting helper (or equivalent) in **TrackRenderer** wherever it formats `t_start`/`t_duration` for debug logs, to avoid similar crashes when intervals are datetime.

