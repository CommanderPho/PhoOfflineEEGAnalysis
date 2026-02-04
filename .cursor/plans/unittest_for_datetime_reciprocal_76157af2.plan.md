---
name: Unittest for datetime reciprocal
overview: Add a new unittest-based test module that verifies the reciprocal/identity relationship between `datetime_to_unix_timestamp` and `datetime_from_unix_timestamp` (round-trip in both directions).
todos: []
isProject: false
---

# Unittest for datetime_to_unix_timestamp / datetime_from_unix_timestamp reciprocal

## Goal

Add **unittest**-based tests that confirm:

1. **Timestamp → datetime → timestamp:** For any float `ts`, `datetime_to_unix_timestamp(datetime_from_unix_timestamp(ts)) == ts`.
2. **Datetime → timestamp → datetime:** For any `dt`, `datetime_from_unix_timestamp(datetime_to_unix_timestamp(dt))` represents the same instant as `dt` (naive datetimes treated as UTC per existing behavior).

## Test file location and layout

- **Path:** [tests/test_datetime_helpers.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\tests\test_datetime_helpers.py) (new file).
- **Directory:** Create `tests/` at pyPhoTimeline project root (no existing tests in the repo).
- **Style:** One test class (e.g. `TestDatetimeUnixTimestampReciprocal`) subclassing `unittest.TestCase`; use `unittest` only (no pytest), per your request.

## Test cases


| Test method                            | What it checks                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_timestamp_roundtrip_identity`    | For several float Unix timestamps (e.g. `0`, `1.5`, `1738646400.25`), assert `datetime_to_unix_timestamp(datetime_from_unix_timestamp(ts))` equals `ts`. Use `assertAlmostEqual` for float comparison.                                                                                                                                                                                                                                 |
| `test_datetime_roundtrip_same_instant` | For (1) a naive UTC-equivalent datetime, (2) an aware UTC datetime, and optionally (3) an aware non-UTC datetime, assert that `datetime_from_unix_timestamp(datetime_to_unix_timestamp(dt))` represents the same instant: e.g. compare `timestamp()` of the round-tripped datetime to `datetime_to_unix_timestamp(dt)` (or to `dt.replace(tzinfo=timezone.utc).timestamp()` for naive). Use `assertAlmostEqual` on the two timestamps. |


## Implementation details

- **Imports:** `import unittest` and `from datetime import datetime, timezone`; import `datetime_to_unix_timestamp` and `datetime_from_unix_timestamp` from `pypho_timeline.utils.datetime_helpers`.
- **Runner:** Tests can be run from project root with:  
`uv run python -m unittest tests.test_datetime_helpers`  
(or `python -m unittest discover -s tests` if you add more test modules later). No change to `pyproject.toml` is required; `unittest` is stdlib.

## Summary

- Add new file: `tests/test_datetime_helpers.py` with one test class and the two test methods above.
- No edits to [datetime_helpers.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\datetime_helpers.py) or to `pyproject.toml`.
- Optional: add empty `tests/__init__.py` if you want `tests` to be a package (not strictly required for `python -m unittest tests.test_datetime_helpers`).

