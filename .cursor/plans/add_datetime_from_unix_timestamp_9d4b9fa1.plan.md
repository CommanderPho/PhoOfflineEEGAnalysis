---
name: Add datetime_from_unix_timestamp
overview: "Add a new function `datetime_from_unix_timestamp(ts: float) -> datetime` that is the reciprocal of `datetime_to_unix_timestamp`: round-tripping through both functions preserves the instant (UTC)."
todos: []
isProject: false
---

# Add reciprocal `datetime_from_unix_timestamp`

## Goal

Add **only** a new function (no changes to existing code):

- **Name:** `datetime_from_unix_timestamp`
- **Behavior:** For any `dt`, `datetime_from_unix_timestamp(datetime_to_unix_timestamp(dt))` must represent the same instant as `dt` (in UTC). Equivalently, for any float `ts`, `datetime_to_unix_timestamp(datetime_from_unix_timestamp(ts)) == ts`.

## Implementation

**File:** [pypho_timeline/utils/datetime_helpers.py](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\pyPhoTimeline\pypho_timeline\utils\datetime_helpers.py)

**Location:** Insert the new function immediately after `datetime_to_unix_timestamp` (after line 247), with two blank lines before the next function `datetime_to_float` (per your style).

**Implementation:**

- **Signature:** `def datetime_from_unix_timestamp(ts: float) -> datetime:`
- **Body:** `return datetime.fromtimestamp(ts, tz=timezone.utc)`
- **Docstring:** State that it converts a Unix timestamp (seconds since 1970-01-01 UTC) to a timezone-aware UTC `datetime`, and that it is the reciprocal of `datetime_to_unix_timestamp` (round-trip preserves the instant).

**Rationale:** `datetime.to_timestamp()` (used in `datetime_to_unix_timestamp`) returns seconds since epoch in UTC for aware datetimes (and for naive we treat as UTC). The inverse is `datetime.fromtimestamp(ts, tz=timezone.utc)`, which is already used elsewhere in this file (e.g. lines 35, 105). No new imports are required (`datetime`, `timezone` already imported).

## Summary


| Action          | Detail                                     |
| --------------- | ------------------------------------------ |
| Add             | One new function between lines 247 and 249 |
| Leave unchanged | All existing functions and imports         |


