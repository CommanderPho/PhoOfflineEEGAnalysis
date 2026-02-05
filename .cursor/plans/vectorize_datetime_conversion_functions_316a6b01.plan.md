---
name: Vectorize datetime conversion functions
overview: Add vectorized array support to `datetime_to_unix_timestamp` and `unix_timestamp_to_datetime` functions while maintaining exact same values for scalar inputs and ensuring round-trip behavior.
todos:
  - id: add_imports
    content: Add numpy and Union/List to imports in datetime_helpers.py
    status: completed
  - id: vectorize_datetime_to_unix
    content: Update datetime_to_unix_timestamp to handle np.ndarray inputs with vectorized pandas operations
    status: completed
  - id: vectorize_unix_to_datetime
    content: Update unix_timestamp_to_datetime to handle np.ndarray inputs with vectorized pandas operations
    status: completed
  - id: verify_exact_values
    content: Ensure vectorized versions produce exactly same values as scalar versions for round-trip tests
    status: completed
---

# Vectorize datetime_to_unix_timestamp and unix_timestamp_to_datetime

## Overview

Add array support to both functions in [`pyPhoTimeline/pypho_timeline/utils/datetime_helpers.py`](pyPhoTimeline/pypho_timeline/utils/datetime_helpers.py) (lines 228-262) using pandas vectorization, ensuring exact same values as scalar versions.

## Changes

### 1. Add numpy import

- Add `import numpy as np` to imports section
- Add `Union, List` to typing imports

### 2. Update `datetime_to_unix_timestamp` (line 228)

- Change signature: `dt: Union[datetime, np.ndarray]` → `Union[float, np.ndarray]`
- For arrays: Use `pd.to_datetime(array).view('int64') / 1e9` or `pd.Series(array).apply(lambda x: x.timestamp() if x.tzinfo is None else x.replace(tzinfo=timezone.utc).timestamp())` to ensure exact same behavior
- Actually, simpler: convert to pandas DatetimeIndex, ensure UTC, then use `.astype('int64') / 1e9` for nanoseconds to seconds
- Or use: `pd.Series(dt_array).apply(lambda x: x.timestamp() if hasattr(x, 'timestamp') else pd.Timestamp(x).timestamp())` but this is not vectorized
- Best approach: `pd.to_datetime(dt_array)` creates DatetimeIndex, ensure UTC with `.tz_localize('UTC')` or `.tz_convert('UTC')`, then `.astype('int64') / 1e9` converts nanoseconds to seconds
- Return `List[float] `for arrays, `float` for scalars

### 3. Update `unix_timestamp_to_datetime` (line 247)

- Change signature: `ts: Union[float, np.ndarray]` → `Union[datetime, List[datetime]]`
- For arrays: Use `pd.to_datetime(ts_array, unit='s', utc=True)` which produces DatetimeIndex, then convert to list of datetime objects
- Return `List[datetime] `for arrays, `datetime` for scalars

### 4. Maintain exact value matching

- Use pandas operations that internally use the same `timestamp()` and `fromtimestamp()` methods
- Ensure timezone handling matches exactly (naive → UTC, aware → convert to UTC)
- Test that round-trip behavior is preserved

## Implementation Details

### datetime_to_unix_timestamp vectorization:

```python
if isinstance(dt, np.ndarray):
    # Convert to pandas DatetimeIndex for vectorized processing
    dt_series = pd.to_datetime(dt)
    # Ensure UTC timezone
    if dt_series.tz is None:
        dt_series = dt_series.tz_localize('UTC')
    else:
        dt_series = dt_series.tz_convert('UTC')
    # Convert to Unix timestamps (nanoseconds to seconds)
    timestamps = (dt_series.astype('int64') / 1e9).tolist()
    return timestamps
```

### unix_timestamp_to_datetime vectorization:

```python
if isinstance(ts, np.ndarray):
    # Use pandas vectorized conversion
    datetimes = pd.to_datetime(ts, unit='s', utc=True).tolist()
    # Convert Timestamp objects to datetime objects
    return [dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt for dt in datetimes]
```

## Testing

- Existing tests in [`pyPhoTimeline/tests/test_datetime_helpers.py`](pyPhoTimeline/tests/test_datetime_helpers.py) should continue to pass
- Scalar behavior must remain identical
- Array inputs should produce same values as list comprehension equivalent