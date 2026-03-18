---
name: Finish time_independent_bad_channels
overview: Complete the `time_independent_bad_channels` implementation in `EEG_data.py` to build and return the output dict, then write the detected bad channels back onto the original `raw.info['bads']` so all downstream MNE stages skip them automatically.
todos:
  - id: implement-out-dict
    content: "Replace lines 422-425 in EEG_data.py: collect pyprep channel lists, compute all_bad_channels union, write to raw.info['bads'], build and return out dict"
    status: completed
isProject: false
---

# Finish `time_independent_bad_channels`

File to edit: `[PhoPyMNEHelper/src/phopymnehelper/EEG_data.py](C:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoPyMNEHelper\src\phopymnehelper\EEG_data.py)`, lines 393–425.

## What pyprep exposes after `.fit()`

- `prep.noisy_channels_original['bad_all']` — every channel flagged as bad in the original signal (before any PREP interpolation)
- `prep.interpolated_channels` — channels PREP actually interpolated
- `prep.still_noisy_channels` — channels still bad **after** PREP's own interpolation (the most critical failures)
- `prep.noisy_channels_after_interpolation` — full breakdown dict of why each channel is still bad

## How MNE communicates bad channels downstream

Setting `raw.info['bads']` is the standard MNE mechanism. Every subsequent operation (`raw.get_data()`, `pick_types`, ICA, re-reference, spectrograms, etc.) respects this list automatically.

## Definitive bad channel set

The union of `bad_channels_original` and `still_noisy_channels` is the safest conservative mask for downstream use. The returned dict exposes each sub-list so callers can inspect or narrow the set.

## Changes

Replace the dangling `prep.noisy_channels_after_interpolation` line and `return out` with:

```python
interpolated_channels = list(prep.interpolated_channels)
bad_channels_original = list(prep.noisy_channels_original['bad_all'])
still_noisy_channels = list(prep.still_noisy_channels)
noisy_channels_after_interpolation = dict(prep.noisy_channels_after_interpolation)

# Union of all detected bad channels; most downstream stages should skip these
all_bad_channels = sorted(set(bad_channels_original) | set(still_noisy_channels))

# Write back onto original raw so downstream stages see and skip them
raw.info['bads'] = sorted(set(list(raw.info.get('bads') or []) + all_bad_channels))

out = dict(
    interpolated_channels=interpolated_channels,
    bad_channels_original=bad_channels_original,
    still_noisy_channels=still_noisy_channels,
    noisy_channels_after_interpolation=noisy_channels_after_interpolation,
    all_bad_channels=all_bad_channels,
)
return out
```

The `print` statements that already exist stay unchanged.