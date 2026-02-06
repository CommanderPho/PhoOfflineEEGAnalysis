---
name: Add stream allowlist/blocklist filtering
overview: Add allowlist/blocklist support to TimelineBuilder init functions to filter XDF streams before processing. Filtering uses regex patterns and is mutually exclusive (only one can be specified at a time).
todos:
  - id: add_filter_helper
    content: Add _filter_streams_by_name helper method to TimelineBuilder class for regex-based stream filtering
    status: completed
  - id: update_build_from_xdf_files
    content: Add stream_allowlist/stream_blocklist parameters to build_from_xdf_files and apply filtering after loading streams
    status: completed
    dependencies:
      - add_filter_helper
  - id: update_build_from_xdf_file
    content: Add stream_allowlist/stream_blocklist parameters to build_from_xdf_file and pass through to build_from_xdf_files
    status: completed
    dependencies:
      - update_build_from_xdf_files
  - id: update_build_from_streams
    content: Add stream_allowlist/stream_blocklist parameters to build_from_streams and apply filtering before processing
    status: completed
    dependencies:
      - add_filter_helper
  - id: update_build_from_eeg_raw
    content: Add stream_allowlist/stream_blocklist parameters to build_from_eeg_raw_and_stream_info and filter stream_infos_df before processing
    status: completed
    dependencies:
      - add_filter_helper
---

# Add Stream Allowlist/Blocklist Filtering to TimelineBuilder

## Overview

Add allowlist/blocklist filtering support to all TimelineBuilder initialization methods that process XDF streams. This will allow users to specify which streams to load and which to skip, reducing unnecessary processing and track creation.

## Implementation Details

### 1. Add Helper Method for Stream Filtering

Create a new private method `_filter_streams_by_name` in [timeline_builder.py](pyPhoTimeline/pypho_timeline/timeline_builder.py) that:

- Takes a list of streams and either an allowlist or blocklist (mutually exclusive)
- Uses regex matching to filter stream names
- Returns filtered list of streams
- Logs which streams were filtered

### 2. Update `build_from_xdf_files` Method

Modify [build_from_xdf_files](pyPhoTimeline/pypho_timeline/timeline_builder.py) (line 122) to:

- Add optional parameters: `stream_allowlist: Optional[List[str]] = None` and `stream_blocklist: Optional[List[str]] = None`
- Validate that only one of allowlist/blocklist is provided
- Filter streams after loading from XDF files (line ~164) but before processing
- Apply filtering to each file's streams individually
- Update docstring to document the new parameters

### 3. Update `build_from_streams` Method

Modify [build_from_streams](pyPhoTimeline/pypho_timeline/timeline_builder.py) (line 280) to:

- Add optional parameters: `stream_allowlist: Optional[List[str]] = None` and `stream_blocklist: Optional[List[str]] = None`
- Validate that only one of allowlist/blocklist is provided
- Filter streams before calling `_process_xdf_streams` (line ~295)
- Update docstring to document the new parameters

### 4. Update `build_from_eeg_raw_and_stream_info` Method

Modify [build_from_eeg_raw_and_stream_info](pyPhoTimeline/pypho_timeline/timeline_builder.py) (line 321) to:

- Add optional parameters: `stream_allowlist: Optional[List[str]] = None` and `stream_blocklist: Optional[List[str]] = None`
- Validate that only one of allowlist/blocklist is provided
- Filter `stream_infos_df` rows based on the 'name' column before processing (line ~392)
- Update docstring to document the new parameters

### 5. Update `build_from_xdf_file` Method

Modify [build_from_xdf_file](pyPhoTimeline/pypho_timeline/timeline_builder.py) (line 104) to:

- Add optional parameters: `stream_allowlist: Optional[List[str]] = None` and `stream_blocklist: Optional[List[str]] = None`
- Pass these parameters through to `build_from_xdf_files` (line ~119)
- Update docstring to document the new parameters

## Technical Notes

- Use Python's `re` module for regex matching
- Stream names are extracted from `stream['info']['name'][0]` for XDF streams
- For `build_from_eeg_raw_and_stream_info`, stream names come from `stream_infos_df['name']` column
- Filtering happens early in the pipeline to avoid unnecessary processing
- Print informative messages about which streams were filtered

## Example Usage

```python
builder = TimelineBuilder()

# Only load streams matching patterns
timeline = builder.build_from_xdf_files(
    xdf_file_paths=[Path("data.xdf")],
    stream_allowlist=[r"EEG.*", r"MOTION.*"]
)

# Exclude specific streams
timeline = builder.build_from_xdf_files(
    xdf_file_paths=[Path("data.xdf")],
    stream_blocklist=[r".*Logger.*", r".*Event.*"]
)
```