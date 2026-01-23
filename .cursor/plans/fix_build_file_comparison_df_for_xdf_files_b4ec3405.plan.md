---
name: Fix build_file_comparison_df for XDF files
overview: Update `build_file_comparison_df` in `historical_data.py` to properly handle .xdf files by using `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file()` to extract datetime from XDF file headers, while maintaining backward compatibility with existing MNE-compatible file formats.
todos: []
isProject: false
---

## Problem Analysis

The `build_file_comparison_df` function at lines 741-792 in [`src/phoofflineeeganalysis/analysis/historical_data.py`](src/phoofflineeeganalysis/analysis/historical_data.py) currently only handles files readable by MNE's `read_raw()` function (like .fif files). When called with .xdf files, it fails because `read_raw()` doesn't support the XDF format.

## Solution

Modify the function to:

1. Detect .xdf files by checking the file suffix
2. Use `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file()` from `xdf_files.py` to extract datetime from XDF headers
3. Fallback to filename parsing if XDF header datetime is missing
4. Maintain existing behavior for non-XDF files

## Implementation Details

### Changes to `build_file_comparison_df` method:

1. **Add import** (inside the method to avoid circular imports):

- Import `LabRecorderXDF` from `phoofflineeeganalysis.analysis.xdf_files`

2. **Modify the file processing loop** (lines 767-779):

- Check if `a_file.suffix.lower() == '.xdf'`
- For .xdf files:
- Call `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file(a_xdf_file=a_file, debug_print=False)`
- Extract `file_datetime` from the returned object
- If `file_datetime` is None or extraction fails, fallback to `extract_datetime_from_filename(a_file.name)`
- Convert datetime to UTC timezone if needed
- For non-XDF files:
- Keep existing `read_raw()` logic unchanged

3. **Error handling**:

- Keep the existing try-except block to catch `ValueError`, `AttributeError`, `TypeError`
- Add handling for potential XDF-specific exceptions (KeyError if header structure is unexpected)
- Ensure both code paths produce the same output structure

### Key Implementation Points:

- The `LabRecorderXDF.init_basic_from_lab_recorder_xdf_file()` method:
- Loads XDF file using `pyxdf.load_xdf()`
- Extracts datetime from `header['info']['datetime'][0]` 
- Parses it with format `"%Y-%m-%dT%H:%M:%S%z"`
- Converts to UTC timezone
- Returns object with `file_datetime` attribute

- The datetime timestamp conversion should match existing logic:
- `start_time = meas_datetime.timestamp() if hasattr(meas_datetime, 'timestamp') else meas_datetime[0]`

- The output structure must remain identical:
- Same dictionary keys: `'src_file_name'`, `'start_t'`, `'src_file'`, `'meas_datetime'`, plus metadata dict
- Same DataFrame structure and sorting

## Files to Modify

- [`src/phoofflineeeganalysis/analysis/historical_data.py`](src/phoofflineeeganalysis/analysis/historical_data.py) - lines 741-792