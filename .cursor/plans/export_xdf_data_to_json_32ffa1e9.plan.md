---
name: Export XDF data to JSON
overview: Create a standalone function that exports loaded XDF data (MNE Raw objects and stream metadata) to a JSON file format compatible with augmented-analytics.kanaries.net. The function will export both raw time-series data and metadata in a structured format.
todos:
  - id: create_export_function
    content: Create the main export_to_json classmethod in LabRecorderXDF with function signature and basic structure
    status: completed
  - id: add_data_conversion_helpers
    content: Add helper functions to convert numpy arrays, datetime objects, and pandas types to JSON-serializable formats
    status: completed
  - id: implement_metadata_export
    content: Extract and structure metadata from stream_infos_df and MNE Raw info objects
    status: completed
    dependencies:
      - create_export_function
  - id: implement_raw_data_export
    content: Extract raw time-series data from MNE Raw objects, convert to JSON format with timestamps and channel data
    status: completed
    dependencies:
      - create_export_function
      - add_data_conversion_helpers
  - id: add_sampling_options
    content: Implement optional downsampling (max_samples_per_stream, sample_interval) for large datasets
    status: completed
    dependencies:
      - implement_raw_data_export
  - id: add_error_handling
    content: Add error handling for edge cases (empty streams, missing data, serialization errors) and progress indicators
    status: completed
    dependencies:
      - implement_metadata_export
      - implement_raw_data_export
---

# Export XDF Data to JSON for Augmented Analytics

## Overview

Create a standalone function that takes loaded XDF data from the notebook (`_out_eeg_raw` and `_out_xdf_stream_infos_df`) and exports it to a JSON file that can be imported by augmented-analytics.kanaries.net.

## Implementation Details

### Function Location

Create a new file `PhoPyMNEHelper/src/phopymnehelper/exporters/JSON_Exporter.py` following the pattern of `AiirTable_Exporter.py` (standalone functions, not class methods).

### Function Signature

```python
def export_xdf_data_to_json(eeg_raws: List[mne.io.BaseRaw], 
                            stream_infos_df: pd.DataFrame, 
                            output_path: Path, 
                            include_raw_data: bool = True, 
                            max_samples_per_stream: Optional[int] = None, 
                            sample_interval: int = 1) -> Path:
```

### Data Structure

The JSON will have a hierarchical structure:

- Top-level object with metadata and data arrays
- Each stream/file as a separate entry in a `streams` array
- Raw data exported as time-series arrays (one per channel)
- Metadata includes: file info, timestamps, channel names, sampling rates, etc.

### Key Features

1. **Raw Data Export**: Convert MNE Raw objects to JSON-serializable format

   - Extract channel data using `raw.get_data()`
   - Convert timestamps to ISO format strings
   - Handle NaN/inf values appropriately

2. **Metadata Export**: Include all relevant stream information

   - File paths, recording dates, durations
   - Channel names and types
   - Sampling rates
   - Stream metadata from `stream_infos_df`

3. **Data Sampling**: Support optional downsampling for large datasets

   - `max_samples_per_stream`: Limit number of samples per stream
   - `sample_interval`: Take every Nth sample

4. **JSON Serialization**: Handle datetime, numpy arrays, and pandas types

   - Convert datetime to ISO strings
   - Convert numpy arrays to lists
   - Handle NaN values (convert to null)

### JSON Structure

```json
{
  "metadata": {
    "export_date": "2026-02-04T...",
    "num_streams": 35,
    "total_duration_seconds": 1234.5
  },
  "streams": [
    {
      "stream_index": 0,
      "file_info": {
        "xdf_filename": "...",
        "recording_datetime": "2025-09-18T...",
        "duration_seconds": 123.4
      },
      "channel_info": {
        "channel_names": ["AF3", "F7", ...],
        "channel_types": ["eeg", "eeg", ...],
        "sampling_rate": 128.0
      },
      "data": {
        "timestamps": ["2025-09-18T...", ...],
        "timestamps_relative_seconds": [0.0, 0.0078125, ...],
        "channels": {
          "AF3": [1.23, 1.45, ...],
          "F7": [2.34, 2.56, ...],
          ...
        }
      }
    },
    ...
  ]
}
```

### Implementation Steps

1. Create new file `PhoPyMNEHelper/src/phopymnehelper/exporters/JSON_Exporter.py`
2. Add helper functions for data conversion (numpy to list, datetime to string)
3. Handle edge cases (empty streams, missing data, large files)
4. Add progress indicators for large exports
5. Include error handling for serialization issues

### Usage Example

```python
from phopymnehelper.exporters.JSON_Exporter import export_xdf_data_to_json
from pathlib import Path

output_path = Path("xdf_export.json")
export_xdf_data_to_json(
    eeg_raws=_out_eeg_raw,
    stream_infos_df=_out_xdf_stream_infos_df,
    output_path=output_path,
    include_raw_data=True,
    max_samples_per_stream=10000  # Optional: limit for large files
)
```

## Files to Create

- `PhoPyMNEHelper/src/phopymnehelper/exporters/JSON_Exporter.py` - New file with export functions

## Considerations

- Memory usage: For very large datasets, consider streaming or chunking
- File size: Large exports may need compression or sampling
- Data types: Ensure all numpy/pandas types are JSON-serializable
- Timestamps: Use ISO 8601 format for compatibility
- Missing data: Handle NaN values appropriately (convert to null in JSON)