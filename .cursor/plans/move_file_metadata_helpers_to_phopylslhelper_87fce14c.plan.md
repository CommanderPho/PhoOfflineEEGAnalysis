---
name: Move File Metadata Helpers to PhoPyLSLhelper
overview: Move all file metadata parsing classes (BaseFileMetadataParser, VideoMetadataParser, DataFileMetadataParser) from PhoOfflineEEGAnalysis to PhoPyLSLhelper, update dependencies, bump major version, and update all imports across both projects.
todos:
  - id: update_phopylslhelper_dependencies
    content: "Update PhoPyLSLhelper/pyproject.toml: add pandas, attrs, opencv-python dependencies and bump version to 1.0.0"
    status: completed
  - id: move_file_metadata
    content: Copy file_metadata.py to PhoPyLSLhelper/src/phopylslhelper/file_metadata.py
    status: completed
    dependencies:
      - update_phopylslhelper_dependencies
  - id: move_video_metadata
    content: Copy video_metadata.py to PhoPyLSLhelper/src/phopylslhelper/video_metadata.py and update import to use phopylslhelper.file_metadata
    status: completed
    dependencies:
      - move_file_metadata
  - id: move_data_file_metadata
    content: Copy data_file_metadata.py to PhoPyLSLhelper/src/phopylslhelper/data_file_metadata.py and update import to use phopylslhelper.file_metadata
    status: completed
    dependencies:
      - move_file_metadata
  - id: update_phopylslhelper_init
    content: Update PhoPyLSLhelper/src/phopylslhelper/__init__.py to export the three parser classes
    status: completed
    dependencies:
      - move_file_metadata
      - move_video_metadata
      - move_data_file_metadata
  - id: update_phoofflineeeganalysis_imports
    content: Update all imports in PhoOfflineEEGAnalysis to use phopylslhelper instead of phoofflineeeganalysis.analysis
    status: completed
    dependencies:
      - move_video_metadata
      - move_data_file_metadata
  - id: update_pypho_timeline_imports
    content: Update imports in pyPhoTimeline to use phopylslhelper instead of phoofflineeeganalysis.analysis
    status: completed
    dependencies:
      - move_video_metadata
  - id: update_pypho_timeline_dependencies
    content: Add phopylslhelper>=1.0.0 to pyPhoTimeline/pyproject.toml dependencies
    status: completed
    dependencies:
      - update_phopylslhelper_dependencies
  - id: delete_original_files
    content: Delete original file_metadata.py, video_metadata.py, and data_file_metadata.py from PhoOfflineEEGAnalysis
    status: completed
    dependencies:
      - update_phoofflineeeganalysis_imports
      - update_pypho_timeline_imports
  - id: verify_phoofflineeeganalysis_dependency
    content: Verify PhoOfflineEEGAnalysis/pyproject.toml has phopylslhelper>=1.0.0 in dependencies
    status: completed
    dependencies:
      - update_phopylslhelper_dependencies
---

# Move File Metadata Helpers to PhoPyLSLhelper

## Overview

Move all file metadata parsing and caching utilities from `PhoOfflineEEGAnalysis` to `PhoPyLSLhelper` to create a shared utility package that both `phoofflineeeganalysis` and `pypho_timeline` can use without circular dependencies.

## Files to Move

1. **PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/file_metadata.py**

   - Contains: `BaseFileMetadataParser`
   - Move to: `PhoPyLSLhelper/src/phopylslhelper/file_metadata.py`

2. **PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/video_metadata.py**

   - Contains: `VideoMetadataParser`
   - Move to: `PhoPyLSLhelper/src/phopylslhelper/video_metadata.py`

3. **PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/data_file_metadata.py**

   - Contains: `DataFileMetadataParser`
   - Move to: `PhoPyLSLhelper/src/phopylslhelper/data_file_metadata.py`

## Implementation Steps

### Step 1: Update PhoPyLSLhelper Dependencies

Update `PhoPyLSLhelper/pyproject.toml`:

- Add `pandas>=1.5.3,<3.0.0` (required for all parsers)
- Add `attrs>=22.2.0,<23` (required for all parsers)
- Add `opencv-python>=4.5.0` (required for VideoMetadataParser)
- Note: `mne` and `pyxdf` remain optional (only needed for DataFileMetadataParser .fif/.xdf support)

### Step 2: Update PhoPyLSLhelper Version

Update version in `PhoPyLSLhelper/pyproject.toml`:

- Current: `0.1.3`
- New: `1.0.0` (major version bump to reflect significant new functionality)

### Step 3: Move Files to PhoPyLSLhelper

1. Copy `file_metadata.py` to `PhoPyLSLhelper/src/phopylslhelper/file_metadata.py`

   - No import changes needed (only uses standard library + pandas + attrs)

2. Copy `video_metadata.py` to `PhoPyLSLhelper/src/phopylslhelper/video_metadata.py`

   - Update import: `from phoofflineeeganalysis.analysis.file_metadata` → `from phopylslhelper.file_metadata`

3. Copy `data_file_metadata.py` to `PhoPyLSLhelper/src/phopylslhelper/data_file_metadata.py`

   - Update import: `from phoofflineeeganalysis.analysis.file_metadata` → `from phopylslhelper.file_metadata`
   - Keep optional imports for `mne` and `LabRecorderXDF` (these will be available when used in PhoOfflineEEGAnalysis context)

### Step 4: Update PhoPyLSLhelper **init**.py

Update `PhoPyLSLhelper/src/phopylslhelper/__init__.py` to export:

```python
from phopylslhelper.file_metadata import BaseFileMetadataParser
from phopylslhelper.video_metadata import VideoMetadataParser
from phopylslhelper.data_file_metadata import DataFileMetadataParser

__all__ = [
    'BaseFileMetadataParser',
    'VideoMetadataParser',
    'DataFileMetadataParser',
]
```

### Step 5: Update Imports in PhoOfflineEEGAnalysis

Update all files in `PhoOfflineEEGAnalysis` that import these classes:

1. **video_metadata.py** (if it still exists after move):

   - Remove file (already moved)

2. **data_file_metadata.py** (if it still exists after move):

   - Remove file (already moved)

3. **UI/timeline/historical_data_timeline.py**:

   - `from phoofflineeeganalysis.analysis.video_metadata` → `from phopylslhelper.video_metadata`

4. **UI/timeline/init.py**:

   - `from phoofflineeeganalysis.analysis.video_metadata` → `from phopylslhelper.video_metadata`

5. **Notebooks** (examples_jupyter/*.ipynb):

   - Update import statements in notebook cells

### Step 6: Update Imports in pyPhoTimeline

Update files in `pyPhoTimeline`:

1. **rendering/datasources/specific/video.py**:

   - `from phoofflineeeganalysis.analysis.video_metadata` → `from phopylslhelper.video_metadata`

2. **rendering/datasources/specific/init.py**:

   - `from phoofflineeeganalysis.analysis.video_metadata` → `from phopylslhelper.video_metadata`

### Step 7: Handle DataFileMetadataParser Optional Dependencies

For `DataFileMetadataParser`, the optional imports (`mne`, `LabRecorderXDF`) should remain as optional:

- When used in `PhoOfflineEEGAnalysis`, these will be available
- When used elsewhere, the parser will gracefully fall back to filename parsing
- This maintains backward compatibility

### Step 8: Delete Original Files

After confirming all imports are updated:

- Delete `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/file_metadata.py`
- Delete `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/video_metadata.py`
- Delete `PhoOfflineEEGAnalysis/src/phoofflineeeganalysis/analysis/data_file_metadata.py`

### Step 9: Update PhoOfflineEEGAnalysis Dependencies

Update `PhoOfflineEEGAnalysis/pyproject.toml`:

- Ensure `phopylslhelper>=1.0.0` is in dependencies (or update if already present)

### Step 10: Update pyPhoTimeline Dependencies

Update `pyPhoTimeline/pyproject.toml`:

- Add `phopylslhelper>=1.0.0` to dependencies

## Key Considerations

1. **Optional Dependencies**: `DataFileMetadataParser` uses optional imports for `mne` and `LabRecorderXDF`. These should remain optional to avoid forcing heavy dependencies on all users.

2. **Backward Compatibility**: The API remains the same, only import paths change. This is a non-breaking change for end users.

3. **Circular Dependencies**: Moving to `PhoPyLSLhelper` avoids circular dependencies since it's a lower-level utility package.

4. **Version Bump**: Major version bump (0.1.3 → 1.0.0) reflects the significant new functionality addition.

## Files to Modify

1. **PhoPyLSLhelper**:

   - `pyproject.toml` (dependencies + version)
   - `src/phopylslhelper/__init__.py` (exports)
   - `src/phopylslhelper/file_metadata.py` (new)
   - `src/phopylslhelper/video_metadata.py` (new)
   - `src/phopylslhelper/data_file_metadata.py` (new)

2. **PhoOfflineEEGAnalysis**:

   - `pyproject.toml` (ensure phopylslhelper dependency)
   - `src/phoofflineeeganalysis/analysis/UI/timeline/historical_data_timeline.py`
   - `src/phoofflineeeganalysis/analysis/UI/timeline/__init__.py`
   - `examples_jupyter/*.ipynb` (notebook imports)
   - Delete: `file_metadata.py`, `video_metadata.py`, `data_file_metadata.py`

3. **pyPhoTimeline**:

   - `pyproject.toml` (add phopylslhelper dependency)
   - `pypho_timeline/rendering/datasources/specific/video.py`
   - `pypho_timeline/rendering/datasources/specific/__init__.py`