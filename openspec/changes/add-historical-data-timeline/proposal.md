## Why
Analysts need a unified, high-performance timeline visualization to explore and compare multiple data modalities (EEG, motion, video, annotations, transcripts) across recording sessions. The timeline enables quick identification of temporal gaps, overlaps, and correlations between different data sources, which is critical for multi-modality analysis and session alignment.

## What Changes
- Add a PyQtGraph-based timeline widget (`TimelineWidget`) that displays multiple synchronized tracks, each representing a different data modality.
- Implement base `TrackWidget` class with extensible architecture for modality-specific tracks.
- Add specialized track implementations:
  - `VideoMetadataTrack`: Video recording intervals from `VideoMetadataParser`
  - `EEGRecordingTrack`: EEG session intervals with duration
  - `MotionRecordingTrack`: Motion sensor recording intervals
  - `PhoLogTrack`: PHO_LOG_TO_LSL annotation intervals
  - `WhisperTrack`: Whisper transcript intervals
  - `XDFStreamTrack`: Generic XDF stream intervals with flexible datetime/duration handling
- Each track displays rectangles representing recording intervals (start, end) with modality-specific colors.
- All tracks share a synchronized datetime x-axis with proper formatting via PyQtGraph's `DateAxisItem`.
- Implement performance optimizations:
  - Interval caching as numpy arrays for fast filtering
  - Debounced updates (50ms) to prevent excessive redraws during pan/zoom
  - Viewport culling: only visible intervals are rendered
- Add left-edge labels for each track showing the modality name (e.g., "Videos", "EEG").
- Support mouse wheel zoom and pan with synchronized updates across all tracks.
- Provide factory functions:
  - `create_timeline_from_xdf_streams()`: Auto-create tracks from XDF stream info DataFrame
  - `create_timeline_widget()`: Manual track creation
- Add `add_tracks_from_xdf_streams()` method to `TimelineWidget` for bulk track creation from stream metadata.

## Impact
- Affected specs: `historical-data-timeline` (new capability)
- Affected code: 
  - New module: `src/phoofflineeeganalysis/analysis/UI/historical_data_timeline.py`
  - Integration point: `VideoMetadataParser` (already exists)
- Dependencies: PyQtGraph (already in project dependencies)
- No breaking changes expected; feature is additive.

