## Context
The historical data timeline feature provides a unified visualization for multiple data modalities recorded during EEG sessions. Analysts need to quickly identify temporal relationships, gaps, and overlaps between video recordings, EEG sessions, motion data, annotations, and transcripts. The timeline must handle large datasets efficiently while maintaining responsive interactions.

## Goals / Non-Goals

### Goals
- High-performance rendering of hundreds to thousands of recording intervals
- Synchronized datetime x-axis across all tracks for easy temporal comparison
- Extensible architecture for adding new modality track types
- Smooth zoom and pan interactions with responsive updates
- Clear visual distinction between different modality types

### Non-Goals
- Real-time data streaming (this is for offline analysis)
- Interactive editing of intervals (read-only visualization)
- Export to static images (focus on interactive widget)
- Integration with MNE browser (separate tool)

## Decisions

### Decision: Use PyQtGraph for Rendering
**Rationale**: PyQtGraph provides high-performance OpenGL-accelerated rendering, native datetime axis support via `DateAxisItem`, and efficient handling of large datasets. It integrates seamlessly with PyQt5 already used in the project.

**Alternatives considered**:
- Matplotlib: Too slow for large datasets, less interactive
- HoloViews/Bokeh: Better for web, but PyQt5 desktop app needs native widgets
- Custom QPainter: More control but significant development overhead

### Decision: Caching Intervals as Numpy Arrays
**Rationale**: Converting datetime intervals to numpy arrays of timestamps enables vectorized boolean masking for fast filtering. This provides O(n) filtering vs O(n²) with Python list comprehensions.

**Implementation**: `_all_intervals_ts` stores `(start_ts, end_ts)` pairs as `numpy.ndarray` with shape `(n_intervals, 2)`.

### Decision: Debounced Updates
**Rationale**: Pan/zoom operations trigger rapid x-range change events. Debouncing with 50ms timer prevents excessive redraws while maintaining responsive feel.

**Implementation**: `QTimer` with `setSingleShot(True)` and 50ms timeout accumulates rapid events and performs single update.

### Decision: Viewport Culling
**Rationale**: Only intervals overlapping the visible time range are rendered. For datasets spanning days/weeks, this dramatically reduces rendering load.

**Implementation**: Numpy boolean mask: `mask = (intervals[:, 1] >= start_ts) & (intervals[:, 0] <= end_ts)`

### Decision: Base Class + Specialized Tracks
**Rationale**: Each modality has different DataFrame column names and duration handling. Base `TrackWidget` provides common rendering logic; subclasses implement `_get_recording_intervals()` for data extraction.

**Benefits**: 
- Easy to add new track types
- Consistent rendering behavior
- Modality-specific colors via `update_display()` override

### Decision: Left-Edge Labels
**Rationale**: Clear visual identification of each track without cluttering the plot area. Fixed-width labels (80px) provide consistent layout.

**Implementation**: `QHBoxLayout` with `QLabel` on left, `PlotWidget` on right with stretch factor.

### Decision: Flexible Duration Parsing
**Rationale**: Duration data comes in various formats (Timedelta, float seconds, string representations). `_parse_duration_to_seconds()` handles all cases gracefully.

**Implementation**: Type checking with fallback chain: Timedelta → string parsing → float conversion.

## Risks / Trade-offs

### Risk: Performance Degradation with Very Large Datasets
**Mitigation**: Viewport culling limits rendered items. If needed, can add spatial indexing (R-tree) for O(log n) queries.

### Risk: Memory Usage with Cached Intervals
**Mitigation**: Timestamps are float64 (8 bytes per interval). 10,000 intervals ≈ 160KB, acceptable.

### Trade-off: Debounce Delay vs Responsiveness
**Current**: 50ms provides good balance. Can be tuned per use case if needed.

### Trade-off: Rectangle Rendering Method
**Chosen**: `PlotDataItem` with fillBrush (simple, works well)
**Alternative**: `QGraphicsRectItem` (more control, but more complex)

## Migration Plan
- Feature is additive; no migration needed
- Existing code continues to work unchanged
- New timeline widget can be integrated into existing UI workflows

## Open Questions
- Should tracks support collapsing/expanding for better space management?
- Should we add tooltips showing detailed interval information on hover?
- Should we support exporting visible time range to clipboard or file?

