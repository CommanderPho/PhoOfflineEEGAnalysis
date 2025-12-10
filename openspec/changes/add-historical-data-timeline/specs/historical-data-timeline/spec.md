## ADDED Requirements

### Requirement: Historical Data Timeline Widget
The system SHALL provide a PyQtGraph-based timeline widget that visualizes multiple data modalities as synchronized tracks, each displaying recording intervals as rectangles aligned to a shared datetime x-axis.

#### Scenario: Create timeline with multiple tracks
- **WHEN** a user creates a `TimelineWidget` and adds multiple track instances
- **THEN** all tracks SHALL be displayed vertically with synchronized x-axis (datetime) and independent y-axis ranges

#### Scenario: Synchronized zoom and pan
- **WHEN** a user zooms or pans any track using mouse wheel or drag
- **THEN** all tracks SHALL update their visible time range to match, maintaining synchronization

#### Scenario: Performance with large datasets
- **WHEN** a timeline contains hundreds or thousands of intervals
- **THEN** zoom and pan operations SHALL remain responsive (<100ms update latency) through viewport culling and debounced updates

### Requirement: Base Track Widget
The system SHALL provide a base `TrackWidget` class that renders recording intervals as filled rectangles, with extensible architecture for modality-specific implementations.

#### Scenario: Track displays intervals as rectangles
- **WHEN** a track is provided with recording intervals (start_datetime, end_datetime)
- **THEN** the track SHALL render each interval as a filled rectangle spanning the time range

#### Scenario: Track has left-edge label
- **WHEN** a track is created with a name
- **THEN** the track SHALL display the name in a fixed-width label on the left edge (80px width)

#### Scenario: Track caches intervals for performance
- **WHEN** intervals are provided to a track
- **THEN** the track SHALL cache intervals as numpy arrays of timestamps for efficient filtering

#### Scenario: Track filters by visible time range
- **WHEN** a track's visible time range changes
- **THEN** the track SHALL render only intervals that overlap the visible range

### Requirement: Video Metadata Track
The system SHALL provide a `VideoMetadataTrack` that displays video recording intervals from a DataFrame with `video_start_datetime` and `video_end_datetime` columns.

#### Scenario: Create video track from DataFrame
- **WHEN** a user creates a `VideoMetadataTrack` with a DataFrame containing video metadata
- **THEN** the track SHALL extract intervals from `video_start_datetime` and `video_end_datetime` columns and render them with blue coloring

#### Scenario: Handle missing end datetime
- **WHEN** a video row has `video_start_datetime` but missing `video_end_datetime`
- **THEN** the track SHALL calculate end datetime from `video_duration` if available, otherwise skip the row

### Requirement: EEG Recording Track
The system SHALL provide an `EEGRecordingTrack` that displays EEG session intervals from a DataFrame with `recording_datetime` and `duration_sec` columns.

#### Scenario: Create EEG track from DataFrame
- **WHEN** a user creates an `EEGRecordingTrack` with a DataFrame containing EEG session metadata
- **THEN** the track SHALL extract intervals from `recording_datetime` and calculate end time from `duration_sec`, rendering with green/blue coloring

#### Scenario: Handle duration variants
- **WHEN** duration is provided as Timedelta, float, int, or string
- **THEN** the track SHALL parse it correctly using `_parse_duration_to_seconds()` helper

### Requirement: Motion Recording Track
The system SHALL provide a `MotionRecordingTrack` that displays motion sensor recording intervals with orange/red coloring.

#### Scenario: Create motion track from DataFrame
- **WHEN** a user creates a `MotionRecordingTrack` with motion session metadata
- **THEN** the track SHALL render intervals with orange/red theme to distinguish from EEG

### Requirement: PHO_LOG Track
The system SHALL provide a `PhoLogTrack` that displays PHO_LOG_TO_LSL annotation intervals from a DataFrame with `onset` and `duration` columns.

#### Scenario: Create PHO_LOG track from DataFrame
- **WHEN** a user creates a `PhoLogTrack` with annotation data
- **THEN** the track SHALL render intervals with purple coloring

#### Scenario: Handle zero-duration annotations
- **WHEN** an annotation has zero or missing duration
- **THEN** the track SHALL use a minimal duration (0.1 seconds) to ensure visibility

### Requirement: Whisper Track
The system SHALL provide a `WhisperTrack` that displays Whisper transcript intervals with cyan/teal coloring.

#### Scenario: Create Whisper track from DataFrame
- **WHEN** a user creates a `WhisperTrack` with transcript data
- **THEN** the track SHALL render intervals with cyan/teal theme

### Requirement: XDF Stream Track
The system SHALL provide an `XDFStreamTrack` that displays generic XDF stream intervals with flexible datetime and duration column handling.

#### Scenario: Create XDF stream track with multiple datetime sources
- **WHEN** a user creates an `XDFStreamTrack` with stream metadata
- **THEN** the track SHALL attempt to use `recording_datetime`, then `first_timestamp_dt`, then `last_timestamp_dt` for start/end times

#### Scenario: Handle marker streams with minimal duration
- **WHEN** a marker stream has no duration information
- **THEN** the track SHALL use a minimal duration (0.1 seconds) to ensure visibility

### Requirement: Datetime X-Axis
All tracks SHALL use PyQtGraph's `DateAxisItem` to display timestamps as formatted datetime strings.

#### Scenario: X-axis shows readable datetime
- **WHEN** a timeline is displayed
- **THEN** the x-axis SHALL show formatted datetime labels (e.g., "2025-12-10 14:30:00") rather than raw timestamps

### Requirement: Factory Functions
The system SHALL provide factory functions for convenient timeline creation.

#### Scenario: Create timeline from XDF streams
- **WHEN** a user calls `create_timeline_from_xdf_streams(xdf_stream_infos_df)`
- **THEN** the function SHALL automatically create appropriate track types based on stream names and types, returning a `TimelineWidget`

#### Scenario: Create timeline with video track
- **WHEN** a user calls `create_timeline_widget(video_df=video_df)`
- **THEN** the function SHALL create a timeline with a `VideoMetadataTrack` if video data is provided

### Requirement: Bulk Track Creation
The `TimelineWidget` SHALL provide `add_tracks_from_xdf_streams()` method to automatically create tracks for multiple streams.

#### Scenario: Add tracks from XDF stream info DataFrame
- **WHEN** a user calls `timeline.add_tracks_from_xdf_streams(xdf_stream_infos_df)`
- **THEN** the method SHALL create tracks for each unique stream name, mapping stream names/types to appropriate track classes

#### Scenario: Stream name to track mapping
- **WHEN** streams include "Epoc X", "Epoc X Motion", "TextLogger", or "EventBoard"
- **THEN** the method SHALL map them to `EEGRecordingTrack`, `MotionRecordingTrack`, and `PhoLogTrack` respectively

#### Scenario: Fallback to generic track
- **WHEN** a stream name doesn't match known mappings
- **THEN** the method SHALL use `XDFStreamTrack` as a generic fallback

### Requirement: Performance Optimizations
The timeline SHALL implement performance optimizations to handle large datasets efficiently.

#### Scenario: Debounced updates during pan/zoom
- **WHEN** a user rapidly pans or zooms the timeline
- **THEN** updates SHALL be debounced (50ms delay) to prevent excessive redraws

#### Scenario: Viewport culling
- **WHEN** a timeline contains intervals spanning days or weeks
- **THEN** only intervals overlapping the visible time range SHALL be rendered

#### Scenario: Efficient interval filtering
- **WHEN** filtering intervals by visible time range
- **THEN** the system SHALL use numpy boolean masking for O(n) performance

### Requirement: Mouse Interactions
The timeline SHALL support mouse wheel zoom and pan interactions.

#### Scenario: Mouse wheel zoom
- **WHEN** a user scrolls the mouse wheel over a track
- **THEN** the timeline SHALL zoom in/out centered on the mouse position

#### Scenario: Pan with mouse drag
- **WHEN** a user drags horizontally on a track
- **THEN** the timeline SHALL pan left/right, maintaining synchronization across tracks

### Requirement: Programmatic Control
The timeline SHALL provide methods for programmatic time range control.

#### Scenario: Set time range programmatically
- **WHEN** a user calls `timeline.set_time_range(start_dt, end_dt)`
- **THEN** all tracks SHALL update to show the specified time range

#### Scenario: Zoom to fit all data
- **WHEN** a user calls `timeline.zoom_to_fit()`
- **THEN** all tracks SHALL adjust to show the full time range covered by all intervals

