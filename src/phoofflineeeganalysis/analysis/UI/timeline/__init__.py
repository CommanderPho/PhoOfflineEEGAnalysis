"""
Historical Data Timeline Widget using PyQtGraph.

This module provides a high-performance timeline visualization for multiple data modalities.
Each modality is rendered as a separate track, with all tracks synchronized by datetime.

Available Track Classes:
    - VideoMetadataTrack: For video recordings (video_start_datetime, video_end_datetime)
    - EEGRecordingTrack: For EEG recordings (recording_datetime, duration_sec)
    - MotionRecordingTrack: For motion recordings (recording_datetime, duration_sec)
    - PhoLogTrack: For PHO_LOG_TO_LSL annotations (onset, duration)
    - WhisperTrack: For Whisper transcript intervals (onset, duration)
    - XDFStreamTrack: Generic track for XDF stream data

Usage Examples:
    # Example 1: Create timeline from XDF stream info DataFrame
    from phoofflineeeganalysis.analysis.UI.timeline import create_timeline_from_xdf_streams
    
    timeline = create_timeline_from_xdf_streams(all_xdf_stream_infos_df)
    timeline.show()
    
    # Example 2: Manual track creation
    from phoofflineeeganalysis.analysis.UI.timeline import (
        TimelineWidget, VideoMetadataTrack, EEGRecordingTrack
    )
    from phoofflineeeganalysis.analysis.video_metadata import VideoMetadataParser
    
    timeline = TimelineWidget()
    
    # Add video track
    video_df = VideoMetadataParser.parse_video_folder(Path("path/to/videos"))
    video_track = VideoMetadataTrack(video_df)
    timeline.add_track(video_track)
    
    # Add tracks from XDF streams
    timeline.add_tracks_from_xdf_streams(all_xdf_stream_infos_df)
    
    timeline.show()
"""

# Import main timeline widget and factory functions
from phoofflineeeganalysis.analysis.UI.timeline.TimelineWidget import (
    TimelineWidget,
    create_timeline_from_xdf_streams,
    create_timeline_widget,
)

# Import all track classes for direct instantiation
from phoofflineeeganalysis.analysis.UI.timeline.tracks import (
    TrackWidget,
    VideoMetadataTrack,
    EEGRecordingTrack,
    MotionRecordingTrack,
    StringDataTrack,
    PhoLogTrack,
    WhisperTrack,
    XDFStreamTrack,
    TrackRegistry,
    get_default_registry,
    register_track,
)

__all__ = [
    # Main widget and factories
    'TimelineWidget',
    'create_timeline_from_xdf_streams',
    'create_timeline_widget',
    # Track classes
    'TrackWidget',
    'VideoMetadataTrack',
    'EEGRecordingTrack',
    'MotionRecordingTrack',
    'StringDataTrack',
    'PhoLogTrack',
    'WhisperTrack',
    'XDFStreamTrack',
    # Registry for extensibility
    'TrackRegistry',
    'get_default_registry',
    'register_track',
]

