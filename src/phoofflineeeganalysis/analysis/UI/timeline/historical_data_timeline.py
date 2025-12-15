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
    from phoofflineeeganalysis.analysis.UI.historical_data_timeline import create_timeline_from_xdf_streams
    
    timeline = create_timeline_from_xdf_streams(all_xdf_stream_infos_df)
    timeline.show()
    
    # Example 2: Manual track creation
    from phoofflineeeganalysis.analysis.UI.historical_data_timeline import (
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

NOTE: This module is maintained for backward compatibility. New code should import from
phoofflineeeganalysis.analysis.UI.timeline instead.
"""

# Re-export everything from the new timeline module for backward compatibility
from phoofflineeeganalysis.analysis.UI.timeline import (
    TimelineWidget,
    create_timeline_from_xdf_streams,
    create_timeline_widget,
    TrackWidget,
    VideoMetadataTrack,
    EEGRecordingTrack,
    MotionRecordingTrack,
    PhoLogTrack,
    WhisperTrack,
    XDFStreamTrack,
    TrackRegistry,
    get_default_registry,
    register_track,
)

__all__ = [
    'TimelineWidget',
    'create_timeline_from_xdf_streams',
    'create_timeline_widget',
    'TrackWidget',
    'VideoMetadataTrack',
    'EEGRecordingTrack',
    'MotionRecordingTrack',
    'PhoLogTrack',
    'WhisperTrack',
    'XDFStreamTrack',
    'TrackRegistry',
    'get_default_registry',
    'register_track',
]


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    from PyQt5.QtWidgets import QApplication
    import pandas as pd
    
    app = QApplication(sys.argv)
    
    output_folder = Path('output').resolve()
    assert output_folder.exists()

    timeline: TimelineWidget = TimelineWidget()

    csv_save_path = output_folder.joinpath('2025-12-15_parsed_videos.csv').resolve()
    assert csv_save_path.exists()
    video_df: pd.DataFrame = pd.read_csv(csv_save_path)

    csv_save_path = output_folder.joinpath('2025-12-15_all_xdf_stream_infos.csv').resolve()
    assert csv_save_path.exists()
    all_xdf_stream_infos_df: pd.DataFrame = pd.read_csv(csv_save_path)

    # Add tracks from different modalities
    timeline.add_track(VideoMetadataTrack(video_df))
    timeline.add_tracks_from_xdf_streams(all_xdf_stream_infos_df)
    timeline.setWindowTitle("Historical Data Timeline")
    timeline.resize(1900, 800)
    timeline.show()
    
    sys.exit(app.exec_())
