"""
Track widgets for timeline visualization.

This package provides various track implementations for displaying different
data modalities in a synchronized timeline view.
"""

from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.VideoMetadataTrack import VideoMetadataTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.EEGRecordingTrack import EEGRecordingTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.MotionRecordingTrack import MotionRecordingTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.StringDataTrack import StringDataTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.PhoLogTrack import PhoLogTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.WhisperTrack import WhisperTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.XDFStreamTrack import XDFStreamTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.registry import (
    TrackRegistry,
    get_default_registry,
    register_track
)

__all__ = [
    'TrackWidget',
    'VideoMetadataTrack',
    'EEGRecordingTrack',
    'MotionRecordingTrack',
    'StringDataTrack',
    'PhoLogTrack',
    'WhisperTrack',
    'XDFStreamTrack',
    'TrackRegistry',
    'get_default_registry',
    'register_track',
]

