"""
Main timeline widget that displays multiple synchronized tracks.
"""

from datetime import datetime
from typing import Optional, List, Tuple
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from pyqtgraph import ViewBox
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.EEGRecordingTrack import EEGRecordingTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.MotionRecordingTrack import MotionRecordingTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.PhoLogTrack import PhoLogTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.VideoMetadataTrack import VideoMetadataTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.XDFStreamTrack import XDFStreamTrack


class TimelineWidget(QWidget):
    """
    Main timeline widget that displays multiple synchronized tracks.
    
    All tracks share the same x-axis (datetime) and can be zoomed/panned together.
    """
    
    # Signal emitted when time range changes
    time_range_changed = pyqtSignal(datetime, datetime)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.tracks: List[TrackWidget] = []
        self.shared_viewbox: Optional[ViewBox] = None
        
        # Debounce timer for updates to improve performance
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._perform_update)
        self._pending_time_range: Optional[Tuple[datetime, datetime]] = None
        
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Create scroll area for tracks
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container widget for tracks
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(2)
        
        self.scroll_area.setWidget(self.tracks_container)
        layout.addWidget(self.scroll_area)
        
        # Track overall time range
        self.overall_time_range: Optional[Tuple[datetime, datetime]] = None
        
    def add_track(self, track: TrackWidget):
        """Add a track to the timeline."""
        self.tracks.append(track)
        self.tracks_layout.addWidget(track)
        
        # Link x-axis if we have a shared viewbox
        if self.shared_viewbox is None and len(self.tracks) > 0:
            # Use first track's viewbox as the master
            self.shared_viewbox = self.tracks[0].plot_widget.getViewBox()
            self.shared_viewbox.sigXRangeChanged.connect(self._on_x_range_changed)
            
            # Enable wheel zoom
            self.shared_viewbox.setMouseMode(self.shared_viewbox.PanMode)
        
        # Link subsequent tracks to the shared viewbox
        if self.shared_viewbox is not None and len(self.tracks) > 1:
            track.plot_widget.setXLink(self.tracks[0].plot_widget)
            # Enable wheel zoom on linked tracks too
            track.plot_widget.getViewBox().setMouseMode(track.plot_widget.getViewBox().PanMode)
        
        # Update overall time range
        track_range = track.get_time_range()
        if track_range is not None:
            if self.overall_time_range is None:
                self.overall_time_range = track_range
            else:
                start_min = min(self.overall_time_range[0], track_range[0])
                end_max = max(self.overall_time_range[1], track_range[1])
                self.overall_time_range = (start_min, end_max)
        
        # Set initial x-axis range if we have data
        if self.overall_time_range is not None:
            start_ts = self.overall_time_range[0].timestamp() if isinstance(self.overall_time_range[0], datetime) else float(self.overall_time_range[0])
            end_ts = self.overall_time_range[1].timestamp() if isinstance(self.overall_time_range[1], datetime) else float(self.overall_time_range[1])
            
            for track_widget in self.tracks:
                track_widget.plot_widget.setXRange(start_ts, end_ts, padding=0.05)
    
    def remove_track(self, track: TrackWidget):
        """Remove a track from the timeline."""
        if track in self.tracks:
            self.tracks.remove(track)
            self.tracks_layout.removeWidget(track)
            track.setParent(None)
            track.deleteLater()
            
            # Recalculate overall time range
            self._update_overall_time_range()
    
    def _update_overall_time_range(self):
        """Recalculate overall time range from all tracks."""
        self.overall_time_range = None
        for track in self.tracks:
            track_range = track.get_time_range()
            if track_range is not None:
                if self.overall_time_range is None:
                    self.overall_time_range = track_range
                else:
                    start_min = min(self.overall_time_range[0], track_range[0])
                    end_max = max(self.overall_time_range[1], track_range[1])
                    self.overall_time_range = (start_min, end_max)
    
    def _on_x_range_changed(self, viewbox):
        """Handle x-axis range changes (zoom/pan) - debounced for performance."""
        if self.shared_viewbox is None:
            return
        
        x_range = self.shared_viewbox.viewRange()[0]
        start_ts, end_ts = x_range
        
        # Convert timestamps back to datetime
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
        
        # Store pending update and debounce
        self._pending_time_range = (start_dt, end_dt)
        self._update_timer.stop()
        self._update_timer.start(50)  # 50ms debounce
    
    def _perform_update(self):
        """Perform the actual update after debounce."""
        if self._pending_time_range is None:
            return
        
        time_range = self._pending_time_range
        self._pending_time_range = None
        
        # Update all tracks (only visible ones will be rendered)
        for track in self.tracks:
            track.update_display(time_range=time_range)
        
        # Emit signal
        self.time_range_changed.emit(time_range[0], time_range[1])
    
    def set_time_range(self, start_dt: datetime, end_dt: datetime):
        """Programmatically set the visible time range."""
        start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
        end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
        
        if self.shared_viewbox is not None:
            self.shared_viewbox.setXRange(start_ts, end_ts, padding=0.05)
    
    def zoom_to_fit(self):
        """Zoom to show all data."""
        if self.overall_time_range is not None:
            self.set_time_range(self.overall_time_range[0], self.overall_time_range[1])
    
    def add_tracks_from_xdf_streams(self, xdf_stream_infos_df: pd.DataFrame, stream_names: Optional[List[str]] = None):
        """
        Add tracks for each stream type from an xdf_stream_infos_df DataFrame.
        
        Args:
            xdf_stream_infos_df: DataFrame with stream information including columns:
                - name: Stream name (e.g., "Epoc X", "Epoc X Motion", "TextLogger")
                - type: Stream type (e.g., "EEG", "SIGNAL", "Markers")
                - recording_datetime: Start datetime
                - duration_sec: Duration (Timedelta or float, can be NaT)
                - duration_sec_check: Alternative duration column (optional)
                - first_timestamp_dt: Alternative start time (optional)
                - last_timestamp_dt: Alternative end time (optional)
            stream_names: Optional list of stream names to include. If None, includes all unique stream names.
        """
        if xdf_stream_infos_df.empty:
            return
        
        # Determine which streams to include
        if stream_names is None:
            stream_names = xdf_stream_infos_df['name'].unique().tolist()
        
        # Map stream names to track classes and colors
        stream_name_to_track = {
            'Epoc X': ('EEG', EEGRecordingTrack),
            'Epoc X Motion': ('Motion', MotionRecordingTrack),
            'TextLogger': ('PHO_LOG', PhoLogTrack),
            'EventBoard': ('PHO_LOG', PhoLogTrack),
            'Epoc X eQuality': ('EEG Quality', EEGRecordingTrack),  # Use EEG track for quality streams
        }
        
        # Also check by type column as fallback
        type_to_track = {
            'EEG': ('EEG', EEGRecordingTrack),
            'SIGNAL': ('Motion', MotionRecordingTrack),
            'Markers': ('PHO_LOG', PhoLogTrack),
            'Raw': ('EEG', EEGRecordingTrack),  # Raw EEG streams
        }
        
        for stream_name in stream_names:
            # Filter DataFrame for this stream
            stream_df = xdf_stream_infos_df[xdf_stream_infos_df['name'] == stream_name].copy()
            
            if stream_df.empty:
                continue
            
            # Determine track class
            track_name = None
            track_class = None
            
            # Try stream name mapping first
            if stream_name in stream_name_to_track:
                track_name, track_class = stream_name_to_track[stream_name]
            else:
                # Fall back to type column
                stream_type = stream_df['type'].iloc[0] if 'type' in stream_df.columns else None
                if stream_type and stream_type in type_to_track:
                    track_name, track_class = type_to_track[stream_type]
            
            # If still no match, use generic XDFStreamTrack
            if track_class is None:
                track_name = stream_name
                track_class = XDFStreamTrack
            
            # Create track
            try:
                track = track_class(stream_df, name=track_name)
                self.add_track(track)
            except Exception as e:
                # Skip streams that fail to create tracks
                print(f"Warning: Failed to create track for stream '{stream_name}': {e}")
                import traceback
                traceback.print_exc()
                continue


def create_timeline_from_xdf_streams(xdf_stream_infos_df: pd.DataFrame, stream_names: Optional[List[str]] = None, video_df: Optional[pd.DataFrame] = None) -> TimelineWidget:
    """
    Create a timeline widget with tracks for each stream type from xdf_stream_infos_df.
    
    Args:
        xdf_stream_infos_df: DataFrame with stream information from XDF files.
            Expected columns: name, type, recording_datetime, duration_sec
        stream_names: Optional list of stream names to include. If None, includes all unique stream names.
        video_df: Optional DataFrame with video metadata to add as a track.
    
    Returns:
        TimelineWidget instance with tracks for each stream type.
    
    Example:
        from phoofflineeeganalysis.analysis.UI.timeline import create_timeline_from_xdf_streams
        
        timeline = create_timeline_from_xdf_streams(all_xdf_stream_infos_df)
        timeline.show()
    """
    timeline = TimelineWidget()
    
    # Add tracks from xdf streams
    if xdf_stream_infos_df is not None and not xdf_stream_infos_df.empty:
        timeline.add_tracks_from_xdf_streams(xdf_stream_infos_df, stream_names=stream_names)
    
    # Add video track if provided
    if video_df is not None and not video_df.empty:
        video_track = VideoMetadataTrack(video_df)
        timeline.add_track(video_track)
    
    return timeline


def create_timeline_widget(video_df: Optional[pd.DataFrame] = None) -> TimelineWidget:
    """
    Factory function to create a timeline widget with video track.
    
    Args:
        video_df: Optional DataFrame with video metadata from VideoMetadataParser.
    
    Returns:
        TimelineWidget instance.
    """
    timeline = TimelineWidget()
    
    if video_df is not None and not video_df.empty:
        video_track = VideoMetadataTrack(video_df)
        timeline.add_track(video_track)
    
    return timeline

