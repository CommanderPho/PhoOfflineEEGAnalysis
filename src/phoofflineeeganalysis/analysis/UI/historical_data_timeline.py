"""
Historical Data Timeline Widget using PyQtGraph.

This module provides a high-performance timeline visualization for multiple data modalities.
Each modality is rendered as a separate track, with all tracks synchronized by datetime.

Usage:
    from phoofflineeeganalysis.analysis.UI.historical_data_timeline import TimelineWidget, VideoMetadataTrack
    from phoofflineeeganalysis.analysis.video_metadata import VideoMetadataParser
    
    # Load video metadata
    video_df = VideoMetadataParser.parse_video_folder(Path("path/to/videos"))
    
    # Create timeline widget
    timeline = TimelineWidget()
    
    # Add video track
    video_track = VideoMetadataTrack(video_df)
    timeline.add_track(video_track)
    
    # Show widget
    timeline.show()
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem, ViewBox


class TrackWidget(QWidget):
    """
    Base class for timeline tracks that display modality-specific data.
    
    Each track renders rectangles corresponding to recording intervals (start, end).
    Subclasses should implement _get_recording_intervals() to provide the data.
    """
    
    def __init__(self, name: str, height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        self.height = height
        self.plot_widget = PlotWidget(parent=self)
        self.plot_widget.setFixedHeight(height)
        self.plot_widget.setLabel('left', name)
        self.plot_widget.hideAxis('left')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setMenuEnabled(False)
        
        # X-axis will show timestamps (can be formatted as datetime by PyQtGraph)
        
        # Store rectangles for efficient updates
        self.rect_items: List[pg.PlotDataItem] = []
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        
    def _get_recording_intervals(self) -> List[tuple]:
        """
        Return list of (start_datetime, end_datetime) tuples for recordings.
        
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _get_recording_intervals()")
    
    def update_display(self, time_range: Optional[tuple] = None):
        """
        Update the track display for the given time range.
        
        Args:
            time_range: Optional (start_datetime, end_datetime) tuple to limit display.
                       If None, displays all recordings.
        """
        # Clear existing rectangles
        self.plot_widget.clear()
        self.rect_items.clear()
        
        # Get recording intervals
        intervals = self._get_recording_intervals()
        if not intervals:
            return
        
        # Filter by time range if provided
        if time_range is not None:
            start_dt, end_dt = time_range
            intervals = [(s, e) for s, e in intervals if e >= start_dt and s <= end_dt]
        
        # Convert datetimes to timestamps for plotting
        for start_dt, end_dt in intervals:
            start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
            end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
            
            # Create rectangle
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            # Use PlotDataItem with fill for efficient rectangle rendering
            # Create rectangle as closed polygon
            x = [start_ts, start_ts, end_ts, end_ts, start_ts]
            y = [0, 1, 1, 0, 0]
            
            pen = pg.mkPen(color=(100, 150, 200, 255), width=1)
            brush = pg.mkBrush(color=(100, 150, 200, 150))
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, brush=brush, fillBrush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)
    
    def get_time_range(self) -> Optional[tuple]:
        """
        Get the overall time range covered by this track's data.
        
        Returns:
            (start_datetime, end_datetime) tuple or None if no data.
        """
        intervals = self._get_recording_intervals()
        if not intervals:
            return None
        
        start_times = [s for s, _ in intervals]
        end_times = [e for _, e in intervals]
        
        return (min(start_times), max(end_times))


class VideoMetadataTrack(TrackWidget):
    """
    Track widget for displaying video recording intervals from VideoMetadataParser.
    
    Expects a DataFrame with columns:
    - video_start_datetime: datetime
    - video_end_datetime: datetime
    """
    
    def __init__(self, video_df: pd.DataFrame, name: str = "Video Recordings", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        self.video_df = video_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'video_start_datetime' in self.video_df.columns:
            self.video_df['video_start_datetime'] = pd.to_datetime(self.video_df['video_start_datetime'])
        if 'video_end_datetime' in self.video_df.columns:
            self.video_df['video_end_datetime'] = pd.to_datetime(self.video_df['video_end_datetime'])
        
        # Initial display update
        self.update_display()
    
    def _get_recording_intervals(self) -> List[tuple]:
        """Extract video recording intervals from DataFrame."""
        if self.video_df.empty or 'video_start_datetime' not in self.video_df.columns:
            return []
        
        intervals = []
        for _, row in self.video_df.iterrows():
            start_dt = row.get('video_start_datetime')
            end_dt = row.get('video_end_datetime')
            
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            
            # If end_dt is not available, try to calculate from duration
            if pd.isna(end_dt) and 'video_duration' in row:
                duration = row.get('video_duration', 0)
                if pd.notna(duration) and duration > 0:
                    from datetime import timedelta
                    end_dt = start_dt + timedelta(seconds=float(duration))
                else:
                    continue
            
            intervals.append((start_dt, end_dt))
        
        return intervals


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
        self.overall_time_range: Optional[tuple] = None
        
    def add_track(self, track: TrackWidget):
        """Add a track to the timeline."""
        self.tracks.append(track)
        self.tracks_layout.addWidget(track)
        
        # Link x-axis if we have a shared viewbox
        if self.shared_viewbox is None and len(self.tracks) > 0:
            # Use first track's viewbox as the master
            self.shared_viewbox = self.tracks[0].plot_widget.getViewBox()
            self.shared_viewbox.sigXRangeChanged.connect(self._on_x_range_changed)
        
        # Link subsequent tracks to the shared viewbox
        if self.shared_viewbox is not None and len(self.tracks) > 1:
            track.plot_widget.setXLink(self.tracks[0].plot_widget)
        
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
        """Handle x-axis range changes (zoom/pan)."""
        if self.shared_viewbox is None:
            return
        
        x_range = self.shared_viewbox.viewRange()[0]
        start_ts, end_ts = x_range
        
        # Convert timestamps back to datetime
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
        
        # Update all tracks
        for track in self.tracks:
            track.update_display(time_range=(start_dt, end_dt))
        
        # Emit signal
        self.time_range_changed.emit(start_dt, end_dt)
    
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


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    from PyQt5.QtWidgets import QApplication
    from phoofflineeeganalysis.analysis.video_metadata import VideoMetadataParser
    
    app = QApplication(sys.argv)
    
    # Load video metadata
    folder_path = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings")
    video_df = VideoMetadataParser.parse_video_folder(folder_path)
    
    # Create timeline
    timeline = create_timeline_widget(video_df=video_df)
    timeline.setWindowTitle("Historical Data Timeline")
    timeline.resize(1200, 600)
    timeline.show()
    
    sys.exit(app.exec_())

