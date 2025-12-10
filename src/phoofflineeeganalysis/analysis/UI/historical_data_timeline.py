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
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ViewBox, DateAxisItem


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
        
        # Create PlotWidget with DateAxisItem for proper datetime x-axis
        self.plot_widget = PlotWidget(parent=self, axisItems={'bottom': DateAxisItem(orientation='bottom')})
        self.plot_widget.setFixedHeight(height)
        self.plot_widget.setLabel('left', name)
        self.plot_widget.hideAxis('left')
        self.plot_widget.setLabel('bottom', 'Time')
        
        # Enable mouse interaction for zoom/pan
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setMenuEnabled(False)
        
        # Configure ViewBox for wheel zoom and pan
        vb = self.plot_widget.getViewBox()
        vb.setMouseMode(vb.PanMode)
        # Enable wheel zoom (should be default, but make explicit)
        vb.enableAutoRange(enable=False)
        # Set limits to allow full range
        vb.setLimits(xMin=None, xMax=None, yMin=0, yMax=1)
        
        # Cache all intervals for performance
        self._all_intervals: List[Tuple[datetime, datetime]] = []
        self._all_intervals_ts: Optional[np.ndarray] = None  # Cached as timestamps
        
        # Store rectangles for efficient updates
        self.rect_items: List[pg.PlotDataItem] = []
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """
        Return list of (start_datetime, end_datetime) tuples for recordings.
        
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _get_recording_intervals()")
    
    def _cache_intervals(self):
        """Cache intervals as timestamps for fast filtering."""
        intervals = self._get_recording_intervals()
        self._all_intervals = intervals
        
        if intervals:
            # Convert to numpy array of timestamps for fast filtering
            starts = np.array([s.timestamp() if isinstance(s, datetime) else float(s) for s, _ in intervals])
            ends = np.array([e.timestamp() if isinstance(e, datetime) else float(e) for _, e in intervals])
            self._all_intervals_ts = np.column_stack([starts, ends])
        else:
            self._all_intervals_ts = None
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """
        Update the track display for the given time range.
        
        Args:
            time_range: Optional (start_datetime, end_datetime) tuple to limit display.
                       If None, displays all recordings.
        """
        # Cache intervals if not already cached
        if self._all_intervals_ts is None:
            self._cache_intervals()
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            self.plot_widget.clear()
            self.rect_items.clear()
            return
        
        # Filter by time range if provided (using numpy for speed)
        if time_range is not None:
            start_dt, end_dt = time_range
            start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
            end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
            
            # Fast numpy filtering: keep intervals that overlap with visible range
            mask = (self._all_intervals_ts[:, 1] >= start_ts) & (self._all_intervals_ts[:, 0] <= end_ts)
            visible_intervals = self._all_intervals_ts[mask]
        else:
            visible_intervals = self._all_intervals_ts
        
        # Clear existing rectangles
        self.plot_widget.clear()
        self.rect_items.clear()
        
        if len(visible_intervals) == 0:
            return
        
        # Batch create rectangles efficiently
        pen = pg.mkPen(color=(100, 150, 200, 255), width=1)
        brush = pg.mkBrush(color=(100, 150, 200, 150))
        
        # Pre-allocate arrays for batch plotting
        if len(visible_intervals) > 0:
            # Create all rectangles in one go using efficient PlotDataItem
            for start_ts, end_ts in visible_intervals:
                width = end_ts - start_ts
                if width <= 0:
                    continue
                
                # Create rectangle as closed polygon (more efficient than individual items)
                x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
                y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
                
                # Use plot() with fillBrush for efficient rendering
                rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
                self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)
    
    def get_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the overall time range covered by this track's data.
        
        Returns:
            (start_datetime, end_datetime) tuple or None if no data.
        """
        if self._all_intervals_ts is None:
            self._cache_intervals()
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            return None
        
        start_ts = self._all_intervals_ts[:, 0].min()
        end_ts = self._all_intervals_ts[:, 1].max()
        
        return (datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts))


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
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Extract video recording intervals from DataFrame."""
        if self.video_df.empty or 'video_start_datetime' not in self.video_df.columns:
            return []
        
        intervals = []
        for _, row in self.video_df.iterrows():
            start_dt = row.get('video_start_datetime')
            end_dt = row.get('video_end_datetime')
            
            if pd.isna(start_dt):
                continue
            
            # If end_dt is not available, try to calculate from duration
            if pd.isna(end_dt):
                if 'video_duration' in row:
                    duration = row.get('video_duration', 0)
                    if pd.notna(duration) and duration > 0:
                        from datetime import timedelta
                        end_dt = start_dt + timedelta(seconds=float(duration))
                    else:
                        continue
                else:
                    continue
            
            if pd.isna(end_dt):
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
    # folder_path = Path(r"M:\ScreenRecordings\EyeTrackerVR_Recordings")
    # video_df = VideoMetadataParser.parse_video_folder(folder_path)

    csv_save_path = Path('output').joinpath('2025-12-09_parsed_videos.csv').resolve()
    assert csv_save_path.exists()
    video_df: pd.DataFrame = pd.read_csv(csv_save_path)
    

    # Create timeline
    timeline = create_timeline_widget(video_df=video_df)
    timeline.setWindowTitle("Historical Data Timeline")
    timeline.resize(1200, 600)
    timeline.show()
    
    sys.exit(app.exec_())
