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

Usage:
    from phoofflineeeganalysis.analysis.UI.historical_data_timeline import (
        TimelineWidget, VideoMetadataTrack, EEGRecordingTrack, MotionRecordingTrack
    )
    from phoofflineeeganalysis.analysis.video_metadata import VideoMetadataParser
    
    # Load video metadata
    video_df = VideoMetadataParser.parse_video_folder(Path("path/to/videos"))
    
    # Create timeline widget
    timeline = TimelineWidget()
    
    # Add video track
    video_track = VideoMetadataTrack(video_df)
    timeline.add_track(video_track)
    
    # Add EEG track (from SessionModality.df)
    eeg_track = EEGRecordingTrack(eeg_df)
    timeline.add_track(eeg_track)
    
    # Show widget
    timeline.show()
"""

from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
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
        
        # Create label for track name (left edge)
        self.name_label = QLabel(name, self)
        self.name_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.name_label.setFixedWidth(80)  # Fixed width for label
        # Rotate text vertically
        font = QFont()
        font.setPointSize(9)
        self.name_label.setFont(font)
        self.name_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 2px;
            }
        """)
        
        # Set up horizontal layout: label on left, plot on right
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.name_label)
        layout.addWidget(self.plot_widget, stretch=1)
        
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
    
    def __init__(self, video_df: pd.DataFrame, name: str = "Videos", height: int = 60, parent: Optional[QWidget] = None):
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
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Override to use video-specific colors."""
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
        
        # Video-specific colors (blue theme)
        pen = pg.mkPen(color=(100, 150, 200, 255), width=1)
        brush = pg.mkBrush(color=(100, 150, 200, 150))
        
        # Create rectangles
        for start_ts, end_ts in visible_intervals:
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
            y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)


class XDFStreamTrack(TrackWidget):
    """
    Generic track widget for displaying XDF stream intervals.
    
    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds), can be NaT
    - duration_sec_check: Alternative duration column (optional)
    - first_timestamp_dt: Alternative start time (optional)
    - last_timestamp_dt: Alternative end time (optional)
    """
    
    def __init__(self, stream_df: pd.DataFrame, name: str = "Stream", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        self.stream_df = stream_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.stream_df.columns:
            self.stream_df['recording_datetime'] = pd.to_datetime(self.stream_df['recording_datetime'])
        if 'first_timestamp_dt' in self.stream_df.columns:
            self.stream_df['first_timestamp_dt'] = pd.to_datetime(self.stream_df['first_timestamp_dt'])
        if 'last_timestamp_dt' in self.stream_df.columns:
            self.stream_df['last_timestamp_dt'] = pd.to_datetime(self.stream_df['last_timestamp_dt'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Extract stream recording intervals from DataFrame."""
        if self.stream_df.empty:
            return []
        
        intervals = []
        for _, row in self.stream_df.iterrows():
            # Try to get start time
            start_dt = None
            if 'recording_datetime' in row and pd.notna(row['recording_datetime']):
                start_dt = row['recording_datetime']
            elif 'first_timestamp_dt' in row and pd.notna(row['first_timestamp_dt']):
                start_dt = row['first_timestamp_dt']
            
            if start_dt is None or pd.isna(start_dt):
                continue
            
            # Try to get end time or calculate from duration
            end_dt = None
            
            # First try: use last_timestamp_dt if available
            if 'last_timestamp_dt' in row and pd.notna(row['last_timestamp_dt']):
                end_dt = row['last_timestamp_dt']
            
            # Second try: calculate from duration_sec_check
            if end_dt is None or pd.isna(end_dt):
                duration = row.get('duration_sec_check', None)
                if pd.notna(duration):
                    if isinstance(duration, pd.Timedelta):
                        duration_seconds = duration.total_seconds()
                    else:
                        duration_seconds = float(duration)
                    if duration_seconds > 0:
                        from datetime import timedelta
                        end_dt = start_dt + timedelta(seconds=duration_seconds)
            
            # Third try: calculate from duration_sec
            if end_dt is None or pd.isna(end_dt):
                duration = row.get('duration_sec', None)
                if pd.notna(duration):
                    if isinstance(duration, pd.Timedelta):
                        duration_seconds = duration.total_seconds()
                    else:
                        duration_seconds = float(duration)
                    if duration_seconds > 0:
                        from datetime import timedelta
                        end_dt = start_dt + timedelta(seconds=duration_seconds)
            
            # If still no end time, skip or use minimal duration
            if end_dt is None or pd.isna(end_dt):
                # For marker streams, use a minimal duration
                if row.get('type') == 'Markers' or row.get('name') == 'TextLogger':
                    from datetime import timedelta
                    end_dt = start_dt + timedelta(seconds=0.1)
                else:
                    continue
            
            intervals.append((start_dt, end_dt))
        
        return intervals
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Override to use stream-specific colors."""
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
        
        # Default colors (gray theme)
        pen = pg.mkPen(color=(150, 150, 150, 255), width=1)
        brush = pg.mkBrush(color=(150, 150, 150, 150))
        
        # Create rectangles
        for start_ts, end_ts in visible_intervals:
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
            y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)


class EEGRecordingTrack(TrackWidget):
    """
    Track widget for displaying EEG recording intervals from SessionModality.
    
    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """
    
    def __init__(self, eeg_df: pd.DataFrame, name: str = "EEG", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        self.eeg_df = eeg_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.eeg_df.columns:
            self.eeg_df['recording_datetime'] = pd.to_datetime(self.eeg_df['recording_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Extract EEG recording intervals from DataFrame."""
        if self.eeg_df.empty or 'recording_datetime' not in self.eeg_df.columns:
            return []
        
        intervals = []
        for _, row in self.eeg_df.iterrows():
            start_dt = row.get('recording_datetime')
            
            if pd.isna(start_dt):
                continue
            
            # Get duration - try duration_sec_check first, then duration_sec
            duration = row.get('duration_sec_check', None)
            if pd.isna(duration):
                duration = row.get('duration_sec', None)
            
            if pd.isna(duration):
                continue
            
            # Convert duration to seconds if it's a Timedelta
            if isinstance(duration, pd.Timedelta):
                duration_seconds = duration.total_seconds()
            else:
                duration_seconds = float(duration)
            
            if duration_seconds <= 0:
                continue
            
            # Calculate end datetime
            from datetime import timedelta
            end_dt = start_dt + timedelta(seconds=duration_seconds)
            
            intervals.append((start_dt, end_dt))
        
        return intervals
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Override to use EEG-specific colors."""
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
        
        # EEG-specific colors (green/blue theme)
        pen = pg.mkPen(color=(50, 200, 100, 255), width=1)
        brush = pg.mkBrush(color=(50, 200, 100, 150))
        
        # Create rectangles
        for start_ts, end_ts in visible_intervals:
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
            y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)


class MotionRecordingTrack(TrackWidget):
    """
    Track widget for displaying motion recording intervals from SessionModality.
    
    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """
    
    def __init__(self, motion_df: pd.DataFrame, name: str = "Motion", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        self.motion_df = motion_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.motion_df.columns:
            self.motion_df['recording_datetime'] = pd.to_datetime(self.motion_df['recording_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Extract motion recording intervals from DataFrame."""
        if self.motion_df.empty or 'recording_datetime' not in self.motion_df.columns:
            return []
        
        intervals = []
        for _, row in self.motion_df.iterrows():
            start_dt = row.get('recording_datetime')
            
            if pd.isna(start_dt):
                continue
            
            # Get duration
            duration = row.get('duration_sec', None)
            if pd.isna(duration):
                continue
            
            # Convert duration to seconds if it's a Timedelta
            if isinstance(duration, pd.Timedelta):
                duration_seconds = duration.total_seconds()
            else:
                duration_seconds = float(duration)
            
            if duration_seconds <= 0:
                continue
            
            # Calculate end datetime
            from datetime import timedelta
            end_dt = start_dt + timedelta(seconds=duration_seconds)
            
            intervals.append((start_dt, end_dt))
        
        return intervals
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Override to use motion-specific colors."""
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
        
        # Motion-specific colors (orange/red theme)
        pen = pg.mkPen(color=(255, 150, 50, 255), width=1)
        brush = pg.mkBrush(color=(255, 150, 50, 150))
        
        # Create rectangles
        for start_ts, end_ts in visible_intervals:
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
            y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)


class PhoLogTrack(TrackWidget):
    """
    Track widget for displaying PHO_LOG_TO_LSL annotation intervals.
    
    Expects a DataFrame with columns:
    - onset: datetime (start time)
    - duration: float or Timedelta (duration in seconds)
    """
    
    def __init__(self, pho_log_df: pd.DataFrame, name: str = "PHO_LOG", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        self.pho_log_df = pho_log_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'onset' in self.pho_log_df.columns:
            self.pho_log_df['onset'] = pd.to_datetime(self.pho_log_df['onset'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Extract PHO_LOG annotation intervals from DataFrame."""
        if self.pho_log_df.empty or 'onset' not in self.pho_log_df.columns:
            return []
        
        intervals = []
        for _, row in self.pho_log_df.iterrows():
            start_dt = row.get('onset')
            
            if pd.isna(start_dt):
                continue
            
            # Get duration
            duration = row.get('duration', None)
            if pd.isna(duration):
                # If no duration, use a minimal duration (e.g., 0.1 seconds for point events)
                duration = 0.1
            
            # Convert duration to seconds if it's a Timedelta
            if isinstance(duration, pd.Timedelta):
                duration_seconds = duration.total_seconds()
            else:
                duration_seconds = float(duration)
            
            if duration_seconds <= 0:
                duration_seconds = 0.1  # Minimum duration for visibility
            
            # Calculate end datetime
            from datetime import timedelta
            end_dt = start_dt + timedelta(seconds=duration_seconds)
            
            intervals.append((start_dt, end_dt))
        
        return intervals
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Override to use PHO_LOG-specific colors."""
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
        
        # PHO_LOG-specific colors (purple theme)
        pen = pg.mkPen(color=(200, 100, 255, 255), width=1)
        brush = pg.mkBrush(color=(200, 100, 255, 150))
        
        # Create rectangles
        for start_ts, end_ts in visible_intervals:
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
            y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)


class WhisperTrack(TrackWidget):
    """
    Track widget for displaying Whisper transcript intervals.
    
    Expects a DataFrame with columns:
    - onset: datetime (start time)
    - duration: float or Timedelta (duration in seconds)
    """
    
    def __init__(self, whisper_df: pd.DataFrame, name: str = "Whisper", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        self.whisper_df = whisper_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'onset' in self.whisper_df.columns:
            self.whisper_df['onset'] = pd.to_datetime(self.whisper_df['onset'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Extract Whisper transcript intervals from DataFrame."""
        if self.whisper_df.empty or 'onset' not in self.whisper_df.columns:
            return []
        
        intervals = []
        for _, row in self.whisper_df.iterrows():
            start_dt = row.get('onset')
            
            if pd.isna(start_dt):
                continue
            
            # Get duration
            duration = row.get('duration', None)
            if pd.isna(duration):
                # If no duration, use a minimal duration (e.g., 0.1 seconds for point events)
                duration = 0.1
            
            # Convert duration to seconds if it's a Timedelta
            if isinstance(duration, pd.Timedelta):
                duration_seconds = duration.total_seconds()
            else:
                duration_seconds = float(duration)
            
            if duration_seconds <= 0:
                duration_seconds = 0.1  # Minimum duration for visibility
            
            # Calculate end datetime
            from datetime import timedelta
            end_dt = start_dt + timedelta(seconds=duration_seconds)
            
            intervals.append((start_dt, end_dt))
        
        return intervals
    
    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Override to use Whisper-specific colors."""
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
        
        # Whisper-specific colors (cyan/teal theme)
        pen = pg.mkPen(color=(50, 200, 255, 255), width=1)
        brush = pg.mkBrush(color=(50, 200, 255, 150))
        
        # Create rectangles
        for start_ts, end_ts in visible_intervals:
            width = end_ts - start_ts
            if width <= 0:
                continue
            
            x = np.array([start_ts, start_ts, end_ts, end_ts, start_ts], dtype=np.float64)
            y = np.array([0, 1, 1, 0, 0], dtype=np.float64)
            
            rect_item = self.plot_widget.plot(x, y, pen=pen, fillLevel=0, fillBrush=brush, brush=brush)
            self.rect_items.append(rect_item)
        
        # Set y-axis range
        self.plot_widget.setYRange(0, 1, padding=0.1)


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
        from phoofflineeeganalysis.analysis.UI.historical_data_timeline import create_timeline_from_xdf_streams
        
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
