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
"""

from datetime import datetime
from typing import Optional, List, Tuple, Union, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QGraphicsRectItem, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPointF, QEvent
from PyQt5.QtGui import QFont, QPen, QBrush
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ViewBox, DateAxisItem


def _parse_duration_to_seconds_vectorized(series: pd.Series) -> pd.Series:
    """
    Convert duration series to seconds, handling various input types vectorially.
    """
    if series.empty:
        return series
    
    # If already numeric, return as is (coerced to float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
        
    # If timedelta, get total_seconds
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
        
    # Try converting to timedelta first, then seconds
    # This handles strings like '0 days 00:00:19.00'
    try:
        # errors='coerce' will turn invalid parsing into NaT
        deltas = pd.to_timedelta(series, errors='coerce')
        return deltas.dt.total_seconds()
    except Exception:
        # Fallback to numeric conversion
        return pd.to_numeric(series, errors='coerce')


def _parse_duration_to_seconds(duration: Union[pd.Timedelta, float, int, str, None]) -> Optional[float]:
    """Legacy helper for scalar conversion."""
    if duration is None or pd.isna(duration):
        return None
    try:
        if isinstance(duration, pd.Timedelta):
            return duration.total_seconds()
        if isinstance(duration, str):
            return pd.to_timedelta(duration).total_seconds()
        return float(duration)
    except Exception:
        return None


class TrackWidget(QWidget):
    """
    Base class for timeline tracks that display modality-specific data.
    
    Optimized to use pg.BarGraphItem for high-performance rendering.
    """
    
    def __init__(self, name: str, height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        self.track_height = height
        
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
        vb.enableAutoRange(enable=False)
        vb.setLimits(xMin=None, xMax=None, yMin=0, yMax=1)
        
        # Cache all intervals for performance
        self._all_intervals_ts: Optional[np.ndarray] = None  # Cached as [N, 2] array of (start_ts, end_ts)
        
        # Store metadata for each interval (index matches _all_intervals_ts)
        self._interval_metadata: List[Dict[str, Any]] = []
        
        # Single item for rendering all bars
        self.bar_graph_item = pg.BarGraphItem(x=[], height=[], width=[], brush='b')
        self.plot_widget.addItem(self.bar_graph_item)
        
        # Default colors (can be overridden by subclasses)
        self._pen_color = (100, 150, 200, 255)
        self._brush_color = (100, 150, 200, 150)
        
        # Cache pen and brush objects
        self._pen = None
        self._brush = None
        
        # Create label for track name (left edge)
        self.name_label = QLabel(name, self)
        self.name_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.name_label.setFixedWidth(80)
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
        
        # Event handling
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._last_hover_idx = -1
        
    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Legacy method for subclasses."""
        raise NotImplementedError("Subclasses must implement _get_recording_intervals() or _get_recording_intervals_vectorized()")
        
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Return cached intervals and metadata.
        Can be overridden by subclasses for performance.
        Default implementation calls _get_recording_intervals() (legacy).
        """
        # Fallback to legacy loop-based method
        intervals = self._get_recording_intervals()
        
        # Force cache metadata via legacy method if not matching
        # (The legacy _get_recording_intervals usually populated _interval_metadata via _cache_intervals logic,
        # but here we need to ensure metadata is ready)
        # Actually _cache_intervals called _get_recording_intervals then _cache_metadata.
        # So we should call _cache_metadata here if we rely on legacy.
        self._cache_metadata() # Populates self._interval_metadata in legacy subclass
        metadata = self._interval_metadata
        
        if not intervals:
            return np.empty((0, 2)), []
            
        n = len(intervals)
        starts = np.empty(n, dtype=np.float64)
        ends = np.empty(n, dtype=np.float64)
        
        for i, (s, e) in enumerate(intervals):
            starts[i] = s.timestamp() if isinstance(s, datetime) else float(s)
            ends[i] = e.timestamp() if isinstance(e, datetime) else float(e)
            
        return np.column_stack([starts, ends]), metadata
    
    def _cache_intervals(self):
        """Cache intervals as timestamps for fast filtering."""
        intervals_ts, metadata = self._get_recording_intervals_vectorized()
        self._all_intervals_ts = intervals_ts
        self._interval_metadata = metadata
        
        if self._all_intervals_ts is not None and len(self._all_intervals_ts) > 0:
             if self._all_intervals_ts.ndim != 2 or self._all_intervals_ts.shape[1] != 2:
                 self._all_intervals_ts = None
                 self._interval_metadata = []
    
    def _cache_metadata(self):
        """Legacy metadata method."""
        pass # Implemented by subclasses
    
    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        if 0 <= interval_index < len(self._interval_metadata):
            return self._interval_metadata[interval_index]
        return {}
    
    def _ensure_utc_naive(self, series: pd.Series) -> pd.Series:
        """
        Normalize a datetime Series to naive UTC.
        - If aware: convert to UTC, then make naive.
        - If naive: assume Local Time, localize to system timezone, convert to UTC, then make naive.
        """
        if series.empty:
            return series

        # Convert to datetime first to ensure properties exist
        series = pd.to_datetime(series, errors='coerce')
        
        # specific check for naive vs aware is tricky on a Series if mixed, 
        # but generally we expect a column to be consistent.
        # However, checking the first non-null value is a good heuristic.
        first_valid = series.dropna().first_valid_index()
        if first_valid is None:
            return series
            
        first_val = series[first_valid]
        if first_val.tzinfo is None:
            # Naive -> Assume Local -> UTC
            # Get system local timezone
            local_tz = datetime.now().astimezone().tzinfo
            return series.dt.tz_localize(local_tz).dt.tz_convert('UTC').dt.tz_convert(None)
        else:
            # Aware -> UTC -> Naive
            return series.dt.tz_convert('UTC').dt.tz_convert(None)

    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        if self._all_intervals_ts is None:
            self._cache_intervals()
            
        # Helper to setup colors if needed
        if self._pen is None:
            self._pen = pg.mkPen(self._pen_color)
            self._brush = pg.mkBrush(self._brush_color) # pg.mkBrush handles (r,g,b,a) tuple
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            return
            
        visible_intervals = self._all_intervals_ts
        
        if time_range is not None:
            start_dt, end_dt = time_range
            start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
            end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
            
            mask = (self._all_intervals_ts[:, 0] <= end_ts) & (self._all_intervals_ts[:, 1] >= start_ts)
            visible_intervals = self._all_intervals_ts[mask]
        
        if len(visible_intervals) == 0:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            return
            
        # Robust filtering: Check for NaNs, Infs, and valid width
        starts = visible_intervals[:, 0]
        ends = visible_intervals[:, 1]
        
        # Check for finiteness (no NaNs or Infs)
        finite_mask = np.isfinite(starts) & np.isfinite(ends)
        
        # Check for valid time order
        order_mask = ends > starts
        
        valid_mask = finite_mask & order_mask
        valid_intervals = visible_intervals[valid_mask]
        
        if len(valid_intervals) > 0:
            v_starts = valid_intervals[:, 0]
            v_ends = valid_intervals[:, 1]
            widths = v_ends - v_starts
            # Center x at start + width/2
            centers = v_starts + (widths / 2.0)
            
        if len(valid_intervals) > 0:
            v_starts = valid_intervals[:, 0]
            v_ends = valid_intervals[:, 1]
            widths = v_ends - v_starts
            # Center x at start + width/2
            centers = v_starts + (widths / 2.0)
            
            self.bar_graph_item.setOpts(
                x=centers,
                height=np.ones_like(centers),
                width=widths,
                brush=self._brush,
                pen=self._pen
            )
        else:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            
        self.plot_widget.setYRange(0, 1, padding=0.0)

    def get_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        if self._all_intervals_ts is None:
            self._cache_intervals()
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            return None
        
        start_ts = np.min(self._all_intervals_ts[:, 0])
        end_ts = np.max(self._all_intervals_ts[:, 1])
        
        return (datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts))
        
    def _find_interval_at_pos(self, x_pos: float) -> int:
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            return -1
        
        starts = self._all_intervals_ts[:, 0]
        ends = self._all_intervals_ts[:, 1]
        mask = (starts <= x_pos) & (ends >= x_pos)
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            return indices[-1]
        return -1

    def _on_mouse_moved(self, pos):
        if self._all_intervals_ts is None:
            return

        # Map to view
        if not self.plot_widget.sceneBoundingRect().contains(pos):
             return
        
        vb = self.plot_widget.getViewBox()
        mouse_point = vb.mapSceneToView(pos)
        x_ts = mouse_point.x()
        y_val = mouse_point.y()
        
        if not (0 <= y_val <= 1):
             self.plot_widget.setToolTip("")
             self._last_hover_idx = -1
             return
             
        idx = self._find_interval_at_pos(x_ts)
        
        if idx != self._last_hover_idx:
            self._last_hover_idx = idx
            if idx != -1:
                metadata = self._get_metadata_for_interval(idx)
                if metadata:
                    start_ts = self._all_intervals_ts[idx, 0]
                    end_ts = self._all_intervals_ts[idx, 1]
                    tooltip = self._format_tooltip(metadata, start_ts, end_ts)
                    self.plot_widget.setToolTip(tooltip)
                else:
                    self.plot_widget.setToolTip("")
            else:
                self.plot_widget.setToolTip("")

    def _on_mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
             vb = self.plot_widget.getViewBox()
             scene_pos = event.scenePos()
             if self.plot_widget.sceneBoundingRect().contains(scene_pos):
                 mouse_point = vb.mapSceneToView(scene_pos)
                 x_ts = mouse_point.x()
                 y_val = mouse_point.y()
                 
                 if 0 <= y_val <= 1:
                     idx = self._find_interval_at_pos(x_ts)
                     if idx != -1:
                        metadata = self._get_metadata_for_interval(idx)
                        start_ts = self._all_intervals_ts[idx, 0]
                        end_ts = self._all_intervals_ts[idx, 1]
                        self._show_metadata_dialog(metadata, start_ts, end_ts)
                        event.accept()

    def _format_tooltip(self, metadata: Dict[str, Any], start_ts: float, end_ts: float) -> str:
        lines = []
        filename = metadata.get('filename', metadata.get('file_path', ''))
        if filename:
            if isinstance(filename, (str, Path)):
                filename = Path(filename).name if filename else ''
            if filename:
                lines.append(f"File: {filename}")
        
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
        lines.append(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {end_dt - start_dt}")
        
        for k in ['duration_sec', 'fps', 'resolution']:
            if k in metadata and metadata[k]:
                label = k.replace('_', ' ').title()
                lines.append(f"{label}: {metadata[k]}")

        return '\n'.join(lines)
    
    def _show_metadata_dialog(self, metadata: Dict[str, Any], start_ts: float, end_ts: float):
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
        
        lines = [f"<b>{self.name} Recording Details</b>", ""]
        lines.append(f"<b>Time Range:</b>")
        lines.append(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        lines.append(f"  End: {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        lines.append(f"  Duration: {end_dt - start_dt}")
        lines.append("")
        
        if metadata:
            lines.append("<b>Metadata:</b>")
            for key, value in sorted(metadata.items()):
                if value is not None and value != '':
                    display_key = key.replace('_', ' ').title()
                    lines.append(f"  {display_key}: {value}")
        
        QMessageBox.information(self, f"{self.name} Recording Details", '\n'.join(lines))


class VideoMetadataTrack(TrackWidget):
    """
    Track widget for displaying video recording intervals from VideoMetadataParser.
    
    Expects a DataFrame with columns:
    - video_start_datetime: datetime
    - video_end_datetime: datetime
    """
    
    def __init__(self, video_df: pd.DataFrame, name: str = "Videos", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set video-specific colors (blue theme)
        self._pen_color = (100, 150, 200, 255)
        self._brush_color = (100, 150, 200, 150)
        
        # Store original df
        self.video_df = video_df.copy()
        self._display_df = pd.DataFrame() # Filtered and processed for display
        
        # Ensure datetime columns are datetime type and normalized
        if 'video_start_datetime' in self.video_df.columns:
            self.video_df['video_start_datetime'] = self._ensure_utc_naive(self.video_df['video_start_datetime'])
        if 'video_end_datetime' in self.video_df.columns:
            self.video_df['video_end_datetime'] = self._ensure_utc_naive(self.video_df['video_end_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract video recording intervals from DataFrame using vectorized operations."""
        if self.video_df.empty or 'video_start_datetime' not in self.video_df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.video_df.copy()
        
        # Calculate end times if missing
        start_dt = df['video_start_datetime']
        end_dt = df['video_end_datetime'] if 'video_end_datetime' in df.columns else pd.Series(pd.NaT, index=df.index)
        
        # If end_dt is NaT, try video_duration
        if 'video_duration' in df.columns:
            # Coerce duration to numeric seconds
            durations = pd.to_numeric(df['video_duration'], errors='coerce')
            # Calculate end from duration where end_dt is null
            calc_ends = start_dt + pd.to_timedelta(durations, unit='s')
            end_dt = end_dt.combine_first(calc_ends)
            
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna() & (end_dt > start_dt)
        self._display_df = df[mask].reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
            
        # Create numpy array of timestamps
        starts = self._display_df['video_start_datetime'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        # Recalculate ends for display_df (since we reset index and combined logic above was on original df)
        # Actually safer to recompute ends on the filtered df or just add the col
        
        # Let's clean up: add computed 'final_end_dt' to df before filtering?
        # Yes, that's better.
        df['final_end_dt'] = end_dt
        self._display_df = df[mask].copy().reset_index(drop=True)
        
        starts = self._display_df['video_start_datetime'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), [] 
    
    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from dataframe."""
        if interval_index < 0 or interval_index >= len(self._display_df):
            return {}
            
        row = self._display_df.iloc[interval_index]
        metadata = {}
        
        # Extract filename from video_file_path
        if 'video_file_path' in row and pd.notna(row['video_file_path']):
            file_path = row['video_file_path']
            metadata['file_path'] = str(file_path)
            metadata['filename'] = Path(file_path).name
        
        # Extract other metadata
        if 'video_duration' in row and pd.notna(row['video_duration']):
            metadata['duration_sec'] = row['video_duration']
        
        if 'video_fps' in row and pd.notna(row['video_fps']):
            metadata['fps'] = row['video_fps']
        
        if 'video_width' in row and pd.notna(row['video_width']) and 'video_height' in row and pd.notna(row['video_height']):
            metadata['resolution'] = f"{int(row['video_width'])}x{int(row['video_height'])}"
        
        if 'video_file_size' in row and pd.notna(row['video_file_size']):
            metadata['file_size'] = row['video_file_size']
            
        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        return [] # Obsolete, but kept to satisfy abstract method if needed (shim handles it)


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
        # Set stream-specific colors (gray theme)
        self._pen_color = (150, 150, 150, 255)
        self._brush_color = (150, 150, 150, 150)
        
        self.stream_df = stream_df.copy()
        self._display_df = pd.DataFrame()
        
        # Ensure datetime columns are datetime type and normalized
        if 'recording_datetime' in self.stream_df.columns:
            self.stream_df['recording_datetime'] = self._ensure_utc_naive(self.stream_df['recording_datetime'])
        if 'first_timestamp_dt' in self.stream_df.columns:
            self.stream_df['first_timestamp_dt'] = self._ensure_utc_naive(self.stream_df['first_timestamp_dt'])
        if 'last_timestamp_dt' in self.stream_df.columns:
            self.stream_df['last_timestamp_dt'] = self._ensure_utc_naive(self.stream_df['last_timestamp_dt'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract stream recording intervals from DataFrame using vectorized operations."""
        if self.stream_df.empty:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.stream_df.copy()
        
        # Calculate start times
        start_dt = df['recording_datetime'] if 'recording_datetime' in df.columns else pd.Series(pd.NaT, index=df.index)
        if 'first_timestamp_dt' in df.columns:
            start_dt = start_dt.combine_first(df['first_timestamp_dt'])
            
        # Initialize end_dt
        end_dt = pd.Series(pd.NaT, index=df.index)
        
        # 1. Use last_timestamp_dt
        if 'last_timestamp_dt' in df.columns:
            end_dt = df['last_timestamp_dt']
            
        # 2. Use duration_sec_check
        if 'duration_sec_check' in df.columns:
            durations = _parse_duration_to_seconds_vectorized(df['duration_sec_check'])
            valid_mask = durations.notna()
            if valid_mask.any():
                calc_ends = pd.Series(pd.NaT, index=df.index)
                calc_ends[valid_mask] = start_dt[valid_mask] + pd.to_timedelta(durations[valid_mask], unit='s')
                end_dt = end_dt.combine_first(calc_ends)
            
        # 3. Use duration_sec
        if 'duration_sec' in df.columns:
            durations = _parse_duration_to_seconds_vectorized(df['duration_sec'])
            valid_mask = durations.notna()
            if valid_mask.any():
                calc_ends = pd.Series(pd.NaT, index=df.index)
                calc_ends[valid_mask] = start_dt[valid_mask] + pd.to_timedelta(durations[valid_mask], unit='s')
                end_dt = end_dt.combine_first(calc_ends)
            
        # 4. Fallback for markers
        is_marker = pd.Series(False, index=df.index)
        if 'type' in df.columns:
            is_marker |= df['type'] == 'Markers'
        if 'name' in df.columns:
            is_marker |= df['name'] == 'TextLogger'
            
        if is_marker.any():
            marker_ends = start_dt + pd.Timedelta(seconds=0.1)
            # Only apply marker default where end is still NaT AND it is a marker
            end_dt = end_dt.combine_first(marker_ends.where(is_marker))
            
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna() & (end_dt > start_dt)
        
        # Save filtered df with computed ends
        df['video_start_datetime'] = start_dt # standardized col name for ease or just keep original? 
        # Actually let's use standardized names for display_df so access is easier? 
        # But _get_metadata_for_interval needs original columns.
        # So I'll just add 'final_start_dt' and 'final_end_dt'
        df['final_start_dt'] = start_dt
        df['final_end_dt'] = end_dt
        
        self._display_df = df[mask].reset_index(drop=True)
        
        if self._display_df.empty:
             return np.empty((0, 2)), []
             
        starts = self._display_df['final_start_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from XDF stream DataFrame."""
        if interval_index < 0 or interval_index >= len(self._display_df):
            return {}
            
        row = self._display_df.iloc[interval_index]
        metadata = {}
        
        # Extract filename from xdf_filename
        if 'xdf_filename' in row and pd.notna(row['xdf_filename']):
            metadata['xdf_filename'] = row['xdf_filename']
            metadata['filename'] = row['xdf_filename']
        
        # Extract processed filenames
        if 'proccessed_fif_filename' in row and pd.notna(row['proccessed_fif_filename']):
            metadata['fif_filename'] = row['proccessed_fif_filename']
        
        if 'proccessed_mat_filename' in row and pd.notna(row['proccessed_mat_filename']):
            metadata['mat_filename'] = row['proccessed_mat_filename']
        
        # Extract stream name and type
        if 'name' in row and pd.notna(row['name']):
            metadata['stream_name'] = row['name']
        
        if 'type' in row and pd.notna(row['type']):
            metadata['stream_type'] = row['type']
        
        # Extract duration
        duration = row.get('duration_sec_check', row.get('duration_sec', None))
        if pd.notna(duration):
             metadata['duration_sec'] = duration
        
        # Extract sampling rate if available
        if 'fs' in row and pd.notna(row['fs']):
            metadata['sampling_rate'] = row['fs']
            
        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        return []
    
    def _cache_metadata(self):
        pass


class EEGRecordingTrack(TrackWidget):
    """
    Track widget for displaying EEG recording intervals from SessionModality.
    
    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """
    
    def __init__(self, eeg_df: pd.DataFrame, name: str = "EEG", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set EEG-specific colors (green/blue theme)
        self._pen_color = (50, 200, 100, 255)
        self._brush_color = (50, 200, 100, 150)
        
        self.eeg_df = eeg_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.eeg_df.columns:
            self.eeg_df['recording_datetime'] = self._ensure_utc_naive(self.eeg_df['recording_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract EEG recording intervals from DataFrame using vectorized operations."""
        if self.eeg_df.empty or 'recording_datetime' not in self.eeg_df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.eeg_df.copy()
        start_dt = df['recording_datetime']
        
        # Calculate durations
        durations = pd.Series(np.nan, index=df.index, dtype=float)
        if 'duration_sec_check' in df.columns:
            durations = _parse_duration_to_seconds_vectorized(df['duration_sec_check'])
        
        if 'duration_sec' in df.columns:
            durations2 = _parse_duration_to_seconds_vectorized(df['duration_sec'])
            durations = durations.combine_first(durations2)
            
        # Calculate ends
        end_dt = pd.Series(pd.NaT, index=df.index)
        valid_dur_mask = durations.notna()
        if valid_dur_mask.any():
            end_dt[valid_dur_mask] = start_dt[valid_dur_mask] + pd.to_timedelta(durations[valid_dur_mask], unit='s')
        
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna() & (end_dt > start_dt)
        self._display_df = df[mask].copy().reset_index(drop=True)
        self._display_df['final_end_dt'] = end_dt[mask].reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
            
        starts = self._display_df['recording_datetime'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from EEG DataFrame."""
        if interval_index < 0 or interval_index >= len(self._display_df):
            return {}
            
        row = self._display_df.iloc[interval_index]
        metadata = {}
        
        # Extract duration
        duration = row.get('duration_sec_check', row.get('duration_sec', None))
        if pd.notna(duration):
             metadata['duration_sec'] = duration
        
        # Extract sampling rate if available
        if 'fs' in row and pd.notna(row['fs']):
            metadata['sampling_rate'] = row['fs']
        
        # Extract xdf filename if available
        if 'xdf_filename' in row and pd.notna(row['xdf_filename']):
            metadata['xdf_filename'] = row['xdf_filename']
            metadata['filename'] = row['xdf_filename']
        
        # Extract processed filenames
        if 'proccessed_fif_filename' in row and pd.notna(row['proccessed_fif_filename']):
            metadata['fif_filename'] = row['proccessed_fif_filename']
            
        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        return []
        
    def _cache_metadata(self):
        pass


class MotionRecordingTrack(TrackWidget):
    """
    Track widget for displaying motion recording intervals from SessionModality.
    
    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """
    
    def __init__(self, motion_df: pd.DataFrame, name: str = "Motion", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set motion-specific colors (orange/red theme)
        self._pen_color = (255, 150, 50, 255)
        self._brush_color = (255, 150, 50, 150)
        
        self.motion_df = motion_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.motion_df.columns:
            self.motion_df['recording_datetime'] = self._ensure_utc_naive(self.motion_df['recording_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract motion recording intervals from DataFrame using vectorized operations."""
        if self.motion_df.empty or 'recording_datetime' not in self.motion_df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.motion_df.copy()
        start_dt = df['recording_datetime']
        
        durations = pd.Series(np.nan, index=df.index, dtype=float)
        if 'duration_sec' in df.columns:
            durations = _parse_duration_to_seconds_vectorized(df['duration_sec'])
            
        end_dt = pd.Series(pd.NaT, index=df.index)
        valid_dur_mask = durations.notna()
        if valid_dur_mask.any():
            end_dt[valid_dur_mask] = start_dt[valid_dur_mask] + pd.to_timedelta(durations[valid_dur_mask], unit='s')
        
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna() & (end_dt > start_dt)
        self._display_df = df[mask].copy().reset_index(drop=True)
        self._display_df['final_end_dt'] = end_dt[mask].reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
            
        starts = self._display_df['recording_datetime'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from motion DataFrame."""
        if interval_index < 0 or interval_index >= len(self._display_df):
            return {}
            
        row = self._display_df.iloc[interval_index]
        metadata = {}
        
        # Extract duration
        if 'duration_sec' in row and pd.notna(row['duration_sec']):
            metadata['duration_sec'] = row['duration_sec']
        
        # Extract sampling rate if available
        if 'fs' in row and pd.notna(row['fs']):
            metadata['sampling_rate'] = row['fs']
        
        # Extract xdf filename if available
        if 'xdf_filename' in row and pd.notna(row['xdf_filename']):
            metadata['xdf_filename'] = row['xdf_filename']
            metadata['filename'] = row['xdf_filename']
            
        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
         return []

    def _cache_metadata(self):
        pass


class PhoLogTrack(TrackWidget):
    """
    Track widget for displaying PHO_LOG_TO_LSL annotation intervals.
    
    Expects a DataFrame with columns:
    - onset: datetime (start time)
    - duration: float or Timedelta (duration in seconds)
    """
    
    def __init__(self, pho_log_df: pd.DataFrame, name: str = "PHO_LOG", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set PHO_LOG-specific colors (purple theme)
        self._pen_color = (200, 100, 255, 255)
        self._brush_color = (200, 100, 255, 150)
        
        self.pho_log_df = pho_log_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'onset' in self.pho_log_df.columns:
            self.pho_log_df['onset'] = self._ensure_utc_naive(self.pho_log_df['onset'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract PHO_LOG annotation intervals from DataFrame using vectorized operations."""
        if self.pho_log_df.empty or 'onset' not in self.pho_log_df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.pho_log_df.copy()
        start_dt = df['onset']
        
        # Calculate END times
        if 'duration' in df.columns:
            durations = _parse_duration_to_seconds_vectorized(df['duration'])
        else:
            durations = pd.Series(0.0, index=df.index)
            
        durations = durations.fillna(0.1)
        durations[durations <= 0] = 0.1
        
        end_dt = start_dt + pd.to_timedelta(durations, unit='s')
        
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna()
        self._display_df = df[mask].copy().reset_index(drop=True)
        self._display_df['final_end_dt'] = end_dt[mask].reset_index(drop=True)
        # Store duration for metadata use
        self._display_df['final_duration'] = durations[mask].reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
            
        starts = self._display_df['onset'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from PhoLog DataFrame."""
        if interval_index < 0 or interval_index >= len(self._display_df):
            return {}
            
        row = self._display_df.iloc[interval_index]
        metadata = {}
        
        # Extract duration from computed final_duration or source
        if 'final_duration' in row:
            metadata['duration_sec'] = row['final_duration']
        elif 'duration' in row and pd.notna(row['duration']):
             metadata['duration_sec'] = row['duration']
            
        # Add message/log info
        if 'message' in row:
             metadata['message'] = str(row['message'])
        if 'label' in row:
             metadata['label'] = str(row['label'])
            
        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
         return []

    def _cache_metadata(self):
        pass


class WhisperTrack(TrackWidget):
    """
    Track widget for displaying Whisper transcript intervals.
    
    Expects a DataFrame with columns:
    - onset: datetime (start time)
    - duration: float or Timedelta (duration in seconds)
    """
    
    def __init__(self, whisper_df: pd.DataFrame, name: str = "Whisper", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set Whisper-specific colors (cyan/teal theme)
        self._pen_color = (50, 200, 255, 255)
        self._brush_color = (50, 200, 255, 150)
        
        self.whisper_df = whisper_df.copy()
        
        # Ensure datetime columns are datetime type
        if 'onset' in self.whisper_df.columns:
            self.whisper_df['onset'] = self._ensure_utc_naive(self.whisper_df['onset'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract Whisper transcript intervals from DataFrame using vectorized operations."""
        if self.whisper_df.empty or 'onset' not in self.whisper_df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.whisper_df.copy()
        start_dt = df['onset']
        
        # Calculate END times
        if 'duration' in df.columns:
            durations = _parse_duration_to_seconds_vectorized(df['duration'])
        else:
            durations = pd.Series(0.0, index=df.index)
            
        durations = durations.fillna(0.1)
        durations[durations <= 0] = 0.1
        
        end_dt = start_dt + pd.to_timedelta(durations, unit='s')
        
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna()
        self._display_df = df[mask].copy().reset_index(drop=True)
        self._display_df['final_end_dt'] = end_dt[mask].reset_index(drop=True)
        self._display_df['final_duration'] = durations[mask].reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
            
        starts = self._display_df['onset'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from Whisper DataFrame."""
        if interval_index < 0 or interval_index >= len(self._display_df):
            return {}
            
        row = self._display_df.iloc[interval_index]
        metadata = {}
        
        if 'final_duration' in row:
            metadata['duration_sec'] = row['final_duration']
        
        # Extract transcript text if available
        if 'text' in row and pd.notna(row['text']):
            text = str(row['text'])
            metadata['text'] = text
            # Use first part of text as preview
            if len(text) > 50:
                metadata['text_preview'] = text[:50] + '...'
            else:
                metadata['text_preview'] = text
        
        # Extract language if available
        if 'language' in row and pd.notna(row['language']):
            metadata['language'] = row['language']
            
        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
         return []

    def _cache_metadata(self):
        pass


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

    
    # Create timeline
    # timeline = create_timeline_widget(video_df=video_df)

    output_folder = Path('output').resolve()
    assert output_folder.exists()

    timeline: TimelineWidget = TimelineWidget()

    csv_save_path = output_folder.joinpath('2025-12-09_parsed_videos.csv').resolve()
    assert csv_save_path.exists()
    video_df: pd.DataFrame = pd.read_csv(csv_save_path)

    csv_save_path = output_folder.joinpath('2025-12-10_all_xdf_stream_infos.csv').resolve()
    assert csv_save_path.exists()
    all_xdf_stream_infos_df: pd.DataFrame = pd.read_csv(csv_save_path)

    # Add tracks from different modalities
    timeline.add_track(VideoMetadataTrack(video_df))
    timeline.add_tracks_from_xdf_streams(all_xdf_stream_infos_df)
    timeline.setWindowTitle("Historical Data Timeline")
    timeline.resize(1900, 800)
    timeline.show()
    
    sys.exit(app.exec_())
