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
from phoofflineeeganalysis.analysis.UI.timeline.tracks.StringDataTrack import StringDataTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.PhoLogTrack import PhoLogTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.VideoMetadataTrack import VideoMetadataTrack
from phoofflineeeganalysis.analysis.UI.timeline.tracks.XDFStreamTrack import XDFStreamTrack


def _normalize_datetime_to_utc_naive(series: pd.Series) -> pd.Series:
    """
    Normalize a datetime Series to naive UTC.
    - If aware: convert to UTC, then make naive.
    - If naive: assume Local Time, localize to system timezone, convert to UTC, then make naive.
    """
    if series.empty:
        return series

    # Convert to datetime first to ensure properties exist
    series = pd.to_datetime(series, errors='coerce')
    
    # Check the first non-null value to determine if aware or naive
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


class TimelineWidget(QWidget):
    """
    Main timeline widget that displays multiple synchronized tracks.
    
    All tracks share the same x-axis (datetime) and can be zoomed/panned together.

    Usage:

        from phoofflineeeganalysis.analysis.UI.timeline.TimelineWidget import TimelineWidget
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
        
        self.setWindowTitle('Timeline')
        
        # Set default size to be at least 8x bigger (3200x2400)
        self.resize(1980, 800)

    def show(self):
        """Show the timeline widget and bring it to the foreground."""
        super().show()
        self.raise_()
        self.activateWindow()

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
            start_dt = self.overall_time_range[0]
            end_dt = self.overall_time_range[1]
            start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
            end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
            
            # Only set range if timestamps are valid
            if start_ts is not None and end_ts is not None and start_ts < end_ts:
                for track_widget in self.tracks:
                    track_widget.plot_widget.setXRange(start_ts, end_ts, padding=0.05)
                    # Explicitly update display to ensure intervals are rendered with the correct view range
                    track_widget.update_display(time_range=(start_dt, end_dt))
    
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
        
        # Convert timestamps back to datetime (safely handle Windows OSError)
        try:
            start_dt = datetime.fromtimestamp(start_ts)
            end_dt = datetime.fromtimestamp(end_ts)
        except (OSError, ValueError, OverflowError):
            # Invalid timestamp range, skip update
            return
        
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
        try:
            start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
            end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
        except (OSError, ValueError, OverflowError):
            # Invalid datetime range, skip
            return
        
        if self.shared_viewbox is not None and start_ts is not None and end_ts is not None and start_ts < end_ts:
            self.shared_viewbox.setXRange(start_ts, end_ts, padding=0.05)
    
    def zoom_to_fit(self):
        """Zoom to show all data."""
        if self.overall_time_range is not None:
            self.set_time_range(self.overall_time_range[0], self.overall_time_range[1])
    
    def add_tracks_from_xdf_streams(self, xdf_stream_infos_df: pd.DataFrame, stream_names: Optional[List[str]] = None, fail_on_exception: bool=False):
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
            'TextLogger': ('PHO_LOG', StringDataTrack),
            'EventBoard': ('PHO_LOG', StringDataTrack),
            'Epoc X eQuality': ('EEG Quality', EEGRecordingTrack),  # Use EEG track for quality streams
        }
        
        # Also check by type column as fallback
        type_to_track = {
            'EEG': ('EEG', EEGRecordingTrack),
            'SIGNAL': ('Motion', MotionRecordingTrack),
            'Markers': ('PHO_LOG', StringDataTrack),
            # 'Markers': ('PHO_LOG', PhoLogTrack),
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
            else:
                ## in general, the stream_name should be 
                track_name = f"{stream_name}<{track_name}>"

            # Normalize column names to match track class expectations
            # Map recording_start_datetime or stream_start_datetime to recording_datetime if needed
            if 'recording_datetime' not in stream_df.columns:
                if 'recording_start_datetime' in stream_df.columns:
                    stream_df['recording_datetime'] = stream_df['recording_start_datetime']
                elif 'stream_start_datetime' in stream_df.columns:
                    stream_df['recording_datetime'] = stream_df['stream_start_datetime']
                elif 'first_timestamp_dt' in stream_df.columns:
                    stream_df['recording_datetime'] = stream_df['first_timestamp_dt']
            
            # Normalize datetime columns to UTC-naive to avoid timezone comparison issues
            datetime_columns = ['recording_datetime', 'first_timestamp_dt', 'last_timestamp_dt', 
                               'recording_start_datetime', 'stream_start_datetime']
            for col in datetime_columns:
                if col in stream_df.columns:
                    stream_df[col] = _normalize_datetime_to_utc_naive(stream_df[col])
            
            # Calculate duration_sec if not present but we have timestamp columns
            if 'duration_sec' not in stream_df.columns and 'duration_sec_check' not in stream_df.columns:
                if 'first_timestamp_dt' in stream_df.columns and 'last_timestamp_dt' in stream_df.columns:
                    # Calculate duration from timestamps (now both are UTC-naive)
                    durations = (stream_df['last_timestamp_dt'] - stream_df['first_timestamp_dt']).dt.total_seconds()
                    stream_df['duration_sec'] = durations
                elif 'first_timestamp' in stream_df.columns and 'last_timestamp' in stream_df.columns:
                    # Calculate duration from numeric timestamps
                    durations = stream_df['last_timestamp'] - stream_df['first_timestamp']
                    stream_df['duration_sec'] = durations
                elif 'n_samples' in stream_df.columns and 'fs' in stream_df.columns:
                    # Calculate duration from sample count and sampling rate
                    durations = stream_df['n_samples'].astype(float) / stream_df['fs'].astype(float)
                    stream_df['duration_sec'] = durations

            # Check if StringDataTrack has required columns
            if track_class == StringDataTrack:
                # StringDataTrack requires 'onset' column (or the default onset_col)
                # Check if we have any time-like column that could be used
                has_onset = 'onset' in stream_df.columns
                has_time_column = any(col in stream_df.columns for col in ['recording_datetime', 'stream_start_datetime', 'recording_start_datetime', 'first_timestamp_dt'])
                
                if not has_onset and not has_time_column:
                    # Skip this stream - StringDataTrack requires a time column
                    print(f"Warning: Skipping stream '{stream_name}' - StringDataTrack requires 'onset' column or time column. Available columns: {list(stream_df.columns)}")
                    continue
                elif not has_onset and has_time_column:
                    # Map a time column to 'onset' for StringDataTrack
                    if 'recording_datetime' in stream_df.columns:
                        stream_df['onset'] = stream_df['recording_datetime']
                    elif 'stream_start_datetime' in stream_df.columns:
                        stream_df['onset'] = stream_df['stream_start_datetime']
                    elif 'recording_start_datetime' in stream_df.columns:
                        stream_df['onset'] = stream_df['recording_start_datetime']
                    elif 'first_timestamp_dt' in stream_df.columns:
                        stream_df['onset'] = stream_df['first_timestamp_dt']

            # if 'time' not in stream_df:
            #     stream_df = stream_df.rename(columns={'onset':'time', 'description':'text'}, inplace=False)

            # Create track
            try:
                # For XDFStreamTrack, wrap DataFrame in IntervalDataframeDatasource
                if track_class == XDFStreamTrack:
                    from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import IntervalDataframeDatasource
                    interval_ds = IntervalDataframeDatasource(stream_df, time_column_name='recording_datetime', datasource_name=track_name)
                    track = track_class(interval_ds, name=track_name)
                else:
                    track = track_class(stream_df, name=track_name)
                self.add_track(track)

            # except ValueError as e:
            #     stream_df = stream_df.rename(columns={'onset':'time', 'description':'text'}, inplace=False)
            #     track = track_class(stream_df, name=track_name)
            #     self.add_track(track)


            except Exception as e:
                # Skip streams that fail to create tracks
                print(f"Warning: Failed to create track for stream '{stream_name}': {e}")
                if fail_on_exception:
                    raise
                else:
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

