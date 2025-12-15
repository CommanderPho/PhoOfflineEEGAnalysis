from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized


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
            durations = parse_duration_to_seconds_vectorized(df['duration_sec'])
            
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

