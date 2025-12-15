from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget


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
