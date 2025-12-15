from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized


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
            durations = parse_duration_to_seconds_vectorized(df['duration'])
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

