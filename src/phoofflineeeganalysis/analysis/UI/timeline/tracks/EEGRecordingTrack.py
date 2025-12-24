from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized
from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import BaseDatasource, IntervalDataframeDatasource


class EEGRecordingTrack(TrackWidget):
    """
    Track widget for displaying EEG recording intervals from SessionModality.
    
    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """
    
    def __init__(self, eeg_source, name: str = "EEG", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set EEG-specific colors (green/blue theme)
        self._pen_color = (50, 200, 100, 255)
        self._brush_color = (50, 200, 100, 150)
        
        # Normalize input into a datasource and backing DataFrame
        if isinstance(eeg_source, BaseDatasource):
            self.set_datasource(eeg_source)
            df = self._get_full_dataframe()
            self.eeg_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        else:
            eeg_df = eeg_source
            self.eeg_df = eeg_df.copy()
            interval_ds = IntervalDataframeDatasource(self.eeg_df, time_column_name='recording_datetime', datasource_name=name)
            self.set_datasource(interval_ds)
        
        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.eeg_df.columns:
            self.eeg_df['recording_datetime'] = self._ensure_utc_naive(self.eeg_df['recording_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract EEG recording intervals from DataFrame (prefer datasource-backed data)."""
        # Prefer datasource-backed DataFrame when available
        df = self._get_full_dataframe()
        if isinstance(df, pd.DataFrame):
            self.eeg_df = df.copy()
        
        if self.eeg_df.empty or 'recording_datetime' not in self.eeg_df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self.eeg_df.copy()
        start_dt = df['recording_datetime']
        
        # Calculate durations
        durations = pd.Series(np.nan, index=df.index, dtype=float)
        if 'duration_sec_check' in df.columns:
            durations = parse_duration_to_seconds_vectorized(df['duration_sec_check'])
        
        if 'duration_sec' in df.columns:
            durations2 = parse_duration_to_seconds_vectorized(df['duration_sec'])
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
    
    def _clear_detailed_items(self) -> None:
        """No detailed items for EEG track (overview only)."""
        pass
    
    def _ensure_detailed_items(self) -> None:
        """No detailed items for EEG track (overview only)."""
        pass

