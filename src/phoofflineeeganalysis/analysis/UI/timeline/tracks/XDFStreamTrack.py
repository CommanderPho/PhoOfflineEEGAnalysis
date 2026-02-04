from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized
from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import BaseDatasource, IntervalDataframeDatasource, XDFDatasource


class XDFStreamTrack(TrackWidget):
    """
    Generic track widget for displaying XDF stream intervals.
    
    Typically backed by an `XDFDatasource` (or other `BaseDatasource`) whose
    DataFrame provides columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds), can be NaT
    - duration_sec_check: Alternative duration column (optional)
    - first_timestamp_dt: Alternative start time (optional)
    - last_timestamp_dt: Alternative end time (optional)
    """
    
    def __init__(self, stream_source: Union[BaseDatasource, Path], name: str = "Stream", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set stream-specific colors (gray theme)
        self._pen_color = (150, 150, 150, 255)
        self._brush_color = (150, 150, 150, 150)

        # Handle Path input by creating XDFDatasource
        if isinstance(stream_source, (Path, str)):
            xdf_file_path = Path(stream_source) if isinstance(stream_source, str) else stream_source
            stream_source = XDFDatasource(a_xdf_file=xdf_file_path, datasource_name=name)

        # Check if stream_source is a BaseDatasource (handle QObject MRO issues)
        is_valid = False
        try:
            if isinstance(stream_source, (BaseDatasource, IntervalDataframeDatasource)):
                is_valid = True
        except (TypeError, AttributeError):
            # If isinstance fails (e.g., due to QObject MRO issues), check MRO directly
            pass
        
        if not is_valid:
            # Fallback: check by MRO for QObject inheritance issues
            source_type = type(stream_source)
            if hasattr(source_type, '__mro__'):
                mro = source_type.__mro__
                if BaseDatasource in mro or IntervalDataframeDatasource in mro:
                    is_valid = True
            # Also check if it has the required interface (duck typing)
            if hasattr(stream_source, 'df') and (hasattr(stream_source, 'get_updated_data_window') or hasattr(stream_source, 'total_datasource_start_end_times')):
                is_valid = True
        
        if not is_valid:
            raise TypeError(f"XDFStreamTrack expects a BaseDatasource (e.g. XDFDatasource) or Path to XDF file, got {type(stream_source)}")

        # Attach datasource (will trigger initial interval caching and display update)
        self.set_datasource(stream_source)

        # Cache intervals immediately
        self._cache_intervals()

        # Initial display update (show all)
        self.update_display()

        # Cached display DataFrame used for metadata lookups
        self._display_df = pd.DataFrame()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract stream recording intervals from DataFrame (prefer datasource-backed data)."""
        df_full = self._get_full_dataframe()
        if not isinstance(df_full, pd.DataFrame) or df_full.empty:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []

        df = df_full.copy()

        # Ensure datetime columns are datetime type and normalized to UTC-naive
        for col in ['recording_datetime', 'first_timestamp_dt', 'last_timestamp_dt']:
            if col in df.columns:
                df[col] = self._ensure_utc_naive(df[col])
        
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
            durations = parse_duration_to_seconds_vectorized(df['duration_sec_check'])
            valid_mask = durations.notna()
            if valid_mask.any():
                calc_ends = pd.Series(pd.NaT, index=df.index)
                calc_ends[valid_mask] = start_dt[valid_mask] + pd.to_timedelta(durations[valid_mask], unit='s')
                end_dt = end_dt.combine_first(calc_ends)
            
        # 3. Use duration_sec
        if 'duration_sec' in df.columns:
            durations = parse_duration_to_seconds_vectorized(df['duration_sec'])
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
    
    def _clear_detailed_items(self) -> None:
        """No detailed items for XDF stream track (overview only)."""
        pass
    
    def _ensure_detailed_items(self) -> None:
        """No detailed items for XDF stream track (overview only)."""
        pass

