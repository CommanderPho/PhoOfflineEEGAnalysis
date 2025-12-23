from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized
from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import BaseDatasource, IntervalDataframeDatasource


class MotionRecordingTrack(TrackWidget):
    """
    Track widget for displaying motion recording intervals from SessionModality.

    Overview mode renders interval bars (summary view).
    Detailed mode renders per-channel motion data (AccX/Y/Z, GyroX/Y/Z) in 6 compact sub-rows
    when the visible time-span is sufficiently small.

    Expects a DataFrame with columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """

    def __init__(self, motion_source, name: str = "Motion", height: int = 60, parent: Optional[QWidget] = None, detailed_data_provider: Optional[Callable[[Dict[str, Any], Tuple[float, float]], Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set motion-specific colors (orange/red theme)
        self._pen_color = (255, 150, 50, 255)
        self._brush_color = (255, 150, 50, 150)

        # Normalize input into a datasource and backing DataFrame
        if isinstance(motion_source, BaseDatasource):
            self.set_datasource(motion_source)
            df = self._get_full_dataframe()
            self.motion_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        else:
            motion_df = motion_source
            self.motion_df = motion_df.copy()
            interval_ds = IntervalDataframeDatasource(self.motion_df, time_column_name='recording_datetime', datasource_name=name)
            self.set_datasource(interval_ds)

        # Detailed data provider: (metadata, (start_ts, end_ts)) -> {channel: (times, values)}
        self.detailed_data_provider = detailed_data_provider

        # Configure detailed rendering behavior
        # Default: enable detailed mode when visible span <= 10 seconds
        self.set_detailed_threshold(10.0)
        self._detailed_channels: List[str] = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
        self._detailed_curves: Dict[str, pg.PlotDataItem] = {}

        # Ensure datetime columns are datetime type
        if 'recording_datetime' in self.motion_df.columns:
            self.motion_df['recording_datetime'] = self._ensure_utc_naive(self.motion_df['recording_datetime'])
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract motion recording intervals from DataFrame (prefer datasource-backed data)."""
        # Prefer datasource-backed DataFrame when available
        df = self._get_full_dataframe()
        if isinstance(df, pd.DataFrame):
            self.motion_df = df.copy()

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
        metadata: Dict[str, Any] = {}

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

        # Keep a reference to the original row index within motion_df if present
        if 'index' in self._display_df.columns:
            metadata['row_index'] = self._display_df.index[interval_index]

        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        # Handled by _get_recording_intervals_vectorized
        return []

    def _cache_metadata(self):
        # Vectorized path stores metadata lazily in _get_metadata_for_interval
        pass

    # ---- Detailed rendering -------------------------------------------------
    def _load_motion_timeseries(self, metadata: Dict[str, Any], window_ts: Tuple[float, float]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load motion timeseries for the given interval metadata and time window.

        This delegates to detailed_data_provider if provided. The provider is
        expected to return a dict mapping channel name to (times, values).
        """
        if self.detailed_data_provider is None:
            return {}
        return self.detailed_data_provider(metadata, window_ts)


    # ==================================================================================================================================================================================================================================================================================== #
    # DetailedRenderingTrackMixin Implementation                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def set_detailed_threshold(self, seconds: Optional[float]) -> None:
        """Set the time-span threshold (in seconds) for switching to detailed rendering."""
        self.detailed_mode_timespan_threshold_sec = seconds


    def _ensure_detailed_items(self) -> None:
        """Create PlotDataItem per channel for detailed mode if not already present."""
        if self._detailed_curves:
            return

        # Simple color palette: accelerometer vs gyro
        acc_color = (255, 120, 80)
        gyro_color = (200, 80, 255)
        for ch in self._detailed_channels:
            if ch.startswith("Acc"):
                pen = pg.mkPen(acc_color, width=1)
            else:
                pen = pg.mkPen(gyro_color, width=1)
            item = pg.PlotDataItem(pen=pen)
            item.setVisible(False)
            self.plot_widget.addItem(item)
            self._detailed_curves[ch] = item


    def _clear_detailed_items(self) -> None:
        """Hide all detailed curves (used when no data or in overview mode)."""
        if not self._detailed_curves:
            return
        for item in self._detailed_curves.values():
            item.setData([], [])
            item.setVisible(False)


    def _render_detailed(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """
        Render detailed 6-channel motion data in compact sub-rows.

        Falls back to overview mode if no detailed_data_provider is installed.
        """
        if self.detailed_data_provider is None or self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            # No provider available; keep existing overview behavior
            self._render_overview(time_range)
            return

        if time_range is None:
            # Without a time range, detailed view is not well-defined; just show overview
            self._render_overview(time_range)
            return

        self._ensure_detailed_items()

        # Hide overview bars while in detailed mode
        self.bar_graph_item.setVisible(False)

        start_dt, end_dt = time_range
        start_ts = self._safe_datetime_to_timestamp(start_dt)
        end_ts = self._safe_datetime_to_timestamp(end_dt)
        if start_ts is None or end_ts is None or not np.isfinite(start_ts) or not np.isfinite(end_ts) or end_ts <= start_ts:
            self._clear_detailed_items()
            return

        # Intervals that intersect the visible window
        mask = (self._all_intervals_ts[:, 0] <= end_ts) & (self._all_intervals_ts[:, 1] >= start_ts)
        visible_indices = np.where(mask)[0]
        if len(visible_indices) == 0:
            self._clear_detailed_items()
            return

        # Collect per-channel data across all visible intervals
        channel_times: Dict[str, List[np.ndarray]] = {ch: [] for ch in self._detailed_channels}
        channel_values: Dict[str, List[np.ndarray]] = {ch: [] for ch in self._detailed_channels}

        for idx in visible_indices:
            interval_start_ts = float(self._all_intervals_ts[idx, 0])
            interval_end_ts = float(self._all_intervals_ts[idx, 1])
            # Intersection of interval and visible window
            window_start = max(interval_start_ts, start_ts)
            window_end = min(interval_end_ts, end_ts)
            if not np.isfinite(window_start) or not np.isfinite(window_end) or window_end <= window_start:
                continue

            metadata = self._get_metadata_for_interval(idx)
            series_dict = self._load_motion_timeseries(metadata, (window_start, window_end))
            if not series_dict:
                continue

            for ch in self._detailed_channels:
                if ch in series_dict:
                    t_arr, v_arr = series_dict[ch]
                    if t_arr is None or v_arr is None or len(t_arr) == 0:
                        continue
                    # Ensure numpy arrays
                    t_arr = np.asarray(t_arr, dtype=float)
                    v_arr = np.asarray(v_arr, dtype=float)
                    # Basic safety filtering
                    valid = np.isfinite(t_arr) & np.isfinite(v_arr)
                    if not np.any(valid):
                        continue
                    channel_times[ch].append(t_arr[valid])
                    channel_values[ch].append(v_arr[valid])

        # Concatenate and normalize into sub-rows
        if not any(channel_times[ch] for ch in self._detailed_channels):
            self._clear_detailed_items()
            return

        n_channels = len(self._detailed_channels)
        gap = 0.02
        total_gap = gap * (n_channels + 1)
        band_height = max((1.0 - total_gap) / max(n_channels, 1), 0.0)

        for i, ch in enumerate(self._detailed_channels):
            curves = self._detailed_curves.get(ch)
            if curves is None:
                continue

            if not channel_times[ch]:
                curves.setData([], [])
                curves.setVisible(False)
                continue

            t_concat = np.concatenate(channel_times[ch])
            v_concat = np.concatenate(channel_values[ch])

            if t_concat.size == 0 or v_concat.size == 0:
                curves.setData([], [])
                curves.setVisible(False)
                continue

            v_min = np.nanmin(v_concat)
            v_max = np.nanmax(v_concat)
            if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
                # Degenerate range; render as flat line
                v_norm = np.zeros_like(v_concat)
            else:
                v_norm = (v_concat - v_min) / (v_max - v_min)

            band_bottom = gap * (i + 1) + band_height * i
            band_top = band_bottom + band_height
            y_vals = band_bottom + v_norm * (band_top - band_bottom)

            curves.setData(t_concat, y_vals)
            curves.setVisible(True)

        # Detailed mode uses [0, 1] y-range for stacked sub-rows
        self.plot_widget.setYRange(0, 1, padding=0.0)




