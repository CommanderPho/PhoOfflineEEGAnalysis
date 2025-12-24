from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from copy import deepcopy
from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg
from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized
from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import BaseDatasource, IntervalDataframeDatasource, DataframeDatasource


class MotionRecordingTrack(TrackWidget):
    """
    Track widget for displaying motion recording intervals from SessionModality.

    Overview mode renders interval bars (summary view).

    Accepts either a BaseDatasource (preferred) or a DataFrame (backwards compatible).
    If a DataFrame is provided, it is wrapped in an IntervalDataframeDatasource.

    The datasource's DataFrame should have columns:
    - recording_datetime: datetime (start time)
    - duration_sec: Timedelta or float (duration in seconds)
    """

    def __init__(self, motion_source, name: str = "Motion", height: int = 60, parent: Optional[QWidget] = None, position_datasource: Optional[DataframeDatasource] = None):
        super().__init__(name=name, height=height, parent=parent)
        # Set motion-specific colors (orange/red theme)
        self._pen_color = (255, 150, 50, 255)
        self._brush_color = (255, 150, 50, 150)
        # Set background color to dark grey
        self.plot_widget.setBackground('darkgrey')


        if isinstance(motion_source, pd.DataFrame):
            # Wrap DataFrame in IntervalDataframeDatasource for backwards compatibility
            motion_df = deepcopy(motion_source)
            motion_source = IntervalDataframeDatasource(motion_df, time_column_name='recording_datetime', datasource_name=name)
            

        # Normalize input into a datasource (backwards compatible: wrap DataFrame if needed)
        assert isinstance(motion_source, (BaseDatasource, DataframeDatasource, IntervalDataframeDatasource)), f"failed to get correct type!: type(motion_source): {type(motion_source)}"
        # Already a datasource, use it directly
        self.set_datasource(motion_source)

        # self.set_datasource(interval_ds)

        # Store position datasource for detailed rendering
        self._position_datasource = position_datasource
        
        # Position column names (AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
        self._position_columns = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
        
        # Dictionary to store PlotDataItems for detailed rendering
        self._detailed_plot_items: Dict[str, pg.PlotDataItem] = {}
        
        # Ensure detailed items are created (but initially hidden)
        self._ensure_detailed_items()

        # Cached display DataFrame used for metadata lookups (populated by _get_recording_intervals_vectorized)
        self._display_df = pd.DataFrame()
        
        # Cache intervals immediately
        self._cache_intervals()
        
        # Initial display update (show all)
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract motion recording intervals from datasource-backed DataFrame."""
        # Always get DataFrame from datasource
        df_full = self._get_full_dataframe()
        if not isinstance(df_full, pd.DataFrame) or df_full.empty:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []

        if 'recording_datetime' not in df_full.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []

        df = df_full.copy()

        # Ensure datetime columns are datetime type and normalized to UTC-naive
        if 'recording_datetime' in df.columns:
            df['recording_datetime'] = self._ensure_utc_naive(df['recording_datetime'])
        
        # Calculate start times
        start_dt = df['recording_datetime']
        
        # Initialize end_dt
        end_dt = pd.Series(pd.NaT, index=df.index)
        
        # Calculate end times from duration_sec
        if 'duration_sec' in df.columns:
            durations = parse_duration_to_seconds_vectorized(df['duration_sec'])
            valid_mask = durations.notna()
            if valid_mask.any():
                end_dt[valid_mask] = start_dt[valid_mask] + pd.to_timedelta(durations[valid_mask], unit='s')
        
        # Filter valid rows
        mask = start_dt.notna() & end_dt.notna() & (end_dt > start_dt)
        
        # Save filtered df with computed ends (similar to XDFStreamTrack pattern)
        df['final_start_dt'] = start_dt
        df['final_end_dt'] = end_dt
        self._display_df = df[mask].copy().reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
        
        starts = self._display_df['final_start_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        ends = self._display_df['final_end_dt'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Lazy load metadata from datasource-backed display DataFrame."""
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

        # Keep a reference to the original row index if present
        if 'index' in self._display_df.columns:
            metadata['row_index'] = self._display_df.index[interval_index]

        return metadata

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        # Handled by _get_recording_intervals_vectorized
        return []

    def _cache_metadata(self):
        # Vectorized path stores metadata lazily in _get_metadata_for_interval
        pass

    def _render_overview(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """Render overview mode with interval bars, clearing detailed items."""
        # Clear detailed items when in overview mode
        self._clear_detailed_items()
        
        # Call parent implementation
        super()._render_overview(time_range)

    def _render_detailed(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """
        Render detailed motion data: interval rectangles + line plots overlay.
        
        Both interval rectangles and line plots should be visible simultaneously.
        """
        # First render interval rectangles (core visualization)
        super()._render_detailed(time_range)  # This renders rectangles and clears detailed items
        
        # Ensure detailed items exist
        self._ensure_detailed_items()
        
        # If no position datasource, fall back to overview
        if self._position_datasource is None:
            self._clear_detailed_items()
            self._render_overview(time_range)
            return
        
        # If no time range provided, clear and return
        if time_range is None:
            self._clear_detailed_items()
            return
        
        start_dt, end_dt = time_range
        start_ts = self._safe_datetime_to_timestamp(start_dt)
        end_ts = self._safe_datetime_to_timestamp(end_dt)
        
        if start_ts is None or end_ts is None or not np.isfinite(start_ts) or not np.isfinite(end_ts) or end_ts <= start_ts:
            self._clear_detailed_items()
            return
        
        # Find visible intervals that overlap with time_range
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            self._clear_detailed_items()
            return
        
        # Find intervals that overlap with the time range
        mask = (self._all_intervals_ts[:, 0] <= end_ts) & (self._all_intervals_ts[:, 1] >= start_ts)
        visible_intervals_ts = self._all_intervals_ts[mask]
        
        if len(visible_intervals_ts) == 0:
            self._clear_detailed_items()
            return
        
        # Query position data for the visible time range
        try:
            # Query position datasource for the time range
            position_df = self._position_datasource.get_updated_data_window(start_ts, end_ts)
            
            if not isinstance(position_df, pd.DataFrame) or position_df.empty:
                self._clear_detailed_items()
                return
            
            # Get time column name from datasource (defaults to 't', but can be 'time')
            time_col = getattr(self._position_datasource, 'time_column_name', 'time')
            
            # Ensure time column exists
            if time_col not in position_df.columns:
                self._clear_detailed_items()
                return
            
            # Filter to the actual time range (in case datasource returned more)
            position_df = position_df[position_df[time_col].between(start_ts, end_ts)].copy()
            
            if position_df.empty:
                self._clear_detailed_items()
                return
            
            # Extract time array (already in timestamp format)
            times = position_df[time_col].values.astype(np.float64)
            
            # For each position column, update the corresponding PlotDataItem
            all_values = []
            for col_name in self._position_columns:
                if col_name not in position_df.columns:
                    # Column missing, hide this plot item
                    self._detailed_plot_items[col_name].setData(x=np.array([]), y=np.array([]))
                    self._detailed_plot_items[col_name].setVisible(False)
                    continue
                
                # Extract values for this column
                values = position_df[col_name].values.astype(np.float64)
                
                # Remove NaN values
                valid_mask = np.isfinite(values) & np.isfinite(times)
                if not valid_mask.any():
                    # No valid data, hide this plot item
                    self._detailed_plot_items[col_name].setData(x=np.array([]), y=np.array([]))
                    self._detailed_plot_items[col_name].setVisible(False)
                    continue
                
                valid_times = times[valid_mask]
                valid_values = values[valid_mask]
                
                # Store for y-axis range calculation
                all_values.append(valid_values)
                
                # Update plot item
                self._detailed_plot_items[col_name].setData(x=valid_times, y=valid_values)
                self._detailed_plot_items[col_name].setVisible(True)
            
            # Set y-axis range
            if all_values:
                # Combine all values to determine overall range
                all_vals_combined = np.concatenate(all_values)
                if len(all_vals_combined) > 0 and np.isfinite(all_vals_combined).any():
                    y_min = np.nanmin(all_vals_combined)
                    y_max = np.nanmax(all_vals_combined)
                    y_range = y_max - y_min
                    
                    if y_range > 0:
                        # Add padding
                        padding = y_range * 0.1
                        self.plot_widget.setYRange(y_min - padding, y_max + padding, padding=0.0)
                    else:
                        # All values are the same, set a small range around it
                        self.plot_widget.setYRange(y_min - 1, y_max + 1, padding=0.0)
                else:
                    # No valid values, use default range
                    self.plot_widget.setYRange(0, 1, padding=0.0)
            else:
                # No valid data, use default range
                self.plot_widget.setYRange(0, 1, padding=0.0)
                
        except Exception as e:
            # On any error, clear detailed items and fall back to overview
            self._clear_detailed_items()
            self._render_overview(time_range)




    def _ensure_detailed_items(self) -> None:
        """Create PlotDataItem per channel for detailed mode if not already present."""
        if self._detailed_plot_items:
            # Already created, skip
            return
        
        # Color scheme: AccX/Y/Z in red/orange shades, GyroX/Y/Z in blue/cyan shades
        colors = {
            'AccX': (255, 100, 50, 255),   # Red-orange
            'AccY': (255, 150, 50, 255),   # Orange
            'AccZ': (255, 200, 100, 255),  # Light orange
            'GyroX': (50, 150, 255, 255),  # Blue
            'GyroY': (100, 200, 255, 255), # Light blue
            'GyroZ': (150, 220, 255, 255), # Cyan
        }
        
        for col_name in self._position_columns:
            color = colors.get(col_name, (255, 255, 255, 255))
            pen = pg.mkPen(color, width=1)
            
            plot_item = pg.PlotDataItem(
                x=np.array([]),
                y=np.array([]),
                pen=pen,
                name=col_name
            )
            
            # Initially hide
            plot_item.setVisible(False)
            
            # Add to plot widget
            self.plot_widget.addItem(plot_item)
            
            # Store in dictionary
            self._detailed_plot_items[col_name] = plot_item


    def _clear_detailed_items(self) -> None:
        """Hide all detailed curves (used when no data or in overview mode)."""
        for plot_item in self._detailed_plot_items.values():
            plot_item.setData(x=np.array([]), y=np.array([]))
            plot_item.setVisible(False)

        