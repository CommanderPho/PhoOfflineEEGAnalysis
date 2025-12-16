from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QFont, QFontMetrics
import pyqtgraph as pg
from phoofflineeeganalysis.analysis.UI.timeline.tracks.StringDataTrack import StringDataTrack
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized


class PhoLogTrack(StringDataTrack):
    """
    Track widget for displaying PHO_LOG_TO_LSL annotation intervals.
    
    Supports two DataFrame formats:
    1. ['onset', 'duration'] - original format (backward compatible)
    2. ['time', 'text'] - new format with optional 'duration' column
    
    When duration is missing or 0, entries are rendered as point markers.
    Text labels are displayed in detailed mode (when zoomed in) with intelligent
    overlap prevention using vertical staggering.
    """
    
    def __init__(self, pho_log_df: pd.DataFrame, name: str = "PHO_LOG", height: int = 60, parent: Optional[QWidget] = None):
        # Detect data format and prepare DataFrame
        df = pho_log_df.copy()
        onset_col = None
        duration_col = None
        
        # Check for 'time' column (new format)
        if 'time' in df.columns:
            onset_col = 'time'
            # Rename 'time' to 'onset' for base class compatibility
            if 'onset' not in df.columns:
                df = df.rename(columns={'time': 'onset'})
                onset_col = 'onset'
            # Check for optional duration column
            if 'duration' in df.columns:
                duration_col = 'duration'
            else:
                # No duration column - set all durations to 0 for point markers
                df['duration'] = 0.0
                duration_col = 'duration'
        elif 'onset' in df.columns:
            # Original format - backward compatible
            onset_col = 'onset'
            if 'duration' in df.columns:
                duration_col = 'duration'
            else:
                # No duration - set to 0 for point markers
                df['duration'] = 0.0
                duration_col = 'duration'
        else:
            raise ValueError("DataFrame must have either 'time' or 'onset' column")
        
        # Ensure 'text' column exists if not present (for text rendering)
        if 'text' not in df.columns:
            # Try to use 'message' or 'label' as text source
            if 'message' in df.columns:
                df['text'] = df['message'].astype(str)
            elif 'label' in df.columns:
                df['text'] = df['label'].astype(str)
            else:
                df['text'] = ''
        
        super().__init__(df=df, name=name, height=height, parent=parent, onset_col=onset_col, duration_col=duration_col)
        
        # Set PHO_LOG-specific colors (purple theme)
        self._pen_color = (200, 100, 255, 255)
        self._brush_color = (200, 100, 255, 150)

        # Rebuild pen/brush with new colors on next update_display call
        self._pen = None
        self._brush = None
        
        # Text rendering infrastructure
        self._text_items: List[pg.TextItem] = []
        self._point_markers: Optional[pg.ScatterPlotItem] = None
        self._max_text_height: int = height
        
        # Enable detailed mode for text rendering (60 second threshold)
        self.detailed_mode_timespan_threshold_sec = 60.0
        
        # Text rendering configuration
        self._max_text_items = 100  # Limit for performance
        self._text_font = QFont()
        self._text_font.setPointSize(9)
        self._text_padding = 2  # Padding around text in pixels

        # Initial display update (show all) with new colors
        self.update_display()
    
    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Override to preserve 0 durations for point markers.
        Base class converts 0 durations to 0.1, but we need 0 for point markers.
        """
        if self._df.empty or self._onset_col not in self._df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []
        
        df = self._df.copy()
        start_dt = df[self._onset_col]
        
        if self._duration_col in df.columns:
            durations = parse_duration_to_seconds_vectorized(df[self._duration_col])
        else:
            durations = pd.Series(0.0, index=df.index)
        
        # Fill NaN with 0.1, but preserve explicit 0 values for point markers
        durations = durations.fillna(0.1)
        # Only convert negative durations, not zero
        durations[durations < 0] = 0.1
        
        end_dt = start_dt + pd.to_timedelta(durations, unit="s")
        
        mask = start_dt.notna() & end_dt.notna()
        self._display_df = df[mask].copy().reset_index(drop=True)
        self._display_df["final_end_dt"] = end_dt[mask].reset_index(drop=True)
        self._display_df["final_duration"] = durations[mask].reset_index(drop=True)
        
        if self._display_df.empty:
            return np.empty((0, 2)), []
        
        starts = self._display_df[self._onset_col].values.astype("datetime64[ns]").astype(np.float64) / 1e9
        # For 0 duration, set end = start (point marker)
        ends = self._display_df["final_end_dt"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
        # Ensure ends >= starts (for 0 duration, end = start)
        ends = np.maximum(ends, starts)
        
        return np.column_stack([starts, ends]), []

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Extend base StringDataTrack metadata with PHO_LOG-specific fields."""
        base_metadata = super()._get_metadata_for_interval(interval_index)
        if interval_index < 0 or self._display_df is None or interval_index >= len(self._display_df):
            return base_metadata

        row = self._display_df.iloc[interval_index]

        if "duration" in row and pd.notna(row["duration"]) and "duration_sec" not in base_metadata:
            base_metadata["duration_sec"] = row["duration"]

        if "message" in row and pd.notna(row["message"]):
            base_metadata["message"] = str(row["message"])
        if "label" in row and pd.notna(row["label"]):
            base_metadata["label"] = str(row["label"])

        return base_metadata
    
    def _prepare_text_for_display(self, text: str, available_width_px: float) -> str:
        """
        Prepare text for display by elliding if necessary.
        
        Args:
            text: Original text string
            available_width_px: Available width in pixels
            
        Returns:
            Processed text string (ellided if needed)
        """
        if not text or available_width_px <= 0:
            return ""
        
        # Use QFontMetrics to measure text width
        font_metrics = QFontMetrics(self._text_font)
        text_width = font_metrics.width(text)
        
        # If text fits, return as-is
        if text_width <= available_width_px:
            return text
        
        # Calculate how many characters fit
        # Approximate: use average character width
        avg_char_width = font_metrics.averageCharWidth()
        max_chars = int((available_width_px - font_metrics.width("...")) / avg_char_width)
        
        if max_chars <= 0:
            return "..."
        
        # Ellide text
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        
        return text
    
    def _layout_text_labels(self, visible_intervals: np.ndarray, visible_indices: np.ndarray) -> List[Tuple[float, float, str, float, int]]:
        """
        Layout text labels to prevent overlap by staggering vertically.
        
        Args:
            visible_intervals: Array of [N, 2] with (start_ts, end_ts) for visible intervals
            visible_indices: Array of indices into self._display_df for visible intervals
            
        Returns:
            List of (x_center, y_pos, text, width_ts, row_index) tuples
        """
        if len(visible_intervals) == 0:
            return []
        
        # Get text and interval data
        layout_data = []
        for i, idx in enumerate(visible_indices):
            if idx < 0 or idx >= len(self._display_df):
                continue
            
            row = self._display_df.iloc[idx]
            start_ts, end_ts = visible_intervals[i]
            width_ts = end_ts - start_ts
            x_center = start_ts + (width_ts / 2.0)
            
            # Get text
            text = ""
            if "text" in row and pd.notna(row["text"]):
                text = str(row["text"])
            elif "message" in row and pd.notna(row["message"]):
                text = str(row["message"])
            elif "label" in row and pd.notna(row["label"]):
                text = str(row["label"])
            
            if text:
                layout_data.append((x_center, start_ts, end_ts, text, width_ts))
        
        if not layout_data:
            return []
        
        # Sort by start time
        layout_data.sort(key=lambda x: x[1])
        
        # Group into rows based on overlap
        rows: List[List[Tuple[float, float, str, float]]] = []
        
        for x_center, start_ts, end_ts, text, width_ts in layout_data:
            # Find first row where this interval doesn't overlap
            placed = False
            for row_items in rows:
                # Check if overlaps with any item in this row
                overlaps = False
                for existing_x, existing_start, existing_end, _, _ in row_items:
                    # Check if intervals overlap
                    if not (end_ts < existing_start or start_ts > existing_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    row_items.append((x_center, start_ts, end_ts, text, width_ts))
                    placed = True
                    break
            
            if not placed:
                # Create new row
                rows.append([(x_center, start_ts, end_ts, text, width_ts)])
        
        # Distribute rows evenly across available height
        n_rows = len(rows)
        if n_rows == 0:
            return []
        
        # Use most of the track height, leave some margin
        y_min = 0.1
        y_max = 0.9
        y_range = y_max - y_min
        
        result = []
        for row_idx, row_items in enumerate(rows):
            # Calculate y position for this row
            if n_rows == 1:
                y_pos = (y_min + y_max) / 2.0
            else:
                y_pos = y_min + (row_idx / (n_rows - 1)) * y_range
            
            # Add all items in this row
            for x_center, start_ts, end_ts, text, width_ts in row_items:
                result.append((x_center, y_pos, text, width_ts, row_idx))
        
        return result
    
    def _render_detailed(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """
        Render detailed view with text labels and point markers.
        """
        # First render bars using base class overview rendering
        self._render_overview(time_range)
        
        # Clear existing text items
        for item in self._text_items:
            self.plot_widget.removeItem(item)
        self._text_items.clear()
        
        # Remove existing point markers
        if self._point_markers is not None:
            self.plot_widget.removeItem(self._point_markers)
            self._point_markers = None
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            return
        
        # Get visible intervals
        visible_intervals = self._all_intervals_ts
        visible_indices = np.arange(len(self._all_intervals_ts))
        
        if time_range is not None:
            start_dt, end_dt = time_range
            start_ts = self._safe_datetime_to_timestamp(start_dt)
            end_ts = self._safe_datetime_to_timestamp(end_dt)
            
            if start_ts is not None and end_ts is not None:
                mask = (self._all_intervals_ts[:, 0] <= end_ts) & (self._all_intervals_ts[:, 1] >= start_ts)
                visible_intervals = self._all_intervals_ts[mask]
                visible_indices = np.where(mask)[0]
        
        if len(visible_intervals) == 0:
            return
        
        # Limit number of items for performance
        if len(visible_intervals) > self._max_text_items:
            # Take first N items
            visible_intervals = visible_intervals[:self._max_text_items]
            visible_indices = visible_indices[:self._max_text_items]
        
        # Separate into bars (duration > 0) and points (duration = 0)
        bar_indices = []
        point_indices = []
        point_data = []
        
        for i, idx in enumerate(visible_indices):
            if idx < 0 or idx >= len(self._display_df):
                continue
            
            row = self._display_df.iloc[idx]
            duration = row.get("final_duration", 0.0) if "final_duration" in row else 0.0
            
            if duration > 0:
                bar_indices.append(i)
            else:
                point_indices.append(i)
                start_ts, end_ts = visible_intervals[i]
                point_data.append((start_ts, 0.5))  # y=0.5 is middle of track
        
        # Render point markers
        if point_data:
            x_points = [p[0] for p in point_data]
            y_points = [p[1] for p in point_data]
            
            self._point_markers = pg.ScatterPlotItem(
                x=x_points,
                y=y_points,
                pen=pg.mkPen(self._pen_color),
                brush=pg.mkBrush(self._pen_color),
                size=6,
                symbol='o'
            )
            self.plot_widget.addItem(self._point_markers)
        
        # Layout and render text labels
        text_layout = self._layout_text_labels(visible_intervals, visible_indices)
        
        if not text_layout:
            return  # No text to render
        
        # Get viewbox to calculate pixel width
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return
        
        view_range = vb.viewRange()[0]
        if len(view_range) != 2:
            return
        
        start_ts_view, end_ts_view = view_range
        view_width_ts = end_ts_view - start_ts_view
        view_width_px = vb.width()
        
        if view_width_px <= 0 or view_width_ts <= 0:
            return
        
        pixels_per_second = view_width_px / view_width_ts
        
        # Render text items (limit to max_text_items for performance)
        text_items_to_render = text_layout[:self._max_text_items]
        for x_center, y_pos, text, width_ts, row_idx in text_items_to_render:
            # Calculate available width in pixels
            available_width_px = width_ts * pixels_per_second - (2 * self._text_padding)
            
            # Prepare text
            display_text = self._prepare_text_for_display(text, available_width_px)
            
            if not display_text:
                continue
            
            # Create text item
            text_item = pg.TextItem(
                text=display_text,
                color=(255, 255, 255, 255),  # White text
                border=pg.mkPen((0, 0, 0, 200)),  # Black border for visibility
                fill=pg.mkBrush((200, 100, 255, 180))  # Purple background matching track color
            )
            text_item.setFont(self._text_font)
            text_item.setPos(x_center, y_pos)
            text_item.setAnchor((0.5, 0.5))  # Center anchor
            
            self.plot_widget.addItem(text_item)
            self._text_items.append(text_item)


