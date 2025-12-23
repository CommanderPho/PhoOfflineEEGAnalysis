from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QLabel, QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem
from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import BaseDatasource


class DetailedRenderingTrackMixin:
    """ Tracks that displayed detailed renderings must override these methods
    """
    # ---- Detailed rendering -------------------------------------------------

    def set_detailed_threshold(self, seconds: Optional[float]) -> None:
        """Set the time-span threshold (in seconds) for switching to detailed rendering."""
        self.detailed_mode_timespan_threshold_sec = seconds

    def _ensure_detailed_items(self) -> None:
        """Create PlotDataItem per channel for detailed mode if not already present."""
        raise NotImplementedError("Implementing class must override")


    def _clear_detailed_items(self) -> None:
        """Hide all detailed curves (used when no data or in overview mode)."""
        raise NotImplementedError("Implementing class must override")

    def _render_detailed(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """
        Default detailed rendering falls back to overview.

        Subclasses can override this to draw data-rich views while reusing
        the same time_range semantics.
        """
        self._render_overview(time_range)





class TrackWidget(DetailedRenderingTrackMixin, QWidget):
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
        # React to zoom/pan changes to update overview/detailed rendering
        vb.sigXRangeChanged.connect(self._on_view_range_changed)
        
        # Cache all intervals for performance
        self._all_intervals_ts: Optional[np.ndarray] = None  # Cached as [N, 2] array of (start_ts, end_ts)
        
        # Store metadata for each interval (index matches _all_intervals_ts)
        self._interval_metadata: List[Dict[str, Any]] = []
        
        # Single item for rendering all bars (overview mode)
        self.bar_graph_item = pg.BarGraphItem(x=[], height=[], width=[], brush='b')
        self.plot_widget.addItem(self.bar_graph_item)
        
        # Default colors (can be overridden by subclasses)
        self._pen_color = (100, 150, 200, 255)
        self._brush_color = (100, 150, 200, 150)
        
        # Cache pen and brush objects
        self._pen = None
        self._brush = None

        # Overview vs detailed rendering configuration
        # If None, track always renders in overview mode.
        self.detailed_mode_timespan_threshold_sec: Optional[float] = None
        self._is_detailed_mode: bool = False
        self._last_visible_range: Optional[Tuple[float, float]] = None
        
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

        # Optional datasource backing this track (for flexible data access)
        self._datasource: Optional[BaseDatasource] = None
        
    def set_datasource(self, datasource: BaseDatasource) -> None:
        """Attach a datasource to this track and react to its change signal."""
        # Disconnect previous datasource if any
        if self._datasource is not None and hasattr(self._datasource, 'source_data_changed_signal'):
            try:
                self._datasource.source_data_changed_signal.disconnect(self._on_datasource_changed)
            except TypeError:
                # Was not connected
                pass

        self._datasource = datasource
        if self._datasource is not None and hasattr(self._datasource, 'source_data_changed_signal'):
            self._datasource.source_data_changed_signal.connect(self._on_datasource_changed)

        # Rebuild cached intervals based on new datasource
        self._cache_intervals()
        self.update_display()

    def get_datasource(self) -> Optional[BaseDatasource]:
        """Return the currently attached datasource, if any."""
        return self._datasource

    def _on_datasource_changed(self, _changed_source: object) -> None:
        """Slot called when the underlying datasource reports data changes."""
        self._cache_intervals()
        self.update_display()

    def _get_full_dataframe(self) -> Optional[pd.DataFrame]:
        """Best-effort helper to obtain the full DataFrame from the datasource.

        Uses the generic BaseDatasource interface first (total_datasource_start_end_times
        and get_updated_data_window). Falls back to a .df attribute if present.
        """
        if self._datasource is None:
            return None

        # Prefer the generic API when available
        try:
            if hasattr(self._datasource, 'total_datasource_start_end_times') and hasattr(self._datasource, 'get_updated_data_window'):
                total_range = self._datasource.total_datasource_start_end_times
                if isinstance(total_range, (tuple, list)) and len(total_range) == 2:
                    start, end = total_range
                    return self._datasource.get_updated_data_window(start, end)
        except Exception:
            # Fall back to accessing a .df attribute if something goes wrong
            pass

        # Fallback: direct .df attribute if it exists
        df = getattr(self._datasource, 'df', None)
        if isinstance(df, pd.DataFrame):
            return df
        return None
        
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
            start_ts = self._safe_datetime_to_timestamp(s) if isinstance(s, datetime) else float(s)
            end_ts = self._safe_datetime_to_timestamp(e) if isinstance(e, datetime) else float(e)
            # Use NaN for invalid timestamps (will be filtered out later)
            starts[i] = start_ts if start_ts is not None else np.nan
            ends[i] = end_ts if end_ts is not None else np.nan
            
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
    
    def _safe_timestamp_to_datetime(self, ts: float) -> Optional[datetime]:
        """Safely convert timestamp to datetime, handling Windows OSError."""
        try:
            return datetime.fromtimestamp(ts)
        except (OSError, ValueError, OverflowError):
            # Windows can raise OSError for out-of-range timestamps
            # Return None to indicate invalid timestamp
            return None
    
    def _safe_datetime_to_timestamp(self, dt: datetime) -> Optional[float]:
        """Safely convert datetime to timestamp, handling Windows OSError."""
        try:
            if isinstance(dt, datetime):
                return dt.timestamp()
            else:
                return float(dt)
        except (OSError, ValueError, OverflowError):
            # Windows can raise OSError for out-of-range datetimes
            return None

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

    def _on_view_range_changed(self, view_box, x_range):
        """Callback for ViewBox range changes; triggers overview/detailed updates."""
        if x_range is None or len(x_range) != 2:
            return
        self._last_visible_range = (float(x_range[0]), float(x_range[1]))
        # Convert to datetime range assuming x-axis is UNIX timestamp seconds
        start_ts, end_ts = self._last_visible_range
        if not np.isfinite(start_ts) or not np.isfinite(end_ts) or end_ts <= start_ts:
            return
        start_dt = self._safe_timestamp_to_datetime(start_ts)
        end_dt = self._safe_timestamp_to_datetime(end_ts)
        if start_dt is None or end_dt is None:
            return
        self.update_display((start_dt, end_dt))

    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        """Dispatch to overview or detailed rendering based on visible time-span."""
        if self._all_intervals_ts is None:
            self._cache_intervals()
            
        # Helper to setup colors if needed
        if self._pen is None:
            self._pen = pg.mkPen(self._pen_color)
            self._brush = pg.mkBrush(self._brush_color)  # pg.mkBrush handles (r,g,b,a) tuple
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            return

        # Determine effective visible range
        effective_range = time_range
        if effective_range is None:
            vb = self.plot_widget.getViewBox()
            if vb is not None:
                x_range = vb.viewRange()[0]
                if len(x_range) == 2:
                    start_ts, end_ts = float(x_range[0]), float(x_range[1])
                    if np.isfinite(start_ts) and np.isfinite(end_ts) and end_ts > start_ts:
                        start_dt = self._safe_timestamp_to_datetime(start_ts)
                        end_dt = self._safe_timestamp_to_datetime(end_ts)
                        if start_dt is not None and end_dt is not None:
                            effective_range = (start_dt, end_dt)

        # Decide mode based on visible time-span
        use_detailed = False
        if effective_range is not None and self.detailed_mode_timespan_threshold_sec is not None:
            start_dt, end_dt = effective_range
            span_sec = (end_dt - start_dt).total_seconds()
            if span_sec <= self.detailed_mode_timespan_threshold_sec:
                use_detailed = True

        self._is_detailed_mode = use_detailed

        if use_detailed:
            self._render_detailed(effective_range)
        else:
            self._render_overview(effective_range)

    def _render_overview(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """Default overview rendering: bar-graph intervals."""
        visible_intervals = self._all_intervals_ts

        if time_range is not None:
            start_dt, end_dt = time_range
            start_ts = self._safe_datetime_to_timestamp(start_dt)
            end_ts = self._safe_datetime_to_timestamp(end_dt)
            
            # If timestamp conversion failed, show all intervals
            if start_ts is None or end_ts is None:
                visible_intervals = self._all_intervals_ts
            else:
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
            
            self.bar_graph_item.setOpts(
                x=centers,
                height=np.ones_like(centers),
                width=widths,
                brush=self._brush,
                pen=self._pen
            )
            self.bar_graph_item.setVisible(True)
        else:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            self.bar_graph_item.setVisible(True)
            
        self.plot_widget.setYRange(0, 1, padding=0.0)


    # ==================================================================================================================================================================================================================================================================================== #
    # DetailedRenderingTrackMixin Implementation                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def set_detailed_threshold(self, seconds: Optional[float]) -> None:
        """Set the time-span threshold (in seconds) for switching to detailed rendering."""
        self.detailed_mode_timespan_threshold_sec = seconds

    def _clear_detailed_items(self) -> None:
        """Hide all detailed curves (used when no data or in overview mode)."""
        raise NotImplementedError("Implementing class must override")

    def _render_detailed(self, time_range: Optional[Tuple[datetime, datetime]]) -> None:
        """
        Default detailed rendering falls back to overview.

        Subclasses can override this to draw data-rich views while reusing
        the same time_range semantics.
        """
        self._render_overview(time_range)




    def get_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        if self._all_intervals_ts is None:
            self._cache_intervals()
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            return None
        
        start_ts = np.min(self._all_intervals_ts[:, 0])
        end_ts = np.max(self._all_intervals_ts[:, 1])
        
        start_dt = self._safe_timestamp_to_datetime(start_ts)
        end_dt = self._safe_timestamp_to_datetime(end_ts)
        
        if start_dt is None or end_dt is None:
            return None
        
        return (start_dt, end_dt)
        
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
        
        start_dt = self._safe_timestamp_to_datetime(start_ts)
        end_dt = self._safe_timestamp_to_datetime(end_ts)
        if start_dt is not None and end_dt is not None:
            lines.append(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Duration: {end_dt - start_dt}")
        else:
            lines.append(f"Start: {start_ts}")
            lines.append(f"End: {end_ts}")
        
        for k in ['duration_sec', 'fps', 'resolution']:
            if k in metadata and metadata[k]:
                label = k.replace('_', ' ').title()
                lines.append(f"{label}: {metadata[k]}")

        return '\n'.join(lines)
    
    def _show_metadata_dialog(self, metadata: Dict[str, Any], start_ts: float, end_ts: float):
        start_dt = self._safe_timestamp_to_datetime(start_ts)
        end_dt = self._safe_timestamp_to_datetime(end_ts)
        
        lines = [f"<b>{self.name} Recording Details</b>", ""]
        lines.append(f"<b>Time Range:</b>")
        if start_dt is not None and end_dt is not None:
            lines.append(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            lines.append(f"  End: {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            lines.append(f"  Duration: {end_dt - start_dt}")
        else:
            lines.append(f"  Start: {start_ts}")
            lines.append(f"  End: {end_ts}")
        lines.append("")
        
        if metadata:
            lines.append("<b>Metadata:</b>")
            for key, value in sorted(metadata.items()):
                if value is not None and value != '':
                    display_key = key.replace('_', ' ').title()
                    lines.append(f"  {display_key}: {value}")
        
        QMessageBox.information(self, f"{self.name} Recording Details", '\n'.join(lines))

