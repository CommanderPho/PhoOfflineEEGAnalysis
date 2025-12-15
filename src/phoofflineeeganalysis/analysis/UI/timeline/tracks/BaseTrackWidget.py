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


class TrackWidget(QWidget):
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
        
        # Cache all intervals for performance
        self._all_intervals_ts: Optional[np.ndarray] = None  # Cached as [N, 2] array of (start_ts, end_ts)
        
        # Store metadata for each interval (index matches _all_intervals_ts)
        self._interval_metadata: List[Dict[str, Any]] = []
        
        # Single item for rendering all bars
        self.bar_graph_item = pg.BarGraphItem(x=[], height=[], width=[], brush='b')
        self.plot_widget.addItem(self.bar_graph_item)
        
        # Default colors (can be overridden by subclasses)
        self._pen_color = (100, 150, 200, 255)
        self._brush_color = (100, 150, 200, 150)
        
        # Cache pen and brush objects
        self._pen = None
        self._brush = None
        
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
            starts[i] = s.timestamp() if isinstance(s, datetime) else float(s)
            ends[i] = e.timestamp() if isinstance(e, datetime) else float(e)
            
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

    def update_display(self, time_range: Optional[Tuple[datetime, datetime]] = None):
        if self._all_intervals_ts is None:
            self._cache_intervals()
            
        # Helper to setup colors if needed
        if self._pen is None:
            self._pen = pg.mkPen(self._pen_color)
            self._brush = pg.mkBrush(self._brush_color) # pg.mkBrush handles (r,g,b,a) tuple
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            return
            
        visible_intervals = self._all_intervals_ts
        
        if time_range is not None:
            start_dt, end_dt = time_range
            start_ts = start_dt.timestamp() if isinstance(start_dt, datetime) else float(start_dt)
            end_ts = end_dt.timestamp() if isinstance(end_dt, datetime) else float(end_dt)
            
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
        else:
            self.bar_graph_item.setOpts(x=[], height=[], width=[])
            
        self.plot_widget.setYRange(0, 1, padding=0.0)

    def get_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        if self._all_intervals_ts is None:
            self._cache_intervals()
        
        if self._all_intervals_ts is None or len(self._all_intervals_ts) == 0:
            return None
        
        start_ts = np.min(self._all_intervals_ts[:, 0])
        end_ts = np.max(self._all_intervals_ts[:, 1])
        
        return (datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts))
        
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
        
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
        lines.append(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {end_dt - start_dt}")
        
        for k in ['duration_sec', 'fps', 'resolution']:
            if k in metadata and metadata[k]:
                label = k.replace('_', ' ').title()
                lines.append(f"{label}: {metadata[k]}")

        return '\n'.join(lines)
    
    def _show_metadata_dialog(self, metadata: Dict[str, Any], start_ts: float, end_ts: float):
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
        
        lines = [f"<b>{self.name} Recording Details</b>", ""]
        lines.append(f"<b>Time Range:</b>")
        lines.append(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        lines.append(f"  End: {end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        lines.append(f"  Duration: {end_dt - start_dt}")
        lines.append("")
        
        if metadata:
            lines.append("<b>Metadata:</b>")
            for key, value in sorted(metadata.items()):
                if value is not None and value != '':
                    display_key = key.replace('_', ' ').title()
                    lines.append(f"  {display_key}: {value}")
        
        QMessageBox.information(self, f"{self.name} Recording Details", '\n'.join(lines))

