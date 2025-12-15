from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import shutil
import platform
import os
import time
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import QTimer, Qt
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
        
        # Double-click detection: track last click time and position
        self._last_click_time = 0.0
        self._last_click_pos = None
        self._double_click_timer = QTimer(self)
        self._double_click_timer.setSingleShot(True)
        self._double_click_timer.timeout.connect(self._on_single_click_timeout)
        self._pending_click_data = None
        
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
    
    def _on_mouse_clicked(self, event):
        """Override to handle double-clicks for video launching."""
        # Check for left button click (Qt.LeftButton = 1)
        if event.button() == 1:  # Qt.LeftButton
            vb = self.plot_widget.getViewBox()
            scene_pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(scene_pos):
                mouse_point = vb.mapSceneToView(scene_pos)
                x_ts = mouse_point.x()
                y_val = mouse_point.y()
                
                if 0 <= y_val <= 1:
                    idx = self._find_interval_at_pos(x_ts)
                    if idx != -1:
                        # Check for double-click (within 300ms and similar position)
                        current_time = time.time()
                        is_double_click = (
                            current_time - self._last_click_time < 0.3 and
                            self._last_click_pos is not None and
                            abs(self._last_click_pos - x_ts) < 1.0  # Within 1 second on timeline
                        )
                        
                        if is_double_click:
                            # Cancel single-click timer and launch video
                            self._double_click_timer.stop()
                            self._handle_double_click(idx, x_ts)
                            event.accept()
                        else:
                            # Store click data and start timer for single-click
                            self._last_click_time = current_time
                            self._last_click_pos = x_ts
                            self._pending_click_data = (idx, x_ts)
                            self._double_click_timer.start(300)  # 300ms delay
                            event.accept()
    
    def _on_single_click_timeout(self):
        """Handle single-click after double-click timeout."""
        if self._pending_click_data is not None and self._all_intervals_ts is not None:
            idx, x_ts = self._pending_click_data
            self._pending_click_data = None
            # Call parent's single-click handler for metadata dialog
            metadata = self._get_metadata_for_interval(idx)
            start_ts = self._all_intervals_ts[idx, 0]
            end_ts = self._all_intervals_ts[idx, 1]
            self._show_metadata_dialog(metadata, start_ts, end_ts)
    
    def _handle_double_click(self, interval_index: int, click_timestamp: float):
        """Handle double-click on video interval to launch video player."""
        if interval_index < 0 or interval_index >= len(self._display_df) or self._all_intervals_ts is None:
            return
        
        # Get video metadata
        metadata = self._get_metadata_for_interval(interval_index)
        video_path_str = metadata.get('file_path', '')
        
        if not video_path_str:
            QMessageBox.warning(self, "Video Launch Error", "No video file path found for this interval.")
            return
        
        video_path = Path(video_path_str)
        
        # Validate video file exists
        if not video_path.exists():
            QMessageBox.warning(self, "Video Launch Error", f"Video file not found:\n{video_path}")
            return
        
        # Calculate offset from video start to click position
        start_ts = self._all_intervals_ts[interval_index, 0]
        end_ts = self._all_intervals_ts[interval_index, 1]
        offset_seconds = click_timestamp - start_ts
        
        # Ensure offset is non-negative and within video duration
        if offset_seconds < 0:
            offset_seconds = 0.0
        
        video_duration = end_ts - start_ts
        if offset_seconds > video_duration:
            offset_seconds = video_duration
        
        # Launch video player
        self._launch_video_player(video_path, offset_seconds)
    
    def _find_vlc_executable(self) -> Optional[Path]:
        """Find VLC executable path."""
        # Try to find VLC in PATH first
        vlc_path = shutil.which('vlc')
        if vlc_path:
            return Path(vlc_path)
        
        # Try common installation paths
        system = platform.system()
        if system == "Windows":
            common_paths = [
                Path("C:/Program Files/VideoLAN/VLC/vlc.exe"),
                Path("C:/Program Files (x86)/VideoLAN/VLC/vlc.exe"),
                Path(os.path.expanduser("~/AppData/Local/Programs/VLC/vlc.exe")),
            ]
        elif system == "Darwin":  # macOS
            common_paths = [
                Path("/Applications/VLC.app/Contents/MacOS/VLC"),
                Path("/usr/local/bin/vlc"),
            ]
        else:  # Linux
            common_paths = [
                Path("/usr/bin/vlc"),
                Path("/usr/local/bin/vlc"),
            ]
        
        for path in common_paths:
            if path.exists():
                return path
        
        return None
    
    def _launch_video_player(self, video_path: Path, start_offset_seconds: float):
        """Launch VLC video player with video file starting at specified offset."""
        vlc_exe = self._find_vlc_executable()
        
        if vlc_exe is None:
            QMessageBox.warning(
                self,
                "VLC Not Found",
                "VLC media player was not found on your system.\n\n"
                "Please install VLC from https://www.videolan.org/\n"
                "or ensure it is in your system PATH."
            )
            return
        
        try:
            # Build VLC command with start time
            cmd = [
                str(vlc_exe),
                "--start-time", str(int(start_offset_seconds)),
                str(video_path)
            ]
            
            # Launch VLC in background (detached process)
            if platform.system() == "Windows":
                # On Windows, use CREATE_NO_WINDOW to avoid console window
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # On Unix-like systems, detach from parent process
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Video Launch Error",
                f"Failed to launch VLC:\n{str(e)}"
            )
