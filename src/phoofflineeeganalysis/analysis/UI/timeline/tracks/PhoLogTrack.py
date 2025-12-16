from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.StringDataTrack import StringDataTrack


class PhoLogTrack(StringDataTrack):
    """
    Track widget for displaying PHO_LOG_TO_LSL annotation intervals.
    
    Expects a DataFrame with columns:
    - onset: datetime (start time)
    - duration: float or Timedelta (duration in seconds)
    """
    
    def __init__(self, pho_log_df: pd.DataFrame, name: str = "PHO_LOG", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(df=pho_log_df, name=name, height=height, parent=parent, onset_col="onset", duration_col="duration")
        # Set PHO_LOG-specific colors (purple theme)
        self._pen_color = (200, 100, 255, 255)
        self._brush_color = (200, 100, 255, 150)

        # Rebuild pen/brush with new colors on next update_display call
        self._pen = None
        self._brush = None

        # Initial display update (show all) with new colors
        self.update_display()

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


