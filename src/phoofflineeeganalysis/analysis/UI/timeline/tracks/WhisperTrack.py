from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget
from phoofflineeeganalysis.analysis.UI.timeline.tracks.StringDataTrack import StringDataTrack


class WhisperTrack(StringDataTrack):
    """
    Track widget for displaying Whisper transcript intervals.
    
    Expects a DataFrame with columns:
    - onset: datetime (start time)
    - duration: float or Timedelta (duration in seconds)
    """
    
    def __init__(self, whisper_df: pd.DataFrame, name: str = "Whisper", height: int = 60, parent: Optional[QWidget] = None):
        super().__init__(df=whisper_df, name=name, height=height, parent=parent, onset_col="onset", duration_col="duration")
        # Set Whisper-specific colors (cyan/teal theme)
        self._pen_color = (50, 200, 255, 255)
        self._brush_color = (50, 200, 255, 150)
        # Rebuild pen/brush with new colors on next update_display call
        self._pen = None
        self._brush = None
        # Initial display update (show all) with new colors
        self.update_display()

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Extend base StringDataTrack metadata with Whisper-specific fields."""
        base_metadata = super()._get_metadata_for_interval(interval_index)
        if interval_index < 0 or self._display_df is None or interval_index >= len(self._display_df):
            return base_metadata

        row = self._display_df.iloc[interval_index]

        # Prefer Whisper text-preview behavior but preserve any existing base keys
        if "text" in row and pd.notna(row["text"]):
            text = str(row["text"])
            base_metadata["text"] = text
            if len(text) > 50:
                base_metadata["text_preview"] = text[:50] + "..."
            else:
                base_metadata["text_preview"] = text

        if "language" in row and pd.notna(row["language"]):
            base_metadata["language"] = row["language"]

        return base_metadata

