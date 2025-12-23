from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget

from phoofflineeeganalysis.analysis.UI.timeline.tracks.BaseTrackWidget import TrackWidget
from phoofflineeeganalysis.analysis.UI.timeline.utils import parse_duration_to_seconds_vectorized
from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import BaseDatasource, IntervalDataframeDatasource


class StringDataTrack(TrackWidget):
    """
    Base class for timeline tracks that display timestamped string/comment data.

    Expects a DataFrame with at least:
    - onset column: datetime (start time)
    - duration column: float or Timedelta (duration in seconds)
    """

    def __init__(self, source, name: str = "Comments", height: int = 60, parent: Optional[QWidget] = None, onset_col: str = "onset", duration_col: str = "duration", defer_update:bool=False):
        super().__init__(name=name, height=height, parent=parent)

        self._onset_col = onset_col
        self._duration_col = duration_col

        if isinstance(source, BaseDatasource):
            self.set_datasource(source)
            df = self._get_full_dataframe()
            self._df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        else:
            df = source
            self._df = df.copy()
            interval_ds = IntervalDataframeDatasource(self._df, time_column_name=self._onset_col, datasource_name=name)
            self.set_datasource(interval_ds)
        self._display_df = pd.DataFrame()

        if self._onset_col in self._df.columns:
            self._df[self._onset_col] = self._ensure_utc_naive(self._df[self._onset_col])

        self._cache_intervals()
        if not defer_update:
            self.update_display()

    def _get_recording_intervals_vectorized(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract generic string/comment intervals from DataFrame using vectorized operations.

        Prefers data coming from an attached datasource when available.
        """
        df_full = self._get_full_dataframe()
        if isinstance(df_full, pd.DataFrame):
            self._df = df_full.copy()

        if self._df.empty or self._onset_col not in self._df.columns:
            self._display_df = pd.DataFrame()
            return np.empty((0, 2)), []

        df = self._df.copy()
        start_dt = df[self._onset_col]

        if self._duration_col in df.columns:
            durations = parse_duration_to_seconds_vectorized(df[self._duration_col])
        else:
            durations = pd.Series(0.0, index=df.index)

        durations = durations.fillna(0.1)
        durations[durations <= 0] = 0.1

        end_dt = start_dt + pd.to_timedelta(durations, unit="s")

        mask = start_dt.notna() & end_dt.notna()
        self._display_df = df[mask].copy().reset_index(drop=True)
        self._display_df["final_end_dt"] = end_dt[mask].reset_index(drop=True)
        self._display_df["final_duration"] = durations[mask].reset_index(drop=True)

        if self._display_df.empty:
            return np.empty((0, 2)), []

        starts = self._display_df[self._onset_col].values.astype("datetime64[ns]").astype(np.float64) / 1e9
        ends = self._display_df["final_end_dt"].values.astype("datetime64[ns]").astype(np.float64) / 1e9

        return np.column_stack([starts, ends]), []

    def _get_recording_intervals(self) -> List[Tuple[datetime, datetime]]:
        """Legacy API - subclasses relying on StringDataTrack should not use this."""
        return []

    def _cache_metadata(self):
        """Legacy metadata hook - not used for StringDataTrack-based subclasses."""
        pass

    def _get_metadata_for_interval(self, interval_index: int) -> Dict[str, Any]:
        """Default metadata for string/comment intervals."""
        if interval_index < 0 or self._display_df is None or interval_index >= len(self._display_df):
            return {}

        row = self._display_df.iloc[interval_index]
        metadata: Dict[str, Any] = {}

        if "final_duration" in row:
            metadata["duration_sec"] = row["final_duration"]

        text_value: Optional[str] = None
        if "text" in row and pd.notna(row["text"]):
            text_value = str(row["text"])
        elif "message" in row and pd.notna(row["message"]):
            text_value = str(row["message"])

        if text_value is not None:
            metadata["text"] = text_value
            if len(text_value) > 50:
                metadata["text_preview"] = text_value[:50] + "..."
            else:
                metadata["text_preview"] = text_value

        if "message" in row and pd.notna(row["message"]):
            metadata["message"] = str(row["message"])
        if "label" in row and pd.notna(row["label"]):
            metadata["label"] = str(row["label"])

        return metadata


