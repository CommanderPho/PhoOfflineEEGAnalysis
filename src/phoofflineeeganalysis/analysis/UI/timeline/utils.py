"""
Utility functions for timeline track processing.
"""

from typing import Optional, Union
import pandas as pd


def parse_duration_to_seconds_vectorized(series: pd.Series) -> pd.Series:
    """
    Convert duration series to seconds, handling various input types vectorially.
    """
    if series.empty:
        return series
    
    # If already numeric, return as is (coerced to float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
        
    # If timedelta, get total_seconds
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
        
    # Try converting to timedelta first, then seconds
    # This handles strings like '0 days 00:00:19.00'
    try:
        # errors='coerce' will turn invalid parsing into NaT
        deltas = pd.to_timedelta(series, errors='coerce')
        return deltas.dt.total_seconds()
    except Exception:
        # Fallback to numeric conversion
        return pd.to_numeric(series, errors='coerce')


def parse_duration_to_seconds(duration: Union[pd.Timedelta, float, int, str, None]) -> Optional[float]:
    """Legacy helper for scalar conversion."""
    if duration is None or pd.isna(duration):
        return None
    try:
        if isinstance(duration, pd.Timedelta):
            return duration.total_seconds()
        if isinstance(duration, str):
            return pd.to_timedelta(duration).total_seconds()
        return float(duration)
    except Exception:
        return None

