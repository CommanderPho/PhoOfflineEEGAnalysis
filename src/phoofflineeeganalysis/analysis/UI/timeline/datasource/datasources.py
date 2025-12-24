from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtCore


class BaseDatasource(QtCore.QObject):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
        
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal(object) # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):

    """
    source_data_changed_signal = QtCore.pyqtSignal(object) # signal emitted when the internal model data has changed.

    @property
    def total_datasource_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        raise NotImplementedError
        # return (earliest_df_time, latest_df_time)
    
    
    ##### Get/Set Properties ####:
        

    def __init__(self, datasource_name='default_base_datasource'):
        # Initialize the datasource as a QObject
        QtCore.QObject.__init__(self)
        # Custom Setup
        self.custom_datasource_name = datasource_name        
        
        
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        raise NotImplementedError
    
    

class DataframeDatasource(BaseDatasource):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
   
    Contains a dataframe.
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal(object) # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):

    """
    
    @property
    def time_column_name(self):
        """ the name of the relevant time column. Defaults to 't' """
        return 't' 
    
    @property
    def time_column_values(self):
        """ the values of only the relevant time columns """
        return self._df[self.time_column_name] # get only the relevant time column
    
    @property
    def data_column_names(self):
        """ the names of only the non-time columns """
        return np.setdiff1d(self._df.columns, np.array([self.time_column_name])) # get only the non-time columns
    
    @property
    def data_column_values(self):
        """ The values of only the non-time columns """
        return self._df[self.data_column_names]
    
    @property
    def datasource_UIDs(self):
        """The datasource_UID property.
        
        Note: Assumes multiple series are given by the non-time columns:
        
        """
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.data_column_values]
    
    
    ## Active-Only versions of data_column_names, data_column_values, and datasource_UIDs that can be overriden to enable only a subset of the values
    @property
    def active_data_column_names(self):
        """ the names of only the non-time columns """
        return self.data_column_values
        # return self.data_column_names # TODO: why does this return the self.data_column_values instead of self.data_column_names??
    
    @property
    def active_data_column_values(self):
        """ The values of only the non-time columns """
        return self._df[self.active_data_column_names]
    
    @property
    def active_datasource_UIDs(self):
        """The datasource_UID property."""
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.active_data_column_values]
    
    @property
    def total_datasource_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        return self.total_df_start_end_times
        
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        earliest_df_time = np.nanmin(self.df[self.time_column_name])
        latest_df_time = np.nanmax(self.df[self.time_column_name])
        df_timestamps = self.df[self.time_column_name].to_numpy()
        earliest_df_time = df_timestamps[0]
        latest_df_time = df_timestamps[-1]
        return (earliest_df_time, latest_df_time)
    
    
    ##### Get/Set Properties ####:
    @property
    def df(self) -> pd.DataFrame:
        """The df property."""
        return self._df
    @df.setter
    def df(self, value: pd.DataFrame):
        self._df = value
        self.source_data_changed_signal.emit(self)
        

    def __init__(self, df: pd.DataFrame, datasource_name='default_plot_datasource'):
        # Initialize the datasource as a BaseDatasource
        BaseDatasource.__init__(self, datasource_name=datasource_name)
        self._df = df
        assert self.time_column_name in df.columns, f"dataframe must have a time column with name '{self.time_column_name}'.\n\tdf.columns: {list(df.columns)}"
        
        
        
    @classmethod
    def init_from_times_values(cls, times, values):
        plot_df = pd.DataFrame({'t': times, 'v': values})
        return cls(plot_df)
        
    
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        return self.df[self.df[self.time_column_name].between(new_start, new_end)]


class IntervalDataframeDatasource(DataframeDatasource):
    """A DataFrame-backed datasource where the time column name can be customized.

    This is useful for interval/metadata style tables that use columns such as
    'recording_datetime' or 'video_start_datetime' instead of the default 't'.
    """

    def __init__(self, df: pd.DataFrame, time_column_name: str, datasource_name: str = 'interval_datasource'):
        self._custom_time_column_name = time_column_name
        super().__init__(df=df, datasource_name=datasource_name)

    @property
    def time_column_name(self):
        """Override the base time column with the caller-provided name."""
        return self._custom_time_column_name
    

from phoofflineeeganalysis.analysis.SavedSessionsProcessor import LabRecorderXDF

class XDFDatasource(BaseDatasource):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
   
    Contains a dataframe.
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal(object) # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):

    Usage:

        from phoofflineeeganalysis.analysis.UI.timeline.datasource.datasources import XDFDatasource, DataframeDatasource, BaseDatasource


    """
    
    @property
    def time_column_name(self):
        """ the name of the relevant time column. Defaults to 't' """
        # return 'recording_start_datetime'
        return 'stream_start_datetime'
    
    @property
    def time_column_values(self):
        """ the values of only the relevant time columns """
        return self.df[self.time_column_name] # get only the relevant time column
    
    @property
    def data_column_names(self):
        """ the names of only the non-time columns """
        return np.setdiff1d(self.df.columns, np.array([self.time_column_name])) # get only the non-time columns
    
    @property
    def data_column_values(self):
        """ The values of only the non-time columns """
        return self.df[self.data_column_names]
    
    @property
    def datasource_UIDs(self):
        """The datasource_UID property.
        
        Note: Assumes multiple series are given by the non-time columns:
        
        """
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.data_column_values]
    
    
    ## Active-Only versions of data_column_names, data_column_values, and datasource_UIDs that can be overriden to enable only a subset of the values
    @property
    def active_data_column_names(self):
        """ the names of only the non-time columns """
        return self.data_column_values
        # return self.data_column_names # TODO: why does this return the self.data_column_values instead of self.data_column_names??
    
    @property
    def active_data_column_values(self):
        """ The values of only the non-time columns """
        return self.df[self.active_data_column_names]
    
    @property
    def active_datasource_UIDs(self):
        """The datasource_UID property."""
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.active_data_column_values]
    
    @property
    def total_datasource_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        return self.total_df_start_end_times
        
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df, as float timestamps """
        df = self.df
        if df.empty:
            return (0.0, 0.0)
        
        # Try to get start/end times from various possible columns
        earliest_df_time = None
        latest_df_time = None
        
        # Try first_timestamp_dt and last_timestamp_dt first (most reliable for intervals)
        if 'first_timestamp_dt' in df.columns and 'last_timestamp_dt' in df.columns:
            earliest_df_time = np.nanmin(df['first_timestamp_dt'].dropna())
            latest_df_time = np.nanmax(df['last_timestamp_dt'].dropna())
        # Fallback to recording_datetime + duration_sec
        elif 'recording_datetime' in df.columns:
            earliest_df_time = np.nanmin(df['recording_datetime'].dropna())
            if 'duration_sec' in df.columns:
                # Calculate end times from start + duration
                durations = pd.to_timedelta(df['duration_sec'], errors='coerce')
                valid_mask = durations.notna() & df['recording_datetime'].notna()
                if valid_mask.any():
                    end_times = df.loc[valid_mask, 'recording_datetime'] + durations[valid_mask]
                    latest_df_time = np.nanmax(end_times.dropna())
                else:
                    latest_df_time = earliest_df_time
            else:
                latest_df_time = earliest_df_time
        # Fallback to stream_start_datetime if available
        elif 'stream_start_datetime' in df.columns:
            earliest_df_time = np.nanmin(df['stream_start_datetime'].dropna())
            latest_df_time = np.nanmax(df['stream_start_datetime'].dropna())
        
        # Convert datetime to timestamp if needed
        if earliest_df_time is not None and latest_df_time is not None:
            if isinstance(earliest_df_time, pd.Timestamp):
                earliest_df_time = earliest_df_time.timestamp()
            elif hasattr(earliest_df_time, 'timestamp'):
                earliest_df_time = earliest_df_time.timestamp()
            elif isinstance(earliest_df_time, (datetime, pd.Timestamp)):
                earliest_df_time = pd.Timestamp(earliest_df_time).timestamp()
            
            if isinstance(latest_df_time, pd.Timestamp):
                latest_df_time = latest_df_time.timestamp()
            elif hasattr(latest_df_time, 'timestamp'):
                latest_df_time = latest_df_time.timestamp()
            elif isinstance(latest_df_time, (datetime, pd.Timestamp)):
                latest_df_time = pd.Timestamp(latest_df_time).timestamp()
            
            return (float(earliest_df_time), float(latest_df_time))
        
        # If all else fails, return (0, 0)
        return (0.0, 0.0)
    
    
    ##### Get/Set Properties ####:
    @property
    def df(self):
        """The df property."""
        # return self.lab_recorder_xdf.streams_timestamp_dfs
        return self.lab_recorder_xdf.stream_infos

    # @df.setter
    # def df(self, value):
    #     self._df = value
    #     self.source_data_changed_signal.emit(self)


    
    @property
    def lab_recorder_xdf(self) -> LabRecorderXDF:
        """The lab_recorder_xdf property."""
        return self._lab_recorder_xdf
    @lab_recorder_xdf.setter
    def lab_recorder_xdf(self, value: LabRecorderXDF):
        self._lab_recorder_xdf = value
        self.source_data_changed_signal.emit(self)
        
    

    def __init__(self, a_xdf_file: Path, datasource_name='default_plot_datasource'):
        # Initialize the datasource as a BaseDatasource
        BaseDatasource.__init__(self, datasource_name=datasource_name)
        self._xdf_file_path = a_xdf_file
        self._lab_recorder_xdf = LabRecorderXDF.init_from_lab_recorder_xdf_file(a_xdf_file=self._xdf_file_path, should_load_full_file_data=False, debug_print=False)
        # assert self.time_column_name in df.columns, f"dataframe must have a time column with name '{self.time_column_name}'.\n\tdf.columns: {list(df.columns)}"
        


    def get_detailed_data(self):
        assert self._lab_recorder_xdf is not None
        stream_infos, streams_timestamp_dfs, datasets, datasets_dict = self._lab_recorder_xdf.perform_load_xdf_streams(debug_print=False)


    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """
        Get data for the window starting at new_start and ending at new_end.
        
        Handles both datetime and float timestamp inputs. For interval data,
        returns intervals that overlap with the time range. If queried with
        the total time range, returns the full DataFrame.
        """
        df = self.df
        if df.empty:
            return df
        
        # Convert inputs to timestamps if they're datetime objects
        if isinstance(new_start, (datetime, pd.Timestamp)):
            new_start = pd.Timestamp(new_start).timestamp()
        if isinstance(new_end, (datetime, pd.Timestamp)):
            new_end = pd.Timestamp(new_end).timestamp()
        
        new_start = float(new_start)
        new_end = float(new_end)
        
        # Check if this is a query for the full dataset (within small tolerance)
        total_range = self.total_df_start_end_times
        if len(total_range) == 2:
            total_start, total_end = total_range
            # If the query range covers the full dataset (with small tolerance), return all
            if abs(new_start - total_start) < 1.0 and abs(new_end - total_end) < 1.0:
                return df.copy()
        
        # For interval data, filter intervals that overlap with the time range
        # Try multiple column combinations for start/end times
        mask = pd.Series(False, index=df.index)
        
        # Strategy 1: Use first_timestamp_dt and last_timestamp_dt (preferred for intervals)
        if 'first_timestamp_dt' in df.columns and 'last_timestamp_dt' in df.columns:
            starts = df['first_timestamp_dt']
            ends = df['last_timestamp_dt']
            # Convert to timestamps for comparison
            starts_ts = pd.to_datetime(starts, errors='coerce').apply(
                lambda x: x.timestamp() if pd.notna(x) else np.nan
            )
            ends_ts = pd.to_datetime(ends, errors='coerce').apply(
                lambda x: x.timestamp() if pd.notna(x) else np.nan
            )
            # Interval overlaps if: start <= new_end AND end >= new_start
            mask = (starts_ts <= new_end) & (ends_ts >= new_start) & starts_ts.notna() & ends_ts.notna()
        
        # Strategy 2: Use recording_datetime + duration_sec
        elif 'recording_datetime' in df.columns:
            starts = pd.to_datetime(df['recording_datetime'], errors='coerce')
            starts_ts = starts.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
            
            if 'duration_sec' in df.columns:
                durations = pd.to_timedelta(df['duration_sec'], errors='coerce')
                valid_mask = durations.notna() & starts.notna()
                ends_ts = pd.Series(np.nan, index=df.index)
                if valid_mask.any():
                    end_times = starts[valid_mask] + durations[valid_mask]
                    ends_ts[valid_mask] = end_times.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
                # Interval overlaps if: start <= new_end AND end >= new_start
                mask = (starts_ts <= new_end) & (ends_ts >= new_start) & starts_ts.notna() & ends_ts.notna()
            else:
                # No duration, treat as point in time
                mask = (starts_ts >= new_start) & (starts_ts <= new_end) & starts_ts.notna()
        
        # Strategy 3: Fallback to time_column_name (stream_start_datetime)
        elif self.time_column_name in df.columns:
            time_col = df[self.time_column_name]
            # Try to convert to timestamps
            if pd.api.types.is_datetime64_any_dtype(time_col):
                time_ts = pd.to_datetime(time_col, errors='coerce').apply(
                    lambda x: x.timestamp() if pd.notna(x) else np.nan
                )
                mask = (time_ts >= new_start) & (time_ts <= new_end) & time_ts.notna()
            else:
                # Assume already numeric timestamps
                mask = (time_col >= new_start) & (time_col <= new_end) & time_col.notna()
        
        return df[mask].copy()
    
