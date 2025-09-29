import time
import re
from datetime import datetime, timezone, timedelta

import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from matplotlib import pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd

import mne
from mne import set_log_level
from copy import deepcopy
import mne

from mne.io import read_raw
import pyedflib

from phoofflineeeganalysis.analysis.motion_data import MotionData ## for creating single EDF+ files containing channel with different sampling rates (e.g. EEG and MOTION data)

mne.viz.set_browser_backend("Matplotlib")

from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

# from phoofflineeeganalysis.EegProcessing import bandpower
# from ..EegProcessing import bandpower
from numpy.typing import NDArray


set_log_level("WARNING")


class MNEHelpers:
    """ General MNE helper Methods
    Usage:
            from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
            
    """
    @classmethod
    def get_recording_files(cls, recordings_dir: Path, recordings_extensions = ['.fif']):
        found_recording_files = []
        for ext in recordings_extensions:
            found_recording_files.extend(recordings_dir.glob(f"*{ext}"))
            found_recording_files.extend(recordings_dir.glob(f"*{ext.upper()}"))
        return found_recording_files

    @classmethod
    def extract_datetime_from_filename(cls, filename) -> datetime:
        """ Extract the recording start datetime from the file's filename
        Examples: 
        '20250730-195857-Epoc X Motion-raw.fif',
        '20250730-200710-Epoc X-raw.fif',
        '20250618-185519-Epoc X-raw.fif',
        """
        # Regex for the prefix (8 digits + dash + 6 digits) at the start
        match = re.match(r'(\d{8})-(\d{6})', filename)
        if not match:
            raise ValueError("Filename does not match expected format.")

        date_str, time_str = match.groups()
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return dt


    @classmethod
    def get_or_parse_datetime_from_raw(cls, raw, allow_setting_meas_date_from_filename:bool=True) -> datetime:
        """ Get the recording start datetime from the raw.info['meas_date'] or parse from filename if not present
        if allow_setting_meas_date_from_filename is True, it will set the raw.info['meas_date'] if it was None
        """        
        metadata_recording_start_datetime = raw.info.get('meas_date', None)
        if metadata_recording_start_datetime is None:
            parsed_recording_start_datetime = cls.extract_datetime_from_filename(Path(raw.filenames[0]).name)
            metadata_recording_start_datetime = parsed_recording_start_datetime
            if allow_setting_meas_date_from_filename:                
                # Make timezone-aware (UTC)
                # dt = dt.rerawe(tzinfo=timezone.utc)
                raw.info.set_meas_date(deepcopy(parsed_recording_start_datetime).replace(tzinfo=timezone.utc))
                ## WAS UPDATED, probably need to re-save or something
                meas_datetime = raw.info['meas_date']
            else:
                ## don't set it, but still use the parsed datetime
                meas_datetime = deepcopy(parsed_recording_start_datetime).replace(tzinfo=timezone.utc) # raw.info['meas_date']  # This is an absolute datetime or tuple (timestamp, 0)

        else:                    
            meas_datetime = raw.info['meas_date']  # This is an absolute datetime or tuple (timestamp, 0)
            
        return meas_datetime
    


    @classmethod
    def get_raw_datetime_indexed_df(cls, a_raw: mne.io.Raw, dt_col_names: List[str]=["time"], also_process_annotations: bool=True, debug_print:bool=False) -> pd.DataFrame:
        """ Convert dt_col_names columns to absolute datetime
        2025-09-18
        
        Usage:

            dataset_EEG_df = HistoricalData.convert_df_columns_to_datetime(dataset_EEG_df, dt_col_names=["start_time", "end_time"])
            dataset_EEG_df

            dataset_MOTION_df = HistoricalData.convert_df_columns_to_datetime(dataset_MOTION_df, dt_col_names=["start_time", "end_time"])
            dataset_MOTION_df

        """
        a_meas_date: datetime = deepcopy(a_raw.info.get('meas_date')) ## Importantly, the startdate in absolute datetime of the file
        a_df: pd.DataFrame = deepcopy(a_raw.to_data_frame(time_format='timedelta'))
        an_annotations_df: Optional[pd.DataFrame] = None
        if also_process_annotations:
            an_annotations = a_raw.annotations
            if an_annotations is not None:
                an_annotations = deepcopy(an_annotations)
                is_annotation_orig_time_wrong: bool = (an_annotations.orig_time != a_meas_date)
                if is_annotation_orig_time_wrong:
                    # print(f'an_annotations.orig_time: {an_annotations.orig_time} != a_meas_date: {a_meas_date}')
                    ## Update the "onset" column so that it's correct:
                    # an_annotations.set_orig_time(a_meas_date)
                    # an_annotations.orig_time = a_meas_date
                    an_annotations_df = deepcopy(an_annotations.to_data_frame(time_format='timedelta'))
                    onset_abstimes = pd.to_datetime(a_meas_date) + pd.to_timedelta(an_annotations_df['onset'].to_numpy(), unit='s')
                    an_annotations_df['onset'] = onset_abstimes

                else:
                    if debug_print:
                        print(f'original annotation time correct!')
                    an_annotations_df = deepcopy(an_annotations.to_data_frame(time_format='datetime')) ## should be absolute now
                ## OUTPUTS: an_annotations_df
                ## Set new annotations object:
                # new_annotations_obj = mne.Annotations(onset=an_annotations_df['onset'].to_numpy(), duration=an_annotations_df['duration'].to_numpy(), description=an_annotations_df['description'].to_numpy(), orig_time=a_meas_date)
                # a_raw.set_annotations(new_annotations_obj)
        else:
            ## not processing annotations
            pass
        # a_df: pd.DataFrame = deepcopy(an_annotations.to_data_frame(time_format='timedelta'))
        ## Update the df_time_col_name column so that it's correct:
        for a_df_time_col_name in dt_col_names:
            df_time_col_name_abstimes = pd.to_datetime(a_meas_date) + pd.to_timedelta(a_df[a_df_time_col_name].to_numpy(), unit='s')
            a_df[a_df_time_col_name] = df_time_col_name_abstimes
        
        return (a_df, an_annotations_df)
    



    @classmethod
    def convert_df_columns_to_datetime(cls, df: pd.DataFrame, dt_col_names: List[str]=["start_time", "end_time"], add_aggregate_date_columns: bool=True) -> pd.DataFrame:
        """ Convert 'start_time' and 'end_time' columns to datetime
        
        Usage:
        
            dataset_EEG_df = HistoricalData.convert_df_columns_to_datetime(dataset_EEG_df, dt_col_names=["start_time", "end_time"])
            dataset_EEG_df

            dataset_MOTION_df = HistoricalData.convert_df_columns_to_datetime(dataset_MOTION_df, dt_col_names=["start_time", "end_time"])
            dataset_MOTION_df
            
        """
        for col in dt_col_names:
            if col in df.columns:    
                df[col] = pd.to_datetime(df[col], unit="s")
        if add_aggregate_date_columns:
            if 'start_time' in df.columns:
                df['day'] = df['start_time'].dt.floor('D')
                
        return df
    

    @classmethod
    def convert_df_with_boolean_col_to_epochs(cls, df: pd.DataFrame, time_col_names: str='time', is_bad_col_name: str='is_moving', 
                                                                                            annotation_description_name:str="BAD_motion",
                                                                                            meas_date=None,
                                                                                            ) -> mne.Annotations:
        """ Convert 'start_time' and 'end_time' columns to datetime

        Usage:
            from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
            
            dataset_moving_annotations: mne.Annotations = MNEHelpers.convert_df_with_boolean_col_to_epochs(a_motion_df, is_bad_col_name="is_moving", annotation_description_name="BAD_motion")
            dataset_moving_annotations

            dataset_MOTION_df = HistoricalData.convert_df_columns_to_datetime(dataset_MOTION_df, dt_col_names=["start_time", "end_time"])
            dataset_MOTION_df

        """
        assert is_bad_col_name in df
        assert time_col_names in df

        ## INPUTS: meas_date
        active_times = deepcopy(df[time_col_names].to_numpy())
        
        is_bad = df[is_bad_col_name]
        # split into contiguous segments
        idx = np.where(np.diff(is_bad.astype(int)) != 0)[0] + 1 # the +1 is to adjust for the diff operation
        segments = np.split(np.arange(len(is_bad)), idx)

        onsets, durations = [], []
        for seg in segments:
            if is_bad[seg[0]]:
                onsets.append(active_times[seg[0]])
                durations.append(active_times[seg[-1]] - active_times[seg[0]])

        is_moving_annots: mne.Annotations = mne.Annotations(onset=onsets, duration=durations, description=[annotation_description_name]*len(onsets), orig_time=meas_date)
        return is_moving_annots



    @classmethod
    def merge_annotations(cls, raw: mne.io.BaseRaw, new_annots: mne.Annotations, align_to_Raw_meas_time: bool=False):
        """Safely merge additional annotations into a Raw object.

        Aligns onset values if orig_time differs.
        Converts absolute onset values into relative if needed.
        Handles empty or None annotations gracefully.
        """
        if (new_annots is None) or (len(new_annots) == 0):
            return raw  # nothing to merge

        existing_annotations = raw.annotations

        ## align annotations:
        eeg_existing_annotations_orig_time = existing_annotations.orig_time
        new_orig_time = new_annots.orig_time
        
        if align_to_Raw_meas_time:
            meas_date = deepcopy(raw.info.get('meas_date'))
            assert meas_date is not None
            if (existing_annotations is not None) and (len(existing_annotations) > 0):
                ## align existing annotations  
                # Align existing annotations to None:
                if eeg_existing_annotations_orig_time != meas_date:
                    if (eeg_existing_annotations_orig_time is None) and (meas_date is not None):
                        # keep new_annots as-is, raw will adopt its orig_time
                        # raw.set_annotations(existing_annotations + new_annots)
                        pass
                        
                    elif (eeg_existing_annotations_orig_time is not None) and (meas_date is not None):
                        # compute shift between two origins
                        if isinstance(meas_date, datetime) and isinstance(eeg_existing_annotations_orig_time, datetime):
                            shift = (eeg_existing_annotations_orig_time - meas_date).total_seconds()
                        else:
                            shift = float(eeg_existing_annotations_orig_time) - float(meas_date)

                        # adjust onsets to be relative to EEG orig_time
                        existing_annotations_updated_onset = existing_annotations.onset + shift
                        existing_annotations = mne.Annotations(onset=existing_annotations_updated_onset,
                                                    duration=existing_annotations.duration,
                                                    description=existing_annotations.description,
                                                    orig_time=None) # change to None-based time
                        ## set the annotations
                        raw.set_annotations(existing_annotations)

                ## Update: eeg_existing_annotations_orig_time, which should now be None
                eeg_existing_annotations_orig_time = existing_annotations.orig_time
                
                ## Align new annotations
                # Align orig_time
                if meas_date != new_orig_time:
                    if (meas_date is None) and (new_orig_time is not None):
                        # keep new_annots as-is, raw will adopt its orig_time
                        # raw.set_annotations(existing_annotations + new_annots)
                        # return raw
                        pass
                    elif (meas_date is not None) and (new_orig_time is not None):
                        # compute shift between two origins
                        if isinstance(new_orig_time, datetime) and isinstance(meas_date, datetime):
                            shift = (new_orig_time - meas_date).total_seconds()
                        else:
                            shift = float(new_orig_time) - float(meas_date)

                        # adjust onsets to be relative to EEG orig_time
                        new_onset = new_annots.onset + shift
                        new_annots = mne.Annotations(onset=new_onset,
                                                    duration=new_annots.duration,
                                                    description=new_annots.description,
                                                    orig_time=None)

                raw.set_annotations(existing_annotations + new_annots)
                                

        else:
            # If no existing annotations, just set new ones
            if (existing_annotations is None) or (len(existing_annotations) == 0):
                raw.set_annotations(new_annots)
                return raw

            # Align orig_time
            if eeg_existing_annotations_orig_time != new_orig_time:
                if (eeg_existing_annotations_orig_time is None) and (new_orig_time is not None):
                    # keep new_annots as-is, raw will adopt its orig_time
                    raw.set_annotations(existing_annotations + new_annots)
                    return raw
                elif (eeg_existing_annotations_orig_time is not None) and (new_orig_time is not None):
                    # compute shift between two origins
                    if isinstance(new_orig_time, datetime) and isinstance(eeg_existing_annotations_orig_time, datetime):
                        shift = (new_orig_time - eeg_existing_annotations_orig_time).total_seconds()
                    else:
                        shift = float(new_orig_time) - float(eeg_existing_annotations_orig_time)

                    # adjust onsets to be relative to EEG orig_time
                    new_onset = new_annots.onset + shift
                    new_annots = mne.Annotations(onset=new_onset,
                                                duration=new_annots.duration,
                                                description=new_annots.description,
                                                orig_time=eeg_existing_annotations_orig_time)

            raw.set_annotations(existing_annotations + new_annots)



    @classmethod
    def debug_compare_raw_alignments(cls, time_col_name: str = 'time', **raws_kwargs):
        """Safely merge additional annotations into a Raw object.

        Aligns onset values if orig_time differs.
        Converts absolute onset values into relative if needed.
        Handles empty or None annotations gracefully.
        """
        raw_dfs_dict: Dict[str, pd.DataFrame] = {k:v.to_data_frame(time_format='datetime') for k, v in raws_kwargs.items()}
        min_max_times_dict: Dict[str, Tuple[datetime, datetime]] = {k:(a_df[time_col_name].min(), a_df[time_col_name].max()) for k, a_df in raw_dfs_dict.items()}
        
        print(f'min_max_times_dict: {min_max_times_dict}')
        
        # eeg_raw_df = eeg_raws.to_data_frame(time_format='datetime')
        # motion_raw_df = motion_raw.to_data_frame(time_format='datetime')
        return raw_dfs_dict, min_max_times_dict

    @classmethod
    def build_dataset_span_dataframe(cls, a_raw_datasets: List[Union[mne.io.RawArray, mne.io.Raw]], time_col_name: str = 'time', dataset_idx_col_name: str='dataset_idx'):
        """ gets the earleist/latest start/end timestamps corresponding to each Raw/RawArray object and returns a dataframe showing these ranges.
        """
        min_max_times_df: List[Tuple[datetime, datetime]] = [(a_df[time_col_name].min(), a_df[time_col_name].max()) for a_df in [v.to_data_frame(time_format='datetime') for v in a_raw_datasets]]
        min_max_times_df = pd.DataFrame(min_max_times_df, columns=['t_start', 't_end'])
        min_max_times_df[dataset_idx_col_name] = np.arange(len(min_max_times_df))
        min_max_times_df['duration'] = min_max_times_df['t_end'] - min_max_times_df['t_start']
        # Sort by columns: 't_start' (descending), 't_end' (descending)
        min_max_times_df = min_max_times_df.sort_values(['t_start', 't_end'], ascending=[True, True], key=lambda s: s.apply(str) if s.name in ['t_start', 't_end'] else s)
        return min_max_times_df




# ==================================================================================================================================================================================================================================================================================== #
# MNE Raw-like objects enhancer/extender                                                                                                                                                                                                                                               #
# ==================================================================================================================================================================================================================================================================================== #
import mne
from datetime import timedelta

class DatasetDatetimeBoundsRenderingMixin:
    """ works for mne.io.RawArray, mne.io.BaseRaw, mne.io.Raw
    """    
    @classmethod
    def get_raw_timerange(cls, raw):
        meas_date = raw.info['meas_date']
        if meas_date is None:
            return None, None
        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        return meas_date, meas_date + timedelta(seconds=(n_samples - 1) / sfreq)

    def raw_timerange(self):
        return self.get_raw_timerange(self)

    @classmethod
    def fmt_timerange(cls, bounds):
        start, end = bounds
        if start is None or end is None:
            return "N/A"
        return f"{start.isoformat()} → {end.isoformat()}"
    


import scipy.io
import numpy as np

class DatasetRawExportToConvertedFormatFileMixin:

    @classmethod
    def save_mne_raw_to_edf(cls, raw: mne.io.BaseRaw, output_path: Path) -> Path:
        """
        Save an MNE Raw EEG object to a single EDF+ file using pyedflib.
        Attempts to preserve as much metadata as possible, including channel info and compatible annotations.

        Usage:
            from phoofflineeeganalysis.analysis.EEG_data import EEGData

            ## INPUTS: raw_eeg
            edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF").resolve()
            edf_export_parent_path.mkdir(exist_ok=True)

            ## Get paths for current raw:
            curr_fif_file_path: Path = Path(raw_eeg.filenames[0]).resolve()
            curr_file_edf_name: str = curr_fif_file_path.with_suffix('.edf').name

            curr_file_edf_path: Path = edf_export_parent_path.joinpath(curr_file_edf_name).resolve()
            save_mne_raw_to_edf(raw_eeg, curr_file_edf_path)

        History 2025-09-19 copied from `PhoLabStreamingReceiver.src.PhoLabStreamingReceiver.analysis.EEG_data.EEGData.save_mne_raw_to_edf`

        """
        # Get data and metadata
        data = raw.get_data()  # shape: (n_channels, n_samples)
        n_channels, n_samples = data.shape
        ch_names = raw.ch_names
        sfreq = int(raw.info['sfreq'])
        meas_date = raw.info.get('meas_date', None)

        global_physical_min: float = round(float(np.nanmin(data)), 6)
        global_physical_max: float = round(float(np.nanmax(data)), 6)

        # Prepare signal headers
        signal_headers = []
        for i, ch in enumerate(ch_names):
            ch_type = raw.get_channel_types(picks=[i])[0]
            unit = 'uV' if ch_type == 'eeg' else ''
            signal_headers.append({
                'label': ch,
                'dimension': unit,
                'sample_frequency': sfreq,
                'physical_min': global_physical_min,  # <-- round to 6 decimals
                'physical_max': global_physical_max,  # <-- round to 6 decimals
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            })
        # Prepare annotations (EDF+ supports "events" as TALs)
        ann = raw.annotations
        tal_list = []
        if ann is not None and len(ann) > 0:
            for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
                # pyedflib expects onset in seconds, duration in seconds, description as string
                tal_list.append([onset, duration, desc])
        # Write EDF+
        with pyedflib.EdfWriter(str(output_path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as edf:
            edf.setSignalHeaders(signal_headers)
            edf.writeSamples([data[i] for i in range(n_channels)])
            if tal_list:
                for onset, duration, desc in tal_list:
                    edf.writeAnnotation(onset, duration, desc)
            # Set start date/time if available
            if meas_date is not None:
                try:
                    # pyedflib expects a datetime object
                    if isinstance(meas_date, tuple):
                        import datetime
                        meas_date = datetime.datetime.utcfromtimestamp(meas_date[0])
                    edf.setStartdatetime(meas_date)
                except Exception as e:
                    print(f"Could not set start datetime: {e}")
        print(f"Saved EDF+ file to: {output_path}")
        return output_path


    @classmethod
    def save_mne_raw_to_fieldtrip_mat(cls, raw: mne.io.BaseRaw, out_matpath: Path, struct_name='data') -> Path:
        # raw: mne.io.Raw (preloaded or preload as needed)
        # out_matpath: path to output .mat file
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        data = raw.get_data()  # shape (n_channels, n_samples)
        # times in seconds
        # raw.first_samp gives first sample index; we'll set time zero at raw.first_time
        times = np.arange(raw.n_times) / sfreq + raw.first_time

        # Annotations
        ann = raw.annotations
        # ann.onset is in seconds relative to raw.first_time (MNE ensures this) :contentReference[oaicite:0]{index=0}
        # Create struct for annotations
        annot_struct = {
            'onset': np.array(ann.onset),
            'duration': np.array(ann.duration),
            'description': np.array(ann.description, dtype=object)
        }

        # Build ft-style struct
        ft_struct = {
            'label': np.array(ch_names, dtype=object),
            'fsample': sfreq,
            'time': np.array(times),
            'trial': data[np.newaxis, ...],  # FieldTrip often expects trial{1} => data
            # optional: 'chanpos' / 'elec' / location info if available
            'annotation': annot_struct
        }

        scipy.io.savemat(out_matpath, {struct_name: ft_struct}, do_compression=True)
        return out_matpath

    def save_to_edf(self, output_path: Path, **kwargs) -> Path:
        return self.save_mne_raw_to_edf(raw=self, output_path=output_path, **kwargs)


    def save_to_fieldtrip_mat(self, output_path: Path, **kwargs) -> Path:
        return self.save_mne_raw_to_fieldtrip_mat(raw=self, out_matpath=output_path, **kwargs)
    


class RawExtended(DatasetRawExportToConvertedFormatFileMixin, DatasetDatetimeBoundsRenderingMixin, mne.io.Raw):
    
    def __repr__(self):
        base = super().__repr__()
        bounds = self.fmt_timerange(self.raw_timerange()) # self.raw_timerange()
        return f"{base}, timerange={bounds}"
    
        def down_convert_to_base_type(self) ->  mne.io.Raw:
            self.__class__ = mne.io.Raw
            return self


class RawArrayExtended(DatasetRawExportToConvertedFormatFileMixin, DatasetDatetimeBoundsRenderingMixin, mne.io.RawArray):
    
    def __repr__(self):
        base = super().__repr__()
        bounds = self.fmt_timerange(self.raw_timerange()) # self.raw_timerange()
        return f"{base}, timerange={bounds}"


    def down_convert_to_base_type(self) -> mne.io.RawArray:
        self.__class__ = mne.io.RawArray
        return self


def up_convert_raw_obj(raw_obj):
    if isinstance(raw_obj, mne.io.RawArray):
        raw_obj.__class__ = RawArrayExtended
    elif isinstance(raw_obj, mne.io.Raw):
        raw_obj.__class__ = RawExtended
    return raw_obj



def up_convert_raw_objects(raw_objects):
    return [up_convert_raw_obj(obj) for obj in raw_objects]


# datasets = []
# mne.viz.set_browser_backend("Matplotlib")

# data = read_raw("E:\Dropbox (Personal)\Databases\UnparsedData\EmotivEpocX_EEGRecordings\fif\20250808-052237-Epoc X-raw.fif", preload=True)
# datasets.insert(0, data)
# data = read_raw("E:\Dropbox (Personal)\Databases\UnparsedData\EmotivEpocX_EEGRecordings\fif\20250808-015634-Epoc X-raw.fif", preload=True)
# datasets.insert(1, data)
# data.plot(events=events, n_channels=15)
# datasets.insert(2, deepcopy(data))
# data = datasets[2]
# mne.concatenate_raws(data, datasets[0])
