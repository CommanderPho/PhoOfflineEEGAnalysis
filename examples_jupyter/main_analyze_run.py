# Standard library imports
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib import pyplot as plt
from nptyping import NDArray
from numpy.typing import NDArray

# MNE imports
import mne
from mne import set_log_level
from mne.io import read_raw
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

# Visualization imports
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import panel as pn
from holoviews import opts

# IPython imports
import IPython
from IPython.core.interactiveshell import InteractiveShell

# Project-specific imports
from phopylslhelper.easy_time_sync import EasyTimeSyncParsingMixin, readable_dt_str, from_readable_dt_str
from phoofflineeeganalysis.analysis.MNE_helpers import (
    MNEHelpers, DatasetDatetimeBoundsRenderingMixin, RawArrayExtended, 
    RawExtended, up_convert_raw_objects, up_convert_raw_obj
)
from phoofflineeeganalysis.analysis.historical_data import HistoricalData
from phoofflineeeganalysis.analysis.motion_data import MotionData
from phoofflineeeganalysis.analysis.EEG_data import EEGComputations, EEGData
from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper
from phoofflineeeganalysis.EegVisualization import VisHelpers
from phoofflineeeganalysis.analysis.SavedSessionsProcessor import (
    SavedSessionsProcessor, SessionModality, DataModalityType,
    LabRecorderXDF, unwrap_single_element_listlike_if_needed, XDFDataStreamAccessor
)
from phoofflineeeganalysis.PendingNotebookCode import (
    batch_compute_all_eeg_datasets, render_all_spectograms_to_high_quality_pdfs,
    plot_all_spectograms, plot_session_spectogram
)

# Configuration
mne.viz.set_browser_backend("qt")
mne.set_config("MNE_BROWSER_BACKEND", "qt")
set_log_level("WARNING")

hv.extension('bokeh', logo=False)
hvplot.extension('bokeh')
pn.extension()

# Jupyter-lab enable printing for any line on its own (instead of just the last one in the cell)
InteractiveShell.ast_node_interactivity = "all"

# Initialize datasets
datasets = []


# db_root_path = Path('/content/drive/MyDrive/Databases').resolve()
db_root_path = Path(r'E:/Dropbox (Personal)/Databases').resolve()
assert db_root_path.exists(), f"'{db_root_path.as_posix()}' does not exist!"

# eeg_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve()
# headset_motion_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve()

# assert eeg_recordings_file_path.exists()
# assert headset_motion_recordings_file_path.exists()

eeg_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve()
flutter_eeg_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEEG_FlutterRecordings').resolve()
flutter_motion_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEEG_FlutterRecordings/MOTION_RECORDINGS').resolve()
flutter_GENERIC_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEEG_FlutterRecordings/GENERIC_RECORDINGS').resolve()

headset_motion_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve()
WhisperVideoTranscripts_LSL_Converted = db_root_path.joinpath('UnparsedData/WhisperVideoTranscripts_LSL_Converted').resolve()
pho_log_to_LSL_recordings_path: Path = db_root_path.joinpath('UnparsedData/PhoLogToLabStreamingLayer_logs').resolve()
## These contain little LSL .fif files with names like: '20250808_062814_log.fif',

eeg_analyzed_parent_export_path = db_root_path.joinpath('AnalysisData/MNE_preprocessed').resolve()
pickled_data_path = db_root_path.joinpath('AnalysisData/MNE_preprocessed/PICKLED_COLLECTION').resolve()
assert pickled_data_path.exists()


# n_most_recent_sessions_to_preprocess: int = None # None means all sessions
# n_most_recent_sessions_to_preprocess: int = 35
n_most_recent_sessions_to_preprocess: int = 5
# n_most_recent_sessions_to_preprocess = None



# SavedSessionProcessor

sso: SavedSessionsProcessor = SavedSessionsProcessor(eeg_recordings_file_path=eeg_recordings_file_path,
                                                     headset_motion_recordings_file_path=headset_motion_recordings_file_path, WhisperVideoTranscripts_LSL_Converted_file_path=WhisperVideoTranscripts_LSL_Converted, pho_log_to_LSL_recordings_path=pho_log_to_LSL_recordings_path,
                                                     eeg_analyzed_parent_export_path=eeg_analyzed_parent_export_path, 
                                                     n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess, 
                                                     should_load_data=True, should_load_preprocessed=False,
                                                    #  should_load_data=True, should_load_preprocessed=True,
													)
# included_xdf_file_names = [
# 	"E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-21T051157.400Z_eeg.xdf", ## When it started to work
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-20T215045.162Z_eeg.xdf"
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-20T164055.381Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-18T092615.398Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-17T215112.606Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-17T124127.644Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-17T214946.083Z_eeg.xdf",
#     # # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-09-22T182649.051Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-16T220233.548Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-16T212744.771Z_eeg.xdf",
#     # # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-16T212721.939Z_eeg.xdf",
#     # # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-16T212528.076Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-09-23T141026.412Z_eeg.xdf",
#     # "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-09-22T213547.659Z_eeg.xdf",
# ]

# included_xdf_file_names = [Path(v).resolve() for v in included_xdf_file_names]
# included_xdf_file_names = [v.name for v in included_xdf_file_names]


included_xdf_file_names = None ## include all 
included_xdf_file_names


lab_recorder_output_path = Path(r"E:\Dropbox (Personal)\Databases\UnparsedData\LabRecorderStudies\sub-P001").resolve()
assert lab_recorder_output_path.exists()

labRecorder_PostProcessed_path: Path = sso.eeg_analyzed_parent_export_path.joinpath(f'LabRecorder_PostProcessed')
labRecorder_PostProcessed_path.mkdir(exist_ok=True)

should_write_final_merged_eeg_fif: bool = True
# should_write_final_merged_eeg_fif: bool = False

# Parallel XDF file processing
from phoofflineeeganalysis.analysis.MNE_helpers import DatasetDatetimeBoundsRenderingMixin, RawArrayExtended, RawExtended, up_convert_raw_objects, up_convert_raw_obj
from phoofflineeeganalysis.analysis.EEG_data import EEGData

assert lab_recorder_output_path.exists()

lab_recorder_xdf_files: list[Path] = list(lab_recorder_output_path.glob('*.xdf'))
n_total_found_files: int = len(lab_recorder_xdf_files)
if included_xdf_file_names is not None:
    print(f'limiting to included_xdf_file_names: {included_xdf_file_names}...')
    lab_recorder_xdf_files = [v for v in lab_recorder_xdf_files if v.name in included_xdf_file_names]
    n_filtered_found_files: int = len(lab_recorder_xdf_files)
    print(f'\tlimited to {n_filtered_found_files}/{n_total_found_files} files')

if (labRecorder_PostProcessed_path is not None) and should_write_final_merged_eeg_fif:
    labRecorder_PostProcessed_path.mkdir(exist_ok=True)

# Determine optimal number of workers
max_workers = min(len(lab_recorder_xdf_files), (os.cpu_count() or 4))
print(f"Processing {len(lab_recorder_xdf_files)} XDF files using {max_workers} parallel workers...")

# Initialize result containers
_out_eeg_raw = [None] * len(lab_recorder_xdf_files)
_out_xdf_stream_infos_df = [None] * len(lab_recorder_xdf_files)

def process_single_xdf_file(idx_file_tuple):
    """Process a single XDF file"""
    an_xdf_file_idx, a_xdf_file = idx_file_tuple
    try:
        print(f'  Processing XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"...')
        
        # Load XDF file
        stream_infos, raws, raws_dict = LabRecorderXDF.init_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file)
        eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])
        
        if len(eeg_raws) != 1:
            raise ValueError(f'for file "{a_xdf_file.as_posix()}": len(eeg_raws): {len(eeg_raws)}, but only handle the single eeg file case.')
        
        eeg_raw = eeg_raws[0]
        
        # Add metadata
        stream_infos['lab_recorder_xdf_file_idx'] = an_xdf_file_idx
        stream_infos['xdf_filename'] = a_xdf_file.name
        
        # Save post-processed data if requested
        if should_write_final_merged_eeg_fif:
            eeg_raw, a_lab_recorder_exports_filepaths_dict = LabRecorderXDF.save_post_processed_to_fif(
                raws_dict=raws_dict,
                a_xdf_file=a_xdf_file,
                labRecorder_PostProcessed_path=labRecorder_PostProcessed_path,
            )
            if a_lab_recorder_exports_filepaths_dict is not None:
                for a_format, an_export_path in a_lab_recorder_exports_filepaths_dict.items():
                    stream_infos[f'proccessed_{a_format}_filename'] = an_export_path.name
        
        # Up-convert and set montage
        eeg_raw = up_convert_raw_obj(eeg_raw)
        EEGData.set_montage(datasets_EEG=[eeg_raw])
        eeg_raw.debug_test_annotations_timestamps()
        

        ## Do post-processing stage
        # a_raw = eeg_raw
        result = None
        try:
            meas_date = eeg_raw.info.get('meas_date', 'Unknown')
            print(f"  Processing dataset {an_xdf_file_idx+1}/{len(active_only_out_eeg_raws)} (meas_date: {meas_date})")
            result = EEGComputations.run_all(raw=eeg_raw)
            print(f"  Completed dataset {an_xdf_file_idx+1}/{len(active_only_out_eeg_raws)} (meas_date: {meas_date})")
            # return an_xdf_file_idx, result
        except Exception as e:
            print(f"  ERROR processing dataset {an_xdf_file_idx+1}: {e}")
            # return an_xdf_file_idx, None



        print(f'  Completed XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"')
        return an_xdf_file_idx, eeg_raw, stream_infos
        
    except (ValueError, KeyError, AssertionError, TypeError) as e:
        print(f'  ERROR in XDF file {an_xdf_file_idx+1}: {e}\n  Skipping file.')
        return an_xdf_file_idx, None, None
        
    except Exception as e:
        print(f'  EXCEPTION in XDF file {an_xdf_file_idx+1}: {e}')
        raise













# Parallel processing using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_idx = {
        executor.submit(process_single_xdf_file, (idx, xdf_file)): idx 
        for idx, xdf_file in enumerate(lab_recorder_xdf_files)
    }
    
    # Collect results as they complete
    for future in as_completed(future_to_idx):
        try:
            idx, eeg_raw, stream_infos = future.result()
            _out_eeg_raw[idx] = eeg_raw
            _out_xdf_stream_infos_df[idx] = stream_infos
        except Exception as e:
            idx = future_to_idx[future]
            print(f"  EXCEPTION collecting result for file {idx+1}: {e}")
            _out_eeg_raw[idx] = None
            _out_xdf_stream_infos_df[idx] = None

# Filter out failed files
valid_indices = [i for i, (raw, info) in enumerate(zip(_out_eeg_raw, _out_xdf_stream_infos_df)) if raw is not None and info is not None]
_out_eeg_raw = [_out_eeg_raw[i] for i in valid_indices]
_out_xdf_stream_infos_df = [_out_xdf_stream_infos_df[i] for i in valid_indices]

# Add xdf_dataset_idx to stream_infos
for dataset_idx, stream_infos in enumerate(_out_xdf_stream_infos_df):
    stream_infos['xdf_dataset_idx'] = dataset_idx

# Combine stream infos
_out_xdf_stream_infos_df = pd.concat(_out_xdf_stream_infos_df)
_out_xdf_stream_infos_df = _out_xdf_stream_infos_df.set_index('xdf_dataset_idx')

# Up-convert and sort
_out_eeg_raw = up_convert_raw_objects(_out_eeg_raw)
_out_eeg_raw.sort(key=lambda r: (r.raw_timerange()[0] is None, r.raw_timerange()[0]))

# Set montage for all datasets
EEGData.set_montage(datasets_EEG=_out_eeg_raw)  # pyright: ignore[reportUnknownMemberType]

xdf_dataset_indicies = np.unique(deepcopy(_out_xdf_stream_infos_df).reset_index(drop=False, inplace=False)['xdf_dataset_idx'].to_numpy())
n_unique_xdf_datasets: int = len(xdf_dataset_indicies)
print(f'n_unique_xdf_datasets: {n_unique_xdf_datasets}')

_out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw)
_out_xdf_stream_infos_df




## INPUTS: _out_eeg_raw
# Process only the last 5 datasets using 4 workers:
# limit_num_items: int = 150
limit_num_items: int = 5
active_only_out_eeg_raws, results = batch_compute_all_eeg_datasets(eeg_raws=_out_eeg_raw, limit_num_items=limit_num_items, max_workers = 4)

## OUTPUTS: active_only_out_eeg_raws, results
# 1m 19.8s for 25 sessions




num_sessions: int = len(results)
num_sessions

# xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
# xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df[_out_xdf_stream_infos_df['name'] == 'Epoc X'], active_only_out_eeg_raws=active_only_out_eeg_raws)
xdf_stream_infos_df


## Extract comments/notes/annotations/etc from the outputs

_extracted_comments = []
ignored_comment_descriptions = ['BAD_motion', '']
for a_raw in active_only_out_eeg_raws:
    an_annotations = a_raw.annotations
    if (an_annotations is not None) and (len(an_annotations) > 0):
        an_annotation_df = an_annotations.to_data_frame(time_format='datetime')
        an_annotation_df = an_annotation_df[np.logical_not(np.isin(an_annotation_df['description'], ignored_comment_descriptions))]
        _extracted_comments.append(an_annotation_df)
        # an_annotation_df


extracted_comments_df: pd.DataFrame = pd.concat(_extracted_comments)
extracted_comments_df = extracted_comments_df.rename(columns={'onset':'time', 'description':'text'}, inplace=False)
extracted_comments_df

# netcdf_save_path = Path("2025-10-16_saved_spectogram.nc")
outputs_root_folder: Path = Path(r'L:\AITEMP\PhoOfflineEEGAnalysisOutputs').resolve()
outputs_root_folder.exists()

# netcdf_save_path = Path("2025-10-16_saved_spectogram.nc")
netcdf_save_path: Path = outputs_root_folder.joinpath("2025-11-12_saved_spectogram.nc").resolve()
combined_spectogram_da.to_netcdf(netcdf_save_path)
print(f'netcdf_save_path: "{netcdf_save_path.as_posix()}"')
# combined_spectogram_da.to_
# combined_spectogram_ds.to_netcdf(