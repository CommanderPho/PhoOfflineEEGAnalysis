import mne
mne.viz.set_browser_backend("qt")  # or "matplotlib"
mne.set_config("MNE_BROWSER_BACKEND", "qt")  # or "matplotlib"
# %gui qt

import xarray as xr # Assuming you're using this
import numpy as np   # For the example

import xarray as xr
import zarr
import panel as pn
import holoviews as hv
hv.extension('bokeh', logo=False)

import hvplot.xarray
import hvplot.pandas
# This line is crucial for displaying plots in a notebook
hvplot.extension('bokeh') # You can also use 'matplotlib' or 'plotly'

# hv.extension('bokeh')
# hv.extension('matplotlib') # or 'matplotlib'
# hv.extension('plotly') # or 'matplotlib'
from holoviews import opts
import panel as pn
pn.extension()

import IPython

# Jupyter-lab enable printing for any line on its own (instead of just the last one in the cell)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Use MNE to load and analyze saved EEG and Motion recordings

import time
import re
from datetime import datetime, timezone

import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from matplotlib import pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import mne
from mne import set_log_level
from copy import deepcopy
import mne

from mne.io import read_raw

datasets = []
# mne.viz.set_browser_backend("Matplotlib")
mne.viz.set_browser_backend("qt")

from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from phopylslhelper.easy_time_sync import EasyTimeSyncParsingMixin, readable_dt_str, from_readable_dt_str

from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
from phoofflineeeganalysis.analysis.historical_data import HistoricalData
from phoofflineeeganalysis.analysis.motion_data import MotionData
from phoofflineeeganalysis.analysis.EEG_data import EEGComputations, EEGData
from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper
# from ..EegProcessing import bandpower
# from phoofflineeeganalysis.EegProcessing import analyze_eeg_trends
from phoofflineeeganalysis.EegVisualization import VisHelpers
from phoofflineeeganalysis.analysis.SavedSessionsProcessor import SavedSessionsProcessor, SessionModality, DataModalityType

set_log_level("WARNING")


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
# 2025-09-18 - LabRecorder XDF Imports
from phoofflineeeganalysis.analysis.EEG_data import EEGData
from phoofflineeeganalysis.analysis.MNE_helpers import DatasetDatetimeBoundsRenderingMixin, RawArrayExtended, RawExtended, up_convert_raw_objects, up_convert_raw_obj
from phoofflineeeganalysis.analysis.SavedSessionsProcessor import LabRecorderXDF, unwrap_single_element_listlike_if_needed
from phoofflineeeganalysis.analysis.SavedSessionsProcessor import XDFDataStreamAccessor

lab_recorder_output_path = Path(r"E:\Dropbox (Personal)\Databases\UnparsedData\LabRecorderStudies\sub-P001").resolve()
assert lab_recorder_output_path.exists()

labRecorder_PostProcessed_path: Path = sso.eeg_analyzed_parent_export_path.joinpath(f'LabRecorder_PostProcessed')
labRecorder_PostProcessed_path.mkdir(exist_ok=True)

should_write_final_merged_eeg_fif: bool = True
# should_write_final_merged_eeg_fif: bool = False
_out_eeg_raw, _out_xdf_stream_infos_df, lab_recorder_xdf_files = LabRecorderXDF.load_and_process_all(lab_recorder_output_path=lab_recorder_output_path, 
                                                                                                     labRecorder_PostProcessed_path=labRecorder_PostProcessed_path, should_write_final_merged_eeg_fif=should_write_final_merged_eeg_fif,
                                                                                                     included_xdf_file_names=included_xdf_file_names, fail_on_exception=False)
xdf_dataset_indicies = np.unique(deepcopy(_out_xdf_stream_infos_df).reset_index(drop=False, inplace=False)['xdf_dataset_idx'].to_numpy())
n_unique_xdf_datasets: int = len(xdf_dataset_indicies)
print(f'n_unique_xdf_datasets: {n_unique_xdf_datasets}')
## 2m 30s

_out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw) # [_out_xdf_stream_infos_df['name'] == 'Epoc X']
_out_xdf_stream_infos_df



from phoofflineeeganalysis.analysis.EEG_data import EEGComputations, EEGData
from phoofflineeeganalysis.PendingNotebookCode import batch_compute_all_eeg_datasets, render_all_spectograms_to_high_quality_pdfs, plot_all_spectograms
from phoofflineeeganalysis.PendingNotebookCode import plot_session_spectogram
## INPUTS: _out_eeg_raw
# Process only the last 5 datasets using 4 workers:
# limit_num_items: int = 150
limit_num_items: int = 5
active_only_out_eeg_raws, results = batch_compute_all_eeg_datasets(eeg_raws=_out_eeg_raw, limit_num_items=limit_num_items, max_workers = 4)

## OUTPUTS: active_only_out_eeg_raws, results
# 1m 19.8s for 25 sessions


from phoofflineeeganalysis.analysis.SavedSessionsProcessor import XDFDataStreamAccessor

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