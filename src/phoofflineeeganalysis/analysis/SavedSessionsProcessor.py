import time
import re
from datetime import datetime, timezone
import pytz
# from pytz import timezone

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import dill

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
# from phoofflineeeganalysis.tzinfo_examples import Eastern


datasets = []
mne.viz.set_browser_backend("Matplotlib")
from attrs import define, field, Factory

from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

from phoofflineeeganalysis.EegProcessing import bandpower
from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers, up_convert_raw_objects
from phoofflineeeganalysis.analysis.historical_data import HistoricalData
from phoofflineeeganalysis.analysis.motion_data import MotionData
from phoofflineeeganalysis.analysis.EEG_data import EEGData
from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper
# from ..EegProcessing import bandpower

from phoofflineeeganalysis.EegProcessing import analyze_eeg_trends
from phoofflineeeganalysis.analysis.EEG_data import EEGData
from phoofflineeeganalysis.analysis.motion_data import MotionData
from phoofflineeeganalysis.analysis.event_data import EventData
from phoofflineeeganalysis.analysis.historical_data import HistoricalData

set_log_level("WARNING")

from enum import Enum, auto

import pyxdf
import mne
import numpy as np
from benedict import benedict


class DataModalityType(Enum):
    """The various types of datastreams produced by my recorder and analyzed."""
    EEG = auto()
    MOTION = auto()
    PHO_LOG_TO_LSL = auto()
    WHISPER = auto()
    # PHO_LOG_TO_LSL = auto()

    def __str__(self):
        return self.name

    @classmethod
    def list_values(cls):
        """Returns a list of all enum values"""
        return list(cls)

    @classmethod
    def list_names(cls):
        """Returns a list of all enum names"""
        return [e.name for e in cls]


@define(slots=False)
class SessionModality:
    """ Data corresponding to a specific type or 'modality' of input (e.g. EEG, MOTION, PHO_LOG_TO_LSL, WHISPER, etc.
    """
    all_data: Optional[Any] = field(default=None)
    all_times: Optional[Any] = field(default=None)
    datasets: Optional[Any] = field(default=None)
    df: Optional[pd.DataFrame] = field(default=None)
    active_indices: Optional[Any] = field(default=None)
    analysis_results: Optional[Any] = field(default=None)


    def filtered_by_day_date(self, search_day_date: datetime, debug_print=False) -> "SessionModality":
        """ Returns a new SessionModality instance filtered to only include datasets from the specified date.
        
        today_only_modality = a_modality.filtered_by_day_date(search_day_date=datetime(2025, 8, 8))
        
        
        """
        if self.df is None or self.datasets is None:
            raise ValueError("Both 'df' and 'datasets' must be loaded to filter by date.")

        # Ensure the date has no time component
        search_day_date = search_day_date.replace(hour=0, minute=0, second=0, microsecond=0)

        today_only_modality = deepcopy(self)
        is_dataset_included = np.isin(self.active_indices, self.df[self.df['day'] == search_day_date]['dataset_IDX'].values)
        if debug_print:
            print(f'\tis_dataset_included: {is_dataset_included}')
        today_only_modality.df = self.df[self.df['day'] == search_day_date] ## filter the today_only modalities version
        today_only_modality.active_indices = self.active_indices[is_dataset_included]
        # _curr_included_IDXs = np.arange(len(a_modality.datasets))[is_dataset_included]
        # print(f'\t_curr_included_IDXs: {_curr_included_IDXs}')
        # today_only_modality.datasets = [a_modality.datasets[i] for i in _curr_included_IDXs]
        today_only_modality.datasets = [self.datasets[i] for i in today_only_modality.active_indices]
        today_only_modality.analysis_results = [self.analysis_results[i] for i in today_only_modality.active_indices]
        return today_only_modality





    

@define(slots=False)
class SavedSessionsProcessor:
    """ Top-level manager of EEG recordings
    

    from phoofflineeeganalysis.analysis.SavedSessionsProcessor import SavedSessionsProcessor, SessionModality, DataModalityType
     
    sso: SavedSessionsProcessor = SavedSessionsProcessor()
    sso

    """
    eeg_recordings_file_path: Path = field(default=Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve())
    headset_motion_recordings_file_path: Path = field(default=Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve())
    WhisperVideoTranscripts_LSL_Converted_file_path: Path = field(default=Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/WhisperVideoTranscripts_LSL_Converted").resolve())
    pho_log_to_LSL_recordings_path: Path = field(default=Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/PhoLogToLabStreamingLayer_logs').resolve())
    ## These contain little LSL .fif files with names like: '20250808_062814_log.fif', 

    eeg_analyzed_parent_export_path = field(default=Path("E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed").resolve())

    # n_most_recent_sessions_to_preprocess: int = None # None means all sessions
    n_most_recent_sessions_to_preprocess: int = field(default=10) #
    should_load_data: bool = field(default=False)
    should_load_preprocessed: bool = field(default=False)

    ## Loaded variables
    found_recording_file_modality_dict: Dict[str, List[Path]] = field(factory=dict, init=False)
    flat_data_modality_dict: Dict[str, Tuple] = field(factory=dict, init=False)

    ## This is the core data-storage variable for this class, that holds all the loaded/parsed results and datasets
    modalities: Dict[str, SessionModality] = field(factory=dict, init=False)


    def run(self):
        """ Loads data (either fresh or pre-processed) and then calls `self.perform_post_processing()`

        Calls:        
                self.perform_post_processing()
                
        """
        ## Load pre-proocessed EEG data:
        if self.should_load_preprocessed:
            self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                eeg_recordings_file_path = self.eeg_analyzed_parent_export_path,
                # headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
                # WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
                # pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
                should_load_data=self.should_load_data,
            )
            ## Just get the previously processed EEG data, do not load other modalities            

            # self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
            #     eeg_recordings_file_path = self.eeg_analyzed_parent_export_path,
            #     # headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
            #     WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
            #     pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
            #     should_load_data=self.should_load_data,
            # )

            ## #TODO 2025-09-09 16:14: - [ ] Find the files that changed since last processing, and only load those:
            self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                eeg_recordings_file_path = self.eeg_recordings_file_path,
                headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
                WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
                pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
                should_load_data=self.should_load_data,
            )


        else:
            ## Old way:
            self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                eeg_recordings_file_path = self.eeg_recordings_file_path,
                headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
                WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
                pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
                should_load_data=self.should_load_data,
            )


        # 1m 10s

        self.perform_post_processing()
        


    def perform_post_processing(self) -> Dict[str, SessionModality]:
        """Performs batch post-processing on all loaded modalities in `self.flat_data_modality_dict`.

        Runs each modality's preprocessing in parallel (threaded) since operations
        are independent and operate on different files. Returns a mapping from
        modality key to `SessionModality` with results.

        
        Calls: 
            self.perform_extended_post_processing_steps()
            
        """
        # Map modality keys to their preprocessors and any relevant param
        preprocessors: Dict[str, Callable[..., Tuple[Any, Any]]] = {
            "EEG": EEGData.preprocess,
            "MOTION": MotionData.preprocess,
            "PHO_LOG_TO_LSL": EventData.preprocess,
            "WHISPER": EventData.preprocess,
        }

        # Only process modalities that are actually present
        keys_to_process: List[str] = [k for k in preprocessors.keys() if k in self.flat_data_modality_dict]

        results: Dict[str, SessionModality] = {}
        errors: Dict[str, Exception] = {}

        def _process_modality(key: str) -> Tuple[str, SessionModality]:
            preproc_func = preprocessors[key]
            unpacked = self.flat_data_modality_dict[key]
            all_data, all_times = unpacked[0]
            datasets = unpacked[1]
            df = unpacked[2]

            print(f'\tstarting post-process modality: {key}')
            if key == "EEG":
                active_indices, analysis_results = preproc_func(
                    datasets_EEG=datasets,
                    preprocessed_EEG_save_path=None,
                    n_most_recent_sessions_to_preprocess=self.n_most_recent_sessions_to_preprocess,
                )
            else:
                active_indices, analysis_results = preproc_func(
                    datasets,
                    n_most_recent_sessions_to_preprocess=self.n_most_recent_sessions_to_preprocess,
                )

            modality_result = SessionModality(
                all_data=all_data,
                all_times=all_times,
                datasets=datasets,
                df=df,
                active_indices=active_indices,
                analysis_results=analysis_results,
            )
            print(f'\tfinished post-process modality: {key}')
            return key, modality_result

        max_workers = max(1, min(len(keys_to_process), (os.cpu_count() or 4)))
        if keys_to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {executor.submit(_process_modality, key): key for key in keys_to_process}
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        k, modality_result = future.result()
                        results[k] = modality_result
                    except Exception as e:
                        print(f"\tERROR while post-processing modality '{key}': {e}")
                        errors[key] = e

        # Update self.modalities with successful results
        for k, modality_result in results.items():
            self.modalities[k] = modality_result

        # Perform extended steps that depend on multiple modalities
        try:
            self.perform_extended_post_processing_steps()
        except (ValueError, TypeError, AttributeError) as e:
            print(f'encountered error: {e} while trying to perform perform_extended_post_processing_steps(). Skipping and returning.')
        except Exception as e:
            raise e

        return results
        
    def setup_specific_modality(self, modality_type: List[DataModalityType], should_load_data: bool=False):
        """ called to discover and load all files related to a specific modality, such as EEG, WHISPER recordings, etc.
        
        """
        if not isinstance(modality_type, (list, tuple)):
            ## wrap in a list
            modality_type = [modality_type] ## single element list


        MAIN_process_recording_files_kwargs = {}
        for a_modality in modality_type:
            ## find the correct kwarg name and corresponding value
            if a_modality.name == DataModalityType.EEG.name:
                MAIN_process_recording_files_kwargs.update(dict(eeg_recordings_file_path = self.eeg_recordings_file_path))
            elif a_modality.name == DataModalityType.MOTION.name:
                MAIN_process_recording_files_kwargs.update(dict(headset_motion_recordings_file_path = self.headset_motion_recordings_file_path))
            elif a_modality.name == DataModalityType.PHO_LOG_TO_LSL.name:
                MAIN_process_recording_files_kwargs.update(dict(pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path))
            elif a_modality.name == DataModalityType.WHISPER.name:
                MAIN_process_recording_files_kwargs.update(dict(WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path))
            # elif a_modality.name == DataModalityType.EEG.name:
            # 	MAIN_process_recording_files_kwargs.update(dict(eeg_recordings_file_path = self.eeg_recordings_file_path))
            else:
                raise NotImplementedError(f'Unknown modality type: {a_modality}')


        flat_data_modality_dict, found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                        **MAIN_process_recording_files_kwargs,
                        should_load_data=should_load_data,
        )
        

        ## iterate and add to self
        for k, v in flat_data_modality_dict.items():
            self.flat_data_modality_dict[k] = v
        
        for k, v in found_recording_file_modality_dict.items():
            self.found_recording_file_modality_dict[k] = v

        ## self.modalities is not changed :[

        return (flat_data_modality_dict, found_recording_file_modality_dict)
    


    def perform_extended_post_processing_steps(self):
        # Do annotation/join only if needed, still avoid repetition:
        if ("PHO_LOG_TO_LSL" in self.modalities):
            (dataset_PHOLOG_df, dataset_EEG_df_PHOLOG) = HistoricalData.add_additional_LOGGING_annotations(
                active_EEG_IDXs=self.modalities["EEG"].active_indices,
                datasets_EEG=self.modalities["EEG"].datasets,
                active_LOGGING_IDXs=self.modalities["PHO_LOG_TO_LSL"].active_indices,
                datasets_LOGGING=self.modalities["PHO_LOG_TO_LSL"].datasets,
                analysis_results_LOGGING=self.modalities["PHO_LOG_TO_LSL"].analysis_results,
                logging_series_identifier="PHO_LOG",
                preprocessed_EEG_save_path=None
            )
            if dataset_EEG_df_PHOLOG is not None:
                self.modalities["EEG"].df = dataset_EEG_df_PHOLOG
            if dataset_PHOLOG_df is not None:
                self.modalities["PHO_LOG_TO_LSL"].df = dataset_PHOLOG_df


        if ("WHISPER" in self.modalities):
            (dataset_WHISPER_df, dataset_EEG_df_WHISPER) = HistoricalData.add_additional_LOGGING_annotations(
                active_EEG_IDXs=self.modalities["EEG"].active_indices,
                datasets_EEG=self.modalities["EEG"].datasets,
                active_LOGGING_IDXs=self.modalities["WHISPER"].active_indices,
                datasets_LOGGING=self.modalities["WHISPER"].datasets,
                analysis_results_LOGGING=self.modalities["WHISPER"].analysis_results,
                logging_series_identifier="WHISPER",
                preprocessed_EEG_save_path=None
            )
            self.modalities["EEG"].df = dataset_EEG_df_WHISPER
            self.modalities["WHISPER"].df = dataset_WHISPER_df

        if ("EEG" in self.modalities) and ("MOTION" in self.modalities):
            dataset_MOTION_df, dataset_EEG_df = HistoricalData.add_bad_periods_from_MOTION_data(active_EEG_IDXs=self.modalities["EEG"].active_indices,
                                                        datasets_EEG=self.modalities["EEG"].datasets,
                                                        active_motion_IDXs=self.modalities["MOTION"].active_indices, datasets_MOTION=self.modalities["MOTION"].datasets, analysis_results_MOTION=self.modalities["MOTION"].analysis_results,
                                                        preprocessed_EEG_save_path=self.eeg_analyzed_parent_export_path)
            self.modalities["EEG"].df = dataset_EEG_df
            self.modalities["MOTION"].df = dataset_MOTION_df


    # ==================================================================================================================================================================================================================================================================================== #
    # Pickling/Exporting                                                                                                                                                                                                                                                                   #
    # ==================================================================================================================================================================================================================================================================================== #

    def save(self, pkl_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION")):
        """ Pickles the object 
        """
        # pkl_path

        # data_path = Path(r"C:/Users/pho/repos/EmotivEpoc/PhoLabStreamingReceiver/data").resolve()
        # assert data_path.exists()

        # pickled_data_path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION").resolve()
        if pkl_path.resolve().is_dir():
            assert pkl_path.exists(), f"Directory {pkl_path.as_posix()} must exist!"
            pkl_path = pkl_path.joinpath("2025-09-02_50records_SSO_all.pkl").resolve()
        else:
            print(f'pkl_path is already a direct pkl file name: "{pkl_path.as_posix()}"')

        print(f'Pickling all data to "{pkl_path.as_posix()}"...')
        with open(pkl_path, "wb") as f:
            dill.dump(self, f)
        print(f'\tdone.')


    @classmethod
    def load(cls, pkl_file: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION/records_SSO_all.pkl")) -> "SavedSessionsProcessor":
        """ un-Pickles the object 
        
        sso: SavedSessionsProcessor = SavedSessionsProcessor.load(pkl_file=Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION/2025-09-02_50records_SSO_all.pkl").resolve())
        """
        assert pkl_file.exists(), f"'{pkl_file.as_posix()}' must exist!"
        assert pkl_file.exists(), f"'{pkl_file.is_file()}' must be a pickle file!"
        with open(pkl_file, "rb") as f:
            loaded_instance = dill.load(f)
            return loaded_instance



    # ==================================================================================================================================================================================================================================================================================== #
    # Exporting to other formats                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def save_to_EDF(self, edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF")) -> List[Path]:
        """ saves the EEG files (post-processing) out to EDF files for viewing in EDFViewer or similar applications.

        edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF").resolve()
                
        written_EDF_file_paths = sso.save_to_EDF()
        
        """
        from phoofflineeeganalysis.analysis.MNE_helpers import up_convert_raw_objects
        from phoofflineeeganalysis.analysis.EEG_data import EEGData


        edf_export_parent_path.mkdir(exist_ok=True)
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = self.flat_data_modality_dict['EEG']  ## Unpacking
        datasets_EEG = up_convert_raw_objects(datasets_EEG) ## upconvert
        written_EDF_file_paths = []
        for i, raw_eeg in enumerate(datasets_EEG):
            ## INPUTS: raw_eeg
            ## Get paths for current raw:
            try:
                curr_fif_file_path: Path = Path(raw_eeg.filenames[0]).resolve()
                curr_file_edf_name: str = curr_fif_file_path.with_suffix('.edf').name
                curr_file_edf_path: Path = edf_export_parent_path.joinpath(curr_file_edf_name).resolve()
                curr_file_edf_path = raw_eeg.save_to_edf(output_path=curr_file_edf_path)
                # EEGData.save_mne_raw_to_edf(raw_eeg, curr_file_edf_path)
                written_EDF_file_paths.append(curr_file_edf_path)
            except (ValueError, FileNotFoundError, FileExistsError, AttributeError, OSError, TypeError) as e:
                print(f'\tWARNING: could not export EEG dataset index {i} to EDF file, skipping... Error: {e}')
                
            except Exception as e:
                raise
        # END for i, raw_eeg in enumerate(datasets_EEG)...
        
        return written_EDF_file_paths
    



@define(slots=False)
class EntireDayMergedData:
    """ Manages data merged for an entire day
    
    from phoofflineeeganalysis.analysis.SavedSessionsProcessor import EntireDayMergedData
    
    """
    datasets: List[mne.io.Raw] = field(default=None)
    

    @classmethod
    def concatenate_with_gaps(cls, datasets: list[mne.io.Raw]) -> mne.io.Raw:
        """ #TODO 2025-09-09 22:09: - [ ] IMPORTATANT - the default MNE merge does not respect time at all
        """
        raws = []
        annotations = []
        total_duration = 0.0

        # Use the first dataset's orig_time as reference
        base_orig_time = datasets[0].annotations.orig_time

        for i, raw in enumerate(datasets):
            this_raw = deepcopy(raw)
            # Align annotation origins to the base_orig_time
            if this_raw.annotations is not None:
                this_raw.set_annotations(this_raw.annotations.copy())
                this_raw.annotations._orig_time = base_orig_time

            if i > 0:
                onset = total_duration
                ann = mne.Annotations(onset=[onset], duration=[0], description=["BAD_DISCONTINUITY"], orig_time=base_orig_time)
                annotations.append(ann)

            total_duration += this_raw.times[-1] + 1 / this_raw.info['sfreq']
            raws.append(this_raw)

        merged = mne.concatenate_raws(raws, preload=True)

        if annotations:
            combined = merged.annotations
            for ann in annotations:
                ann._orig_time = base_orig_time
                combined += ann
            merged.set_annotations(combined)

        return merged
    

    # @classmethod
    # def concatenate_datasets(cls, datasets: List[mne.io.Raw]) -> mne.io.Raw:
    #     """ Concatenates a list of mne.io.Raw datasets into a single Raw dataset.

    #     Args:
    #         datasets (List[mne.io.Raw]): List of Raw datasets to concatenate.

    #     Returns:
    #         mne.io.Raw: Concatenated Raw dataset.
    #     """
    #     if not datasets:
    #         raise ValueError("The datasets list is empty.")

    #     assert len(datasets) > 0, "The datasets list must contain at least one Raw object."

    #     concatenated_raw = deepcopy(datasets[0])
    #     # concatenated_raw = datasets[0]
    #     for raw in datasets[1:]:
    #         a_ds = deepcopy(raw)
    #         # concatenated_raw.append(deepcopy(raw))
    #         concatenated_raw.append(a_ds)

    #     return concatenated_raw
    

    @classmethod
    def find_and_merge_for_day_date(cls, sso: SavedSessionsProcessor, search_day_date: datetime,
                                    edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF"),
                                    save_edf: bool=False, save_fif: bool=False) -> mne.io.Raw:
        """ Finds all EEG datasets in the SavedSessionsProcessor for the specified date and merges them into a single Raw dataset.

        Args:
            sso (SavedSessionsProcessor): The SavedSessionsProcessor instance containing the datasets.
            search_day_date (datetime): The date for which to find and merge datasets.

        Returns:
            mne.io.Raw: Merged Raw dataset for the specified date.
        """
        from phoofflineeeganalysis.analysis.MNE_helpers import up_convert_raw_objects, up_convert_raw_obj

        if "EEG" not in sso.modalities:
            raise ValueError("The SavedSessionsProcessor does not contain any EEG modality data.")

        eeg_modality = sso.modalities["EEG"]
        today_only_eeg_modality = eeg_modality.filtered_by_day_date(search_day_date=search_day_date)

        if not today_only_eeg_modality.datasets:
            raise ValueError(f"No EEG datasets found for the date {search_day_date.date()}.")

        today_only_eeg_modality.datasets = up_convert_raw_objects(today_only_eeg_modality.datasets)
        ## Flatten the EEG sessions into a single dataset for the entire day
        # concatenated_raw = cls.concatenate_datasets(today_only_eeg_modality.datasets)
        concatenated_raw = cls.concatenate_with_gaps(today_only_eeg_modality.datasets)
        concatenated_raw = up_convert_raw_obj(concatenated_raw)

        ## convert to day-specific version:
        if save_fif:
            ## Save out the concatenated raw to a specific folder:
            day_grouped_processed_output_parent_path: Path = sso.eeg_analyzed_parent_export_path.joinpath('dayProcessed').resolve()
            day_grouped_processed_output_parent_path.mkdir(parents=True, exist_ok=True)

            ## INPUTS: search_day_date
            curr_day_grouped_output_folder: Path = day_grouped_processed_output_parent_path.joinpath(search_day_date.strftime("%Y-%m-%d")).resolve()
            curr_day_grouped_output_folder.mkdir(parents=True, exist_ok=True)
            print(f'curr_day_grouped_output_folder: "{curr_day_grouped_output_folder.as_posix()}"')            

            a_path = Path(concatenated_raw.filenames[0]).resolve()
            name_parts = a_path.name.split('-', maxsplit=4) # ['20250908', '121104', 'Epoc X', 'raw.fif']
            name_parts[1] = '000000'  # Set time part to '000000'
            new_name: str = '-'.join(name_parts)
            new_path: Path = curr_day_grouped_output_folder.joinpath(new_name).resolve()


            # TODO 2025-09-09 22:03: - [ ] IMPORTANT:
            # If Raw is a concatenation of several raw files, **be warned** that only the measurement information from the first raw file is stored. This likely means that certain operations with external tools may not work properly on a saved concatenated file (e.g., probably some or all forms of SSS). It is recommended not to concatenate and then save raw files for this reason.
            # Samples annotated BAD_ACQ_SKIP are not stored in order to optimize memory. Whatever values, they will be loaded as 0s when reading file.        
            concatenated_raw.save(new_path.as_posix(), overwrite=True)
        else:
            print(f'save_fif is False so skipping save.')
            
        ## Save EDF:

        ## INPUTS: raw_eeg
        if save_edf:
            if edf_export_parent_path is None:
                edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF").resolve()
                
            edf_export_parent_path.mkdir(exist_ok=True)

            ## Get paths for current raw:
            curr_file_edf_name: str = new_path.with_suffix('.edf').name
            curr_file_edf_path: Path = edf_export_parent_path.joinpath(curr_file_edf_name).resolve()
            # EEGData.save_mne_raw_to_edf(concatenated_raw, curr_file_edf_path)
            curr_file_edf_path = concatenated_raw.save_to_edf(output_path=curr_file_edf_path)
        else:
            print(f'save_edf is False so skipping save.')

        return concatenated_raw
    


def unwrap_single_element_listlike_if_needed(a_list):
    try:
        if len(a_list) == 1:
            return a_list[0]
        else:
            return a_list
    except (TypeError, AttributeError) as e:
        return a_list ## return the original
    except Exception as e:
        raise e



@pd.api.extensions.register_dataframe_accessor("xdf_streams")
class XDFDataStreamAccessor(object):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals 
    
    from phoofflineeeganalysis.analysis.SavedSessionsProcessor import XDFDataStreamAccessor
        
    xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
    
    """

    dt_col_names = ['recording_datetime', 'recording_day_date']
    timestamp_column_names = ['created_at', 'first_timestamp', 'last_timestamp']
    timestamp_dt_column_names = ['created_at_dt', 'first_timestamp_dt', 'last_timestamp_dt']
    timestamp_rel_column_names = ['created_at_rel', 'first_timestamp_rel', 'last_timestamp_rel']

    # _required_column_names = ['start', 'stop', 'label', 'duration']


    def __init__(self, pandas_obj):      
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj
        # self._obj = self._obj.sort_values(by=["start"]) # sorts all values in ascending order
        # Optional: If the 'label' column of the dataframe is empty, should populate it with the index (after sorting) as a string.
        # # self._obj['label'] = self._obj.index
        # self._obj["label"] = self._obj["label"].astype("str")
        # # Optional: Add 'duration' column:
        # self._obj["duration"] = self._obj["stop"] - self._obj["start"]


    @classmethod
    def init_from_results(cls, _out_xdf_stream_infos_df: pd.DataFrame, active_only_out_eeg_raws: List):
        num_sessions: int = len(active_only_out_eeg_raws)

        xdf_stream_infos_df: pd.DataFrame = deepcopy(_out_xdf_stream_infos_df)
        xdf_stream_infos_df['xdf_dataset_idx'] = -1
        xdf_stream_infos_df['recording_datetime'] = datetime.now()
        xdf_stream_infos_df['recording_day_date'] = datetime.now()
                

        for an_xdf_dataset_idx in np.arange(num_sessions):
            a_raw = active_only_out_eeg_raws[an_xdf_dataset_idx]
            a_meas_date = a_raw.info.get('meas_date')
            a_meas_day_date = a_meas_date.replace(hour=0, minute=0, second=0, microsecond=0)
            xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_datetime'] = a_meas_date
            xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_day_date'] = a_meas_day_date
            xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'xdf_dataset_idx'] = an_xdf_dataset_idx
            
            
        # end for an_xdf_dat... 
        xdf_stream_infos_df[cls.dt_col_names] = xdf_stream_infos_df[cls.dt_col_names].convert_dtypes()
        # xdf_stream_infos_df['created_at_rel'] = ((xdf_stream_infos_df['created_at_dt'] - xdf_stream_infos_df['recording_day_date']) / pd.Timedelta(hours=24.0))
        # xdf_stream_infos_df['first_timestamp']
        # xdf_stream_infos_df['duration_sec'] = [pd.Timedelta(seconds=v) for v in (xdf_stream_infos_df['n_samples'].astype(float) * (1.0/xdf_stream_infos_df['fs'].astype(float)))]
        xdf_stream_infos_df['duration_sec'] = [pd.Timedelta(seconds=v) if np.isfinite(v) else pd.NaT for v in (xdf_stream_infos_df['n_samples'].astype(float) * (1.0/xdf_stream_infos_df['fs'].astype(float)))]
        
        for a_ts_col_name, a_ts_dt_col_name, a_ts_rel_col_name in zip(cls.timestamp_column_names, cls.timestamp_dt_column_names, cls.timestamp_rel_column_names):
            try:
                # a_ts_dt_col_name: str = f'{a_ts_col_name}_dt'
                # xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + [pd.Timestamp(v) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                # xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + [pd.Timedelta(seconds=float(v)) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                # xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                # xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) if np.isfinite(v) else 0.0 for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) if np.isfinite(v) else pd.NaT for v in xdf_stream_infos_df[a_ts_col_name].to_numpy().astype(float)]
                xdf_stream_infos_df[a_ts_rel_col_name] = (xdf_stream_infos_df[a_ts_dt_col_name] / pd.Timedelta(hours=24.0))
                xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + xdf_stream_infos_df[a_ts_dt_col_name]

            except (ValueError, AttributeError) as e:
                print(f'failed to add column "{a_ts_dt_col_name}" due to error: {e}. Skipping col.')
                raise
            except Exception as e:
                raise

        ## try to add the updated duration column
        try:
            active_duration_col_name: str = 'duration_sec'
            if active_duration_col_name in xdf_stream_infos_df.columns:
                active_duration_col_name = 'duration_sec_check'
            if ('last_timestamp_dt' in xdf_stream_infos_df.columns) and ('first_timestamp_dt' in xdf_stream_infos_df.columns):            
                xdf_stream_infos_df[active_duration_col_name] = xdf_stream_infos_df['last_timestamp_dt'] - xdf_stream_infos_df['first_timestamp_dt']
                
            assert active_duration_col_name in xdf_stream_infos_df.columns, f"active_duration_col_name: '{active_duration_col_name}' still missing from xdf_stream_infos_df.columns: {list(xdf_stream_infos_df.columns)}"
            xdf_stream_infos_df['duration_rel'] = (xdf_stream_infos_df[active_duration_col_name] / pd.Timedelta(hours=24.0))


        except (ValueError, AttributeError) as e:
            print(f'failed to add column "{a_ts_dt_col_name}" due to error: {e}. Skipping col.')
            raise
        except Exception as e:
            raise
        
        return xdf_stream_infos_df
    

    # @classmethod
    # def adding_needed_columns(cls, obj):


    #     xdf_stream_infos_df: pd.DataFrame = deepcopy(_out_xdf_stream_infos_df)
    #     xdf_stream_infos_df['recording_datetime'] = datetime.now()
    #     xdf_stream_infos_df['recording_day_date'] = datetime.now()


    #     for an_xdf_dataset_idx in np.arange(num_sessions):
    #         a_raw = active_only_out_eeg_raws[an_xdf_dataset_idx]
    #         a_meas_date = a_raw.info.get('meas_date')
    #         a_meas_day_date = a_meas_date.replace(hour=0, minute=0, second=0, microsecond=0)
    #         xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_datetime'] = a_meas_date
    #         xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_day_date'] = a_meas_day_date
    #         # a_result = results[an_xdf_dataset_idx]
    #         # a_stream_info = deepcopy(xdf_stream_infos_df).loc[an_xdf_dataset_idx]    
    #         # # print(f'i: {i}, a_meas_date: {a_meas_date}, a_stream_info: {a_stream_info}\n\n')
    #         # print(f'i: {an_xdf_dataset_idx}, a_meas_date: {a_meas_date}')
    #         # a_df = a_raw.annotations.to_data_frame(time_format='datetime')
    #         # a_df = a_df[a_df['description'] != 'BAD_motion']
    #         # a_df['xdf_dataset_idx'] = an_xdf_dataset_idx
    #         # flat_annotations.append(a_df)
    #     # end for an_xdf_dat... 
    #     xdf_stream_infos_df[dt_col_names] = xdf_stream_infos_df[dt_col_names].convert_dtypes()
    #     xdf_stream_infos_df



    @classmethod
    def _validate(cls, obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming


    @property
    def extra_data_column_names(self):
        """Any additional columns in the dataframe beyond those that exist by default. """
        return list(set(self._obj.columns) - set(self._required_column_names))

    @property
    def extra_data_dataframe(self) -> pd.DataFrame:
        """The subset of the dataframe containing additional information in its columns beyond that what is required. """
        return self._obj[self.extra_data_column_names]

    # def as_array(self) -> NDArray:
    #     return self._obj[["start", "stop"]].to_numpy()


    def adding_or_updating_metadata(self, **metadata_update_kwargs) -> pd.DataFrame:
        """ updates the dataframe's `df.attrs` dictionary metadata, building it as a new dict if it doesn't yet exist

        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            maze_epochs_df = maze_epochs_df.epochs.adding_or_updating_metadata(train_test_period='train')
            maze_epochs_df

        """
        ## Add the metadata:
        if self._obj.attrs is None:
            self._obj.attrs = {} # create a new metadata dict on the dataframe
        self._obj.attrs.update(**metadata_update_kwargs)
        return self._obj


# def readable_dt_str(a_dt: datetime) -> str:
#     """ returns the datetime in a readible string format """
#     return str(a_dt.astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %I:%M:%S %p"))

def readable_dt_str(a_dt: datetime, tz: pytz.timezone = pytz.timezone("US/Eastern")) -> str:
    """ returns the datetime in a readible string format """
    return str(a_dt.astimezone(tz).strftime("%Y-%m-%d %I:%M:%S %p"))

def from_readable_dt_str(a_dt_str: str, tz: pytz.timezone = pytz.timezone("US/Eastern")) -> datetime:
    """ Inverse of `readable_dt_str(...)` """
    return datetime.strptime(a_dt_str, "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=tz)



@define(slots=False)
class LabRecorderXDF:
    """ Loads a `.xdf` file saved by LabRecorder which may contain one or more LSL Streams of differing types
    
    from phoofflineeeganalysis.analysis.SavedSessionsProcessor import LabRecorderXDF, unwrap_single_element_listlike_if_needed
    
    """
    lab_recorder_to_mne_to_type_dict = {'EEG':'eeg', 'ACC':'eeg', 'GYRO':'eeg', 'RAW': 'eeg'} # 'RAW' for eeg quality
    stream_name_to_modality_dict = {'Epoc X': DataModalityType.EEG, 'Epoc X Motion':DataModalityType.MOTION, 'Epoc X eQuality':None, 'TextLogger': DataModalityType.PHO_LOG_TO_LSL, 'EventBoard': DataModalityType.PHO_LOG_TO_LSL}

    datasets: List[mne.io.Raw] = field(default=None)
    
    @classmethod
    def init_from_lab_recorder_xdf_file(cls, a_xdf_file: Path):
        """

            Conclusions: `stream_clock_times` is not really needed if auto-sync is working.


            =========================================
            With `synchronize_clocks=True`:
                trying to process XDF file 0/1: "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-18T192330.926Z_eeg.xdf"...
                file_datetime: 2025-10-18 03:23:30 PM
                ======== STREAM "TextLogger":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                    stream_approx_dur_sec: 19.940502
                    stream_timestamps: [310118.9797208478, 310132.17171209387, 310138.9202364168]
                    stream_clock_times: [310117.99792570004, 310122.99849055, 310127.99873414997, 310132.99948935, 310137.99964735, 310143.0005398, 310148.00089784997, 310153.00175355]
                    post-zeroed stream_timestamps: [0.0, 13.191991246072575, 19.940515569003765]
                    post-zeroed stream_clock_times: [0.0, 5.000564849935472, 10.000808449927717, 15.001563649973832, 20.00172164995456, 25.00261409993982, 30.00297214993043, 35.0038278499851]
                ======== STREAM "EventBoard":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                    stream_approx_dur_sec: 0.0
                    stream_timestamps: [310141.53154871357]
                    stream_clock_times: [310117.99793914997, 310122.99851445, 310127.99872775003, 310132.99950185, 310137.99965555, 310143.00056144997, 310148.00089795, 310153.0017462]
                    post-zeroed stream_timestamps: [0.0]
                    post-zeroed stream_clock_times: [0.0, 5.0005753000150435, 10.000788600067608, 15.001562700024806, 20.001716400031, 25.00262230000226, 30.002958800061606, 35.00380705005955]
                ======== STREAM "Epoc X Motion":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    stream_approx_dur_sec: 39.980819
                    stream_timestamps: [310112.3528809373, 310112.38261473324, 310112.4146451288, 310112.44568922446, 310112.4766869202,..., ]
                    stream_clock_times: [204.484430350014, 209.48498810001183, 214.4852262000204, 219.48597295000218, 224.4861887000734, 229.48706944996957, 234.48741724999854, 239.4882807499962]
                    post-zeroed stream_timestamps: [0.0, 0.02973379596369341, 0.061764191545080394, 0.09280828718328848, 0.12380598293384537, 0.15590557851828635, 0.1929814734030515, 0.21782896999502555, 0.2487867656745948, ..., ]
                    post-zeroed stream_clock_times: [0.0, 5.000557749997824, 10.000795850006398, 15.00154259998817, 20.00175835005939, 25.00263909995556, 30.002986899984535, 35.003850399982184]
                ======== STREAM "Epoc X":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    stream_approx_dur_sec: 39.989998
                    stream_timestamps: [310112.34278103936, 310112.3521599422, 310112.3596535444, 310112.36563644616, 310112.3766195494, 310112.3816435509, 310112.3937489544, 310112.3987795559, 310112.4097387591, 310112.4137093603, ..., ]
                    stream_clock_times: [204.4844580499921, 209.4850055000279, 214.48526310001034, 219.48599920002744, 224.48616700002458, 229.48707915004343, 234.48742110002786, 239.4882879499928]
                    post-zeroed stream_timestamps: [0.0, 0.009378902846947312, 0.01687250501709059, 0.022855406801681966, 0.03383851004764438, 0.03886251151561737, 0.05096791504183784, 0.05599851656006649, 0.06695771974045783, ..., ]
                    post-zeroed stream_clock_times: [0.0, 5.0005474500358105, 10.000805050018243, 15.001541150035337, 20.00170895003248, 25.00262110005133, 30.00296305003576, 35.00382990000071]

                =========================================
                With `synchronize_clocks=False`:
                    limiting to included_xdf_file_names: ['LabRecorder_Apogee_2025-10-18T192330.926Z_eeg.xdf']...
                    limited to 1/49 files
                    trying to process XDF file 0/1: "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-18T192330.926Z_eeg.xdf"...
                    file_datetime: 2025-10-18 03:23:30 PM
                    ======== STREAM "TextLogger":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                        stream_approx_dur_sec: 19.940502
                        stream_timestamps: [310118.9797418, 310132.1717244, 310138.9202443]
                        stream_clock_times: [310117.99792570004, 310122.99849055, 310127.99873414997, 310132.99948935, 310137.99964735, 310143.0005398, 310148.00089784997, 310153.00175355]
                        post-zeroed stream_timestamps: [0.0, 13.191982600023039, 19.940502499986906]
                        post-zeroed stream_clock_times: [0.0, 5.000564849935472, 10.000808449927717, 15.001563649973832, 20.00172164995456, 25.00261409993982, 30.00297214993043, 35.0038278499851]
                    ======== STREAM "EventBoard":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                        stream_approx_dur_sec: 0.0
                        stream_timestamps: [310141.5315568]
                        stream_clock_times: [310117.99793914997, 310122.99851445, 310127.99872775003, 310132.99950185, 310137.99965555, 310143.00056144997, 310148.00089795, 310153.0017462]
                        post-zeroed stream_timestamps: [0.0]
                        post-zeroed stream_clock_times: [0.0, 5.0005753000150435, 10.000788600067608, 15.001562700024806, 20.001716400031, 25.00262230000226, 30.002958800061606, 35.00380705005955]
                    ======== STREAM "Epoc X Motion":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        stream_approx_dur_sec: 39.980819
                        stream_timestamps: [198.8393982, 198.869132, 198.9011624, 198.9322065, 198.9632042, 198.9953038, 199.0323797, 199.0572272, 199.088185, 199.1191771, 199.1532168, 199.1831693, 199.2131749, 199.250184, 199.2752947, ..., ]
                        stream_clock_times: [204.484430350014, 209.48498810001183, 214.4852262000204, 219.48597295000218, 224.4861887000734, 229.48706944996957, 234.48741724999854, 239.4882807499962]
                        post-zeroed stream_timestamps: [0.0, 0.029733800000002475, 0.06176419999999894, 0.09280830000000151, 0.12380600000000186, 0.15590559999998277, 0.19298150000000192, 0.21782899999999472, 0.24878680000000486, 0.2797788999999966, ..., ]
                        post-zeroed stream_clock_times: [0.0, 5.000557749997824, 10.000795850006398, 15.00154259998817, 20.00175835005939, 25.00263909995556, 30.002986899984535, 35.003850399982184]
                    ======== STREAM "Epoc X":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        stream_approx_dur_sec: 39.989998
                        stream_timestamps: [198.829322, 198.8387009, 198.8461945, 198.8521774, 198.8631605, 198.8681845, 198.8802899, 198.8853205, 198.8962797, 198.9002503, 198.9123016, 198.9162398, 198.922399, 198.9312574, 198.9392109, 198.9472146, 198.9552784, ..., ]
                        stream_clock_times: [204.4844580499921, 209.4850055000279, 214.48526310001034, 219.48599920002744, 224.48616700002458, 229.48707915004343, 234.48742110002786, 239.4882879499928]
                        post-zeroed stream_timestamps: [0.0, 0.009378900000001522, 0.016872500000005175, 0.022855399999997417, 0.03383850000000166, 0.03886250000002178, 0.05096790000001761, 0.05599850000001538, 0.06695770000001744, 0.07092830000001982, 0.08297960000001581, ..., ]
                        post-zeroed stream_clock_times: [0.0, 5.0005474500358105, 10.000805050018243, 15.001541150035337, 20.00170895003248, 25.00262110005133, 30.00296305003576, 35.00382990000071]
                    n_unique_xdf_datasets: 1
        """
        from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers, RawArrayExtended, RawExtended, up_convert_raw_obj, up_convert_raw_objects
        from phoofflineeeganalysis.analysis.motion_data import MotionData

        # Load .xdf
        # streams, header = pyxdf.load_xdf(a_xdf_file)
        # streams, header = pyxdf.load_xdf(a_xdf_file, synchronize_clocks=False, handle_clock_resets=False, dejitter_timestamps=False, verbose=True) ## disabled sync since it wasn't working anyway
        streams, header = pyxdf.load_xdf(a_xdf_file, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False, verbose=True) ## disabled sync since it wasn't working anyway

        file_datetime: datetime = datetime.strptime(header['info']['datetime'][0], "%Y-%m-%dT%H:%M:%S%z") # '2025-09-11T17:04:20-0400' -> datetime.datetime(2025, 9, 11, 17, 4, 20, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=72000)))           
        file_datetime = file_datetime.astimezone(timezone.utc)
             
        print(f'file_datetime: {readable_dt_str(file_datetime)}')

        ## claims that ['time_stamps'] are pre-synchronized across streams

        num_streams: int = len(streams)
        
        stream_infos = []
        raws = []
        raws_dict = {}

        streams_timestamp_dfs = {}

        all_annotations = []

        for stream in streams:
            name: str = stream['info']['name'][0]
            a_modality: DataModalityType = cls.stream_name_to_modality_dict.get(name, None)
            if a_modality is not None:
                a_modality = a_modality.value
            if a_modality not in raws_dict:
                raws_dict[a_modality] = []

            print(f'======== STREAM "{name}":')
            
            fs = float(stream['info']['nominal_srate'][0])
            stream_info_dict: Dict = {'name': name, 'fs': fs}

            sample_count: int = stream['footer']['info']['sample_count'][0]

            if len(stream['time_series']) == 0:
                print(f'\tWARN: skipping empty stream: "{name}"')
                continue ## skip this stream
            else:
                n_samples, n_channels = np.shape(stream['time_series'])
                stream_info_dict.update(**{'n_samples': n_samples, 'n_channels': n_channels})
                ## stream info keys:
                for a_key in ('type', 'stream_id', 'effective_srate', 'hostname', 'source_id', 'channel_count', 'channel_format', 'type', 'created_at', 'source_id', 'version', 'uid'):
                    a_value = stream['info'].get(a_key, None)
                    a_value = unwrap_single_element_listlike_if_needed(a_value)
                    if a_value is not None:
                        stream_info_dict[a_key] = a_value

                ## stream footer:
                for a_key in ('first_timestamp', 'last_timestamp', 'sample_count'):
                    a_value = stream.get('footer', {}).get('info', {}).get(a_key, None)
                    a_value = unwrap_single_element_listlike_if_needed(a_value)
                    if a_value is not None:
                        stream_info_dict[a_key] = float(a_value)

                ## Update the timestamp keys to float values, and the create a datetime column by adding them to the `file_datetime`
                timestamp_keys = ('created_at', 'first_timestamp', 'last_timestamp')
                for a_key in timestamp_keys:
                    if stream_info_dict.get(a_key, None) is not None:
                        a_ts_value: float = float(stream_info_dict[a_key]) # ['169993.1081304000']
                        a_ts_value_dt: datetime = file_datetime + pd.Timedelta(nanoseconds=a_ts_value)
                        a_dt_key: str = f'{a_key}_dt'
                        stream_info_dict[a_dt_key] = a_ts_value_dt
                        print(f'\t{a_dt_key}: {readable_dt_str(a_ts_value_dt)}')
                        

                ## try to get the special marker timestamp helpers:
                desc_info_dict = dict(stream['info'].get('desc', [{}])[0])
                # assert 'recording_start_lsl_local_offset_seconds' in desc_info_dict
                assert len(desc_info_dict) > 0
                custom_timestamp_keys = {'recording_start_lsl_local_offset_seconds': (lambda v: float(v)), 'recording_start_datetime': (lambda v: from_readable_dt_str(v))}
                for a_key, a_value_type_convert_fn in custom_timestamp_keys.items():
                    ## NOTE IMPORTANT: this operates on `desc_info_dict` dict, not the same `stream_info_dict` as above
                    if desc_info_dict.get(a_key, None) is not None:
                        a_ts_value = a_value_type_convert_fn(unwrap_single_element_listlike_if_needed(desc_info_dict[a_key])) # ['169993.1081304000']
                        # a_ts_value_dt: datetime = file_datetime + pd.Timedelta(nanoseconds=a_ts_value)
                        stream_info_dict[a_key] = a_ts_value ## In-contrast to what we get the data from, we SET the data to `stream_info_dict` just as above (flattening)
                        print(f'\t FOUND CUSTOM TIMESTAMP SYNC KEY: "{a_key}": {a_ts_value}')




                ############ pd.TimeDelta unit: `nanoseconds`
                # file_datetime: 2025-10-17 05:51:12 PM
                # WARN: skipping empty stream: "EventBoard"
                #     created_at_dt: 2025-10-17 05:51:12 PM
                #     first_timestamp_dt: 2025-10-17 05:51:12 PM
                #     last_timestamp_dt: 2025-10-17 05:51:12 PM

                ############ pd.TimeDelta unit: `milliseconds`
                # file_datetime: 2025-10-17 05:51:12 PM
                # WARN: skipping empty stream: "EventBoard"
                # 	created_at_dt: 2025-10-17 05:55:04 PM
                # 	first_timestamp_dt: 2025-10-17 05:55:04 PM
                # 	last_timestamp_dt: 2025-10-17 05:55:09 PM
                # 	best_found_unit: "ms"

                ############ pd.TimeDelta unit: `seconds`
                # file_datetime: 2025-10-17 05:51:12 PM
                # WARN: skipping empty stream: "EventBoard"
                #     created_at_dt: 2025-10-20 10:26:18 AM
                #     first_timestamp_dt: 2025-10-20 10:27:33 AM
                #     last_timestamp_dt: 2025-10-20 11:43:06 AM
                    

                # if stream_info_dict.get('created_at', None) is not None:
                #     stream_created_at: float = float(stream_info_dict['created_at']) # ['169993.1081304000']
                #     stream_created_at_dt: datetime = file_datetime + pd.Timedelta(nanoseconds=stream_created_at)
                # else:
                #     stream_created_at_dt: datetime = file_datetime
                
                # stream_info_dict['stream_created_at_dt'] = stream_created_at_dt
                # print(f'stream_created_at_dt: {stream_created_at_dt.astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %I:%M:%S %p")}')


                ## Add stream info dict to the stream_infos list:
                stream_infos.append(stream_info_dict)
                
                ## Process Data:
                # stream_info_dict

                stream_first_timestamp: float = float(stream['footer']['info']['first_timestamp'][0]) # 29605.4462984
                stream_last_timestamp: float = float(stream['footer']['info']['last_timestamp'][0]) # 30373.1166288

                stream_first_timestamp = pd.Timedelta(seconds=stream_first_timestamp)
                stream_last_timestamp = pd.Timedelta(seconds=stream_last_timestamp)

                # stream_num_samples: int = int(stream['footer']['info']['sample_count'][0])
                stream_approx_dur_sec: float = (stream_last_timestamp - stream_first_timestamp).total_seconds()
                print(f'\tstream_approx_dur_sec: {stream_approx_dur_sec}')
                # best_found_unit: str = MNEHelpers.determine_best_timedelta_unit_for_annotations(unknown_unit_timestamps=logger_timestamps, stream_approx_dur_sec=stream_approx_dur_sec)
                # best_found_unit: str = 'ns' ## always nanoseconds
                best_found_unit: str = 'ms' ## always nanoseconds
                # print(f'\tbest_found_unit: "{best_found_unit}"')
                

                stream_timestamps = deepcopy(np.array(stream['time_stamps']))
                stream_clock_times = deepcopy(np.array(stream['clock_times']))

                print(f'\tstream_timestamps: {stream_timestamps.tolist()}')
                print(f'\tstream_clock_times: {stream_clock_times.tolist()}')

                zeroed_stream_timestamps = deepcopy(stream_timestamps)
                zeroed_stream_clock_times = deepcopy(stream_clock_times)

                if len(zeroed_stream_timestamps) > 0:
                    zeroed_stream_timestamps = zeroed_stream_timestamps - zeroed_stream_timestamps[0] ## subtract out the first timestamp
                if len(zeroed_stream_clock_times) > 0:
                    zeroed_stream_clock_times = zeroed_stream_clock_times - zeroed_stream_clock_times[0] ## subtract out the first timestamp
                
                zeroed_stream_timestamps_dt = np.array([pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## convert to timedelta (for no reason)
                stream_datetimes = np.array([stream_info_dict.get('recording_start_datetime', file_datetime) + pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## List[datetime]

                ## OUTPUTS: stream_datetimes

                ## post-zeroed:
                print(f'\tpost-zeroed stream_timestamps: {stream_timestamps.tolist()}')
                print(f'\tpost-zeroed stream_clock_times: {stream_clock_times.tolist()}')

                ## STREAM OUTPUTS: stream_timestamps, stream_clock_times, zeroed_stream_timestamps, zeroed_stream_clock_times, zeroed_stream_timestamps_dt, stream_datetimes
                # a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=zeroed_stream_timestamps, onset_dt=zeroed_stream_timestamps_dt, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                # all_annotations.append(a_raw_df)

                ## UPDATE: `streams_timestamp_dfs`
                streams_timestamp_dfs[name] = pd.DataFrame(dict(stream_timestamps=stream_timestamps,
                    zeroed_stream_timestamps=zeroed_stream_timestamps, zeroed_stream_timestamps_dt=zeroed_stream_timestamps_dt,
                    # stream_clock_times=stream_clock_times,  zeroed_stream_clock_times=zeroed_stream_clock_times,
                    stream_datetimes = stream_datetimes,
                ))


                if (fs == 0):  
                    # irregular event streams
                    ch_names = ['TextLogger_Markers']
                    ch_types = ['misc']
                    logger_strings = [unwrap_single_element_listlike_if_needed(v) for v in stream['time_series']]
                    assert len(stream_timestamps) == len(logger_strings), f"len(stream_timestamps): {len(stream_timestamps)} != len(logger_strings): {len(logger_strings)}"
                    # info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
                    # data = np.array(stream['time_series']).T
                    # raw = mne.io.RawArray(data, info)

                    ## check
                    assert ((stream_info_dict['created_at_dt'] - file_datetime).total_seconds() < (90.0 * 60.0)) # should be less than 10 seconds between the file start and the logging stream (usually...)
                    # stream_clock_times = [(file_datetime + pd.Timedelta(nanoseconds=v)) for v in stream_clock_times]
                    # stream_clock_times = [(file_datetime + pd.Timedelta(nanoseconds=v)) for v in stream_clock_times] # TO

                    # stream_timestamps = [(logger_clock_times[0] + pd.Timedelta(nanoseconds=v)) for v in stream_timestamps]

                    # pd. logger_timestamps
                    # stream_first_timestamp: float = float(stream['footer']['info']['first_timestamp'][0])
                    # stream_last_timestamp: float = float(stream['footer']['info']['last_timestamp'][0])
                    # # stream_num_samples: int = int(stream['footer']['info']['sample_count'][0])
                    # stream_approx_dur_sec: float = stream_last_timestamp - stream_first_timestamp


                    # [pd.to_timedelta(a_start_stop_diff, unit=a_unit).total_seconds() for a_unit in ('ns', 'us', 'ms', 's')] 
                    # pd.to_timedelta(a_start_stop_diff, unit='ns')

                    # If orig_time is None, the annotations are synced to the start of the data (0 seconds). Otherwise the annotations are synced to sample 0 and raw.first_samp is taken into account the same way as with events.
                    # meas_date = deepcopy(file_datetime) # deepcopy(a_ds.info['meas_date'])
                    # converted = pd.to_timedelta(logger_timestamps, unit=best_found_unit)
                    # converted = file_datetime + converted

                    # logger_timestamps = [(logger_clock_times[0] + pd.Timedelta(nanoseconds=v)) for v in logger_timestamps]

                    converted_dt = [(file_datetime + pd.to_timedelta(v, unit=best_found_unit)) for v in stream_timestamps]

                    # converted_dt = zeroed_stream_timestamps_dt # [(file_datetime + pd.to_timedelta(v, unit=best_found_unit)) for v in stream_timestamps]
                    # converted = [(v - file_datetime).total_seconds() for v in converted]

                    # a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=zeroed_stream_timestamps, onset_dt=zeroed_stream_timestamps_dt, converted_dt=converted_dt, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                    a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=stream_datetimes, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                    all_annotations.append(a_raw_df)

                    # converted = [(v - file_datetime).total_seconds() for v in converted_dt]
                    # converted = file_datetime + pd.to_timedelta(logger_timestamps, unit="ns") ## starts out in nanoseconds (ns) relative to `file_datetime`
                    # converted = file_datetime + pd.to_timedelta(logger_timestamps, unit=best_found_unit) ## starts out in specified unit relative to `file_datetime`
                    # converted = converted - file_datetime ## subtract out the `file_datetime` component
                    # converted = converted.total_seconds() ## use .total_seconds() to get the value in seconds
                    # raw = mne.Annotations(onset=logger_timestamps, duration=([0.0] * len(logger_timestamps)), description=logger_strings, orig_time=file_datetime.astimezone(timezone.utc))
                    # raw = mne.Annotations(onset=converted, duration=([0.0] * len(stream_timestamps)), description=logger_strings, orig_time=None) ## set orig_time=None

                    raw = mne.Annotations(onset=zeroed_stream_timestamps, duration=([0.0] * len(zeroed_stream_timestamps)), description=logger_strings, orig_time=None) ## set orig_time=None
                    
                    # A POSIX Timestamp, datetime or a tuple containing the timestamp as the first element and microseconds as the second element. Determines the starting time of annotation acquisition. If None (default), starting time is determined from beginning of raw data acquisition. In general, raw.info['meas_date'] (or None) can be used for syncing the annotations with raw data if their acquisition is started at the same time. If it is a string, it should conform to the ISO8601 format. More precisely to this '%%Y-%%m-%%d %%H:%%M:%%S.%%f' particular case of the ISO8601 format where the delimiter between date and time is ' '.
                    # raw = mne.Annotations(onset=converted, duration=([0.0] * len(logger_timestamps)), description=logger_strings, orig_time=file_datetime)
                    # raw = mne.Annotations(onset=pd.to_timedelta(logger_timestamps, unit="ns"), duration=([0.0] * len(logger_timestamps)), description=logger_strings, orig_time=file_datetime)     
                    
                    ## UPDATE `raws` and `raws_dict` with the new raw object:
                    raws.append(raw)
                    if a_modality is not None:
                        raws_dict[a_modality].append(raw)

                else:
                    ## fixed sampling rate streams:
                    _channels_dict = benedict(stream['info']['desc'][0]['channels'][0])
                    channels_df: pd.DataFrame = pd.DataFrame.from_records([{k:v[0] for k, v in ch_v.items()} for ch_v in _channels_dict.flatten()['channel']])
                    data = np.array(stream['time_series']).T
                    if stream_info_dict['type'] == 'EEG':
                        pass
                    # ch_names = [f"{name}_{i}" for i in range(data.shape[0])]
                    # ch_types = ["eeg"] * data.shape[0]  # adjust depending on stream type
                    ch_names = channels_df['label'].to_list()
                    ch_types = [cls.lab_recorder_to_mne_to_type_dict[v] for v in channels_df['type']]
                    
                    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
                    info = info.set_meas_date(file_datetime)
                    info['description'] = a_xdf_file.as_posix()
                    info['device_info'] = {'type':'USB', 'model':'EpocX', 'serial': '', 'site':'pho'} # #TODO 2025-09-22 08:51: - [ ] Add Hostname<USB> or Hostname<BLE>
                    
                    raw = mne.io.RawArray(data, info) ## also have , first_samp=0

                    ## UPDATE `raws` and `raws_dict` with the new raw object:
                    raws.append(raw)
                    if a_modality is not None:
                        raws_dict[a_modality].append(raw)
        ## END for stream in streams...

        stream_infos: pd.DataFrame = pd.DataFrame.from_records(stream_infos)
        stream_infos


        # - [ ] TODO 2025-10-18 Attempt to appropriately re-zero each stream's `'stream_timestamps'` (seconds since recording start conceptually) to the same zero so they can easily be concatenated). Currently assumes they all started at the same time with no offset (which wouldn't be true if I started the logger after the EEG stream, for example).
        ## streams_timestamp_dfs
        ## find earliest stream_timestamp across all streams:
        stream_earliest_timestamp_sec_dict = {k:np.nanmin(df['stream_timestamps']) for k, df in streams_timestamp_dfs.items()}
        absolute_earliest_ts_sec: float = np.nanmin([v for v in stream_earliest_timestamp_sec_dict.values()])

        earliest_stream_zeroed_stream_timestamps_dict = {}
        for k, df in streams_timestamp_dfs.items():
            earliest_stream_zeroed_stream_timestamps_dict[k] = df['stream_timestamps'] - absolute_earliest_ts_sec
        stream_earliest_timestamp_sec_dict = {k:np.nanmin(df['stream_timestamps']) }



        time_col_name: str = 'onset'
        ## set the annotations for the EEG-type modalities

        for an_eeg_ds in raws_dict.get(DataModalityType.EEG.value, []):
            an_eeg_ds = up_convert_raw_obj(an_eeg_ds)
            EEGData.set_montage(datasets_EEG=an_eeg_ds)
            
            # ==================================================================================================================================================================================================================================================================================== #
            # Adding `DataModalityType.PHO_LOG_TO_LSL` before `DataModalityType.MOTION` annotations works, while the opposite order seems to lose the MOTION annotations                                                                                                                           #
            # ==================================================================================================================================================================================================================================================================================== #
            an_all_annotations = deepcopy(all_annotations) #[]

            # for an_annotation_ds in raws_dict.get(DataModalityType.PHO_LOG_TO_LSL.value, []):
            #     num_annotations_to_add: int = len(an_annotation_ds)
            #     before_add_num_annotations: int = len(an_eeg_ds.annotations)
            #     # an_all_annotations.append(an_annotation_ds)
            #     # meas_date = deepcopy(an_eeg_ds.info.get('meas_date'))
            #     # MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=an_annotation_ds, align_to_Raw_meas_time=True)
            #     # MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=an_annotation_ds, align_to_Raw_meas_time=False)
            #     after_add_num_annotations: int = len(an_eeg_ds.annotations)

            #     actually_added_annotations: int = (after_add_num_annotations - before_add_num_annotations)
            #     if (actually_added_annotations < num_annotations_to_add):
            #         missing_annotations: int = num_annotations_to_add - actually_added_annotations
            #         print(f'failed to add {missing_annotations} annotations.\n\tnum_annotations_to_add: {num_annotations_to_add}, before_add_num_annotations: {before_add_num_annotations}, after_add_num_annotations: {after_add_num_annotations} ')
                    

            #     if (an_eeg_ds.annotations is None) or (len(an_eeg_ds.annotations) < 1):
            #         # an_eeg_ds.annotations = an_annotation_ds
            #         an_eeg_ds.set_annotations(an_annotation_ds)
            #     else:
            #         # a_raw: mne.io.Raw = mne.io.Raw(an_eeg_ds)
            #         an_eeg_ds.set_annotations(an_annotation_ds)
            #         # an_eeg_ds.set_annotation(an_annotation_ds)
            #     an_eeg_ds.annotations
            # ## END for an_annotation_ds in raws_d...
            
            # if not an_eeg_ds.debug_test_annotations_timestamps():
            #     raise

            # ==================================================================================================================================================================================================================================================================================== #
            # Add Motion Annotations                                                                                                                                                                                                                                                               #
            # ==================================================================================================================================================================================================================================================================================== #
            for an_motion_raw_ds in raws_dict.get(DataModalityType.MOTION.value, []):

                # motion_annots: mne.Annotations = MotionData.find_high_accel_periods(an_motion_raw_ds, should_set_bad_period_annotations=True)
                motion_annots: mne.Annotations = MotionData.find_high_accel_periods(an_motion_raw_ds, should_set_bad_period_annotations=False) # should_set_bad_period_annotations=False must be False so it doesn't overwrite existing annotations
                # an_all_annotations.append(motion_annots)
                motion_annots_df: pd.DataFrame = motion_annots.to_data_frame(time_format='datetime')
                
                motion_annots_df[time_col_name] = motion_annots_df[time_col_name].dt.tz_localize('UTC')
                an_all_annotations.append(motion_annots_df)

                # MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=motion_annots, align_to_Raw_meas_time=True)
            ## END for an_motion_raw_ds in raws_...


            # if not an_eeg_ds.debug_test_annotations_timestamps():
            #     raise

            ## TODO: handle
             
            # an_all_annotations_df = pd.concat([v.to_data_frame('datetime') for v in an_all_annotations])

            if len(an_all_annotations) > 0:
                an_all_annotations_df = pd.concat(an_all_annotations)
                # df = an_all_annotations_df
                # time_col_name: str = 'onset'
                # df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
                # if df[time_col_name].dt.tz is not None:
                #     df[time_col_name] = df[time_col_name].dt.tz_convert('UTC')
                # else:
                #     df[time_col_name] = df[time_col_name].dt.tz_localize('UTC')
                an_all_annotations_df = an_all_annotations_df.sort_values(by='onset', axis='index', na_position='first', ignore_index=True, ascending=True, inplace=False)
                # an_all_annotations_df['onset'] = (an_all_annotations_df['onset'].dt.tz_localize(tz='utc') - file_datetime).dt.total_seconds() 
                # an_all_annotations_df['onset'] = (an_all_annotations_df['onset'] - file_datetime).dt.total_seconds() 


                an_all_annotations_df['onset'] = [(v - file_datetime).total_seconds() for v in an_all_annotations_df['onset']] ## convert to non-timedelta float in units of seconds


                # an_all_annotations_df
                # [v.to_data_frame('ms') for v in an_all_annotations]
                final_annots = mne.Annotations(onset=an_all_annotations_df['onset'].to_numpy(), duration=an_all_annotations_df['duration'].to_numpy(), description=an_all_annotations_df['description'].to_numpy(), orig_time=None) ## set orig_time=None
                an_eeg_ds = MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=final_annots, align_to_Raw_meas_time=False)
                if not an_eeg_ds.debug_test_annotations_timestamps():
                    raise
            ## END if len(an_all_annotations) > 0...


        ## END for an_eeg_ds in raws_dict.get(DataModalityType.EEG.value, [])...
        


        return stream_infos, raws, raws_dict
        

    @classmethod
    def save_post_processed_to_fif(cls, raws_dict, a_xdf_file: Path, labRecorder_PostProcessed_path: Path, export_mat: bool=True):
        """ 

        eeg_raw, a_lab_recorder_filepath = LabRecorderXDF.save_post_processed_to_fif(
            raws_dict=raws_dict,
            a_xdf_file=a_xdf_file,
            labRecorder_PostProcessed_path=sso.eeg_analyzed_parent_export_path.joinpath(f'LabRecorder_PostProcessed'),
        )

        LabRecorder_Apogee_2025-09-18T15-18-39
        LabRecorder_2025-09-19T02-22-10.mat
        
                 
        
        """
        ## When done processing the entire LabRecorder.xdf, save only the EEG data (with all annotations and such added) to a new file
        eeg_raws = raws_dict[DataModalityType.EEG.value]
        eeg_raws = up_convert_raw_objects(eeg_raws)
        assert len(eeg_raws) == 1, f"len(eeg_raws): {len(eeg_raws)}, but only handle the single eeg file case."
        if len(eeg_raws) == 1:
            eeg_raw = eeg_raws[0]


        labRecorder_PostProcessed_path.mkdir(exist_ok=True)

        a_lab_recorder_filename: str = a_xdf_file.stem
        # a_lab_recorder_filename_parts = a_lab_recorder_filename.split('_')
        
        ## drop the last useless part like '_egg'
        a_clean_filename: str = a_lab_recorder_filename.removeprefix('LabRecorder_').removesuffix('_eeg')
        a_lab_recorder_filename_parts = a_clean_filename.split('_')
        final_output_filename_parts = []
        datetime_part = a_lab_recorder_filename_parts[-1] ## always true, but will be discarded
        if len(a_lab_recorder_filename_parts) == 1:
            ## no hostname
            pass
        elif len(a_lab_recorder_filename_parts) > 2:
            ## has hostname
            hostname_parts = '_'.join(a_lab_recorder_filename_parts[:-1])
            print(f'hostname_parts: {hostname_parts} will be discarded')
            # final_output_filename_parts.append(hostname_parts)

        ## replace with the eeg meas date
        meas_date = eeg_raw.info.get('meas_date')
        # a_lab_recorder_filename_parts[-2] = meas_date.strftime("%Y-%m-%dT%H-%M-%S")
        final_output_filename_parts.append(meas_date.strftime("%Y-%m-%dT%H-%M-%S"))
        
        a_lab_recorder_filename: str = '_'.join(final_output_filename_parts)
        # a_lab_recorder_filename: str = '_'.join(a_lab_recorder_filename_parts[:-1]) ## drop only the last part
        # a_lab_recorder_filename

        a_lab_recorder_filepath = labRecorder_PostProcessed_path.joinpath(a_lab_recorder_filename)
        # a_lab_recorder_filepath.with_suffix('.fif')

        a_lab_recorder_filepath = a_lab_recorder_filepath.with_suffix('.fif')
        print(f'saving finalized EEG data out to "{a_lab_recorder_filepath.as_posix()}"')
        eeg_raw.save(a_lab_recorder_filepath, overwrite=True)
        
        export_filepaths_dict = {'fif': a_lab_recorder_filepath}
        if export_mat:
            mat_export_folder = a_lab_recorder_filepath.parent.joinpath('mat')
            mat_export_folder.mkdir(exist_ok=True)
            mat_export_path = mat_export_folder.joinpath(a_lab_recorder_filename).with_suffix('.mat')
            export_filepaths_dict['mat'] = eeg_raw.save_to_fieldtrip_mat(mat_export_path)


        return eeg_raw, export_filepaths_dict
    

    @classmethod
    def load_and_process_all(cls, lab_recorder_output_path: Path, 
                                  labRecorder_PostProcessed_path: Optional[Path] = Path("E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/LabRecorder_PostProcessed").resolve(),
                                    should_write_final_merged_eeg_fif: bool = True,
                                    debug_print: bool = False,
                                    included_xdf_file_names=None,
                                    fail_on_exception: bool=False,
                                                          ):

        """ main load function for all XDF files exported by LabRecorder
        """
        from phoofflineeeganalysis.analysis.MNE_helpers import DatasetDatetimeBoundsRenderingMixin, RawArrayExtended, RawExtended, up_convert_raw_objects, up_convert_raw_obj
        from phoofflineeeganalysis.analysis.EEG_data import EEGData
                                       
        assert lab_recorder_output_path.exists()

        lab_recorder_xdf_files: List[Path] = list(lab_recorder_output_path.glob('*.xdf'))
        n_total_found_files: int = len(lab_recorder_xdf_files)
        if included_xdf_file_names is not None:
            print(f'limiting to included_xdf_file_names: {included_xdf_file_names}...')
            lab_recorder_xdf_files = [v for v in lab_recorder_xdf_files if v.name in included_xdf_file_names]
            n_filtered_found_files: int = len(lab_recorder_xdf_files)
            print(f'\tlimited to {n_filtered_found_files}/{n_total_found_files} files')

        if (labRecorder_PostProcessed_path is not None) and should_write_final_merged_eeg_fif:
            labRecorder_PostProcessed_path.mkdir(exist_ok=True)
        
        # a_xdf_file = lab_recorder_xdf_files[-3]
        # a_xdf_file = lab_recorder_xdf_files[-1]
        # a_xdf_file = Path(r"E:\Dropbox (Personal)\Databases\UnparsedData\LabRecorderStudies\sub-P001\LabRecorder_2025-09-18T031842.989Z_eeg.xdf").resolve()
        # a_xdf_file = Path(r"E:\Dropbox (Personal)\Databases\UnparsedData\LabRecorderStudies\sub-P001\LabRecorder_2025-09-18T121337.267Z_eeg.xdf").resolve()

        _out_eeg_raw = []
        _out_xdf_stream_infos_df = []

        for an_xdf_file_idx, a_xdf_file in enumerate(lab_recorder_xdf_files):
            print(f'trying to process XDF file {an_xdf_file_idx}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.as_posix()}"...')
            try:
                stream_infos, raws, raws_dict = cls.init_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file)
                eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])
                if len(eeg_raws) != 1:
                     raise ValueError(f'for file "{a_xdf_file.as_posix()}": len(eeg_raws): {len(eeg_raws)}, but only handle the single eeg file case.')
                else:
                    eeg_raw = eeg_raws[0]        

                stream_infos['lab_recorder_xdf_file_idx'] = an_xdf_file_idx
                stream_infos['xdf_dataset_idx'] = len(_out_xdf_stream_infos_df) ## the actual index of the good datsets
                stream_infos['xdf_filename'] = a_xdf_file.name ## just the name

                if should_write_final_merged_eeg_fif:
                    eeg_raw, a_lab_recorder_exports_filepaths_dict = cls.save_post_processed_to_fif(
                        raws_dict=raws_dict,
                        a_xdf_file=a_xdf_file,
                        labRecorder_PostProcessed_path=labRecorder_PostProcessed_path,
                    )
                    if a_lab_recorder_exports_filepaths_dict is not None:
                        for a_format, an_export_path in a_lab_recorder_exports_filepaths_dict.items():
                            stream_infos[f'proccessed_{a_format}_filename'] = an_export_path.name ## just the name

                eeg_raw = up_convert_raw_obj(eeg_raw)
                EEGData.set_montage(datasets_EEG=[eeg_raw])
                eeg_raw.debug_test_annotations_timestamps()
                _out_eeg_raw.append(eeg_raw)
                # stream_infos['xdf_dataset_idx'] = a_xdf_file.name ## just the name
                _out_xdf_stream_infos_df.append(stream_infos)
                
            except (ValueError, KeyError) as e:
                print(f'\t failed with error: {e}\n\tskipping file.')
                if fail_on_exception:
                    raise
                else:
                    continue
                
            except Exception as e:
                print(f'\t failed with error: {e}\n\tskipping file.')
                raise
                # continue
        ## END for an_xdf_file_idx, a_x...
        
        _out_xdf_stream_infos_df = pd.concat(_out_xdf_stream_infos_df)
        _out_xdf_stream_infos_df = _out_xdf_stream_infos_df.set_index('xdf_dataset_idx')
        
        _out_eeg_raw = up_convert_raw_objects(_out_eeg_raw)
        _out_eeg_raw.sort(key=lambda r: (r.raw_timerange()[0] is None, r.raw_timerange()[0]))
        
        EEGData.set_montage(datasets_EEG=_out_eeg_raw)
        

        # _out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw) # [_out_xdf_stream_infos_df['name'] == 'Epoc X']
        
        
        return _out_eeg_raw, _out_xdf_stream_infos_df, lab_recorder_xdf_files



    @classmethod
    def to_hdf(cls, active_only_out_eeg_raws, results, xdf_stream_infos_df: pd.DataFrame, file_path: Path, root_key: str='/', debug_print=True):
        """ 
        from phoofflineeeganalysis.PendingNotebookCode import batch_compute_all_eeg_datasets
                
        LabRecorderXDF.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

        from phoofflineeeganalysis.analysis.EEG_data import EEGComputations

        active_only_out_eeg_raws, results = batch_compute_all_eeg_datasets(eeg_raws=_out_eeg_raw, limit_num_items=150, max_workers = 4)
                
        # EEGComputations.to_hdf(a_result=results[0], file_path="")
        hdf5_out_path: Path = Path('E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/outputs').joinpath('2025-09-23_eegComputations.h5').resolve()
        hdf5_out_path

        for idx, (a_raw, a_raw_outputs) in enumerate(zip(active_only_out_eeg_raws, results)):
            # a_path: Path = Path(a_raw.filenames[0])
            # basename: str = a_path.stem
            # basename: str = a_raw.info.get('meas_date')
            src_file_path: Path = Path(a_raw.info.get('description')).resolve()
            basename: str = src_file_path.stem

            print(f'basename: {basename}')
            EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

            # EEGComputations.to_hdf(a_result=results[0], file_path="", root_key=f"/{basename}/")

            # for an_output_key, an_output_dict in a_raw_outputs.items():
            #     for an_output_subkey, an_output_value in an_output_dict.items():
            #         final_data_key: str = '/'.join([basename, an_output_key, an_output_subkey])
            #         print(f'\tfinal_data_key: "{final_data_key}"')
            #         # all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)

            # spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
            # fs = a_raw_outputs['spectogram']['fs']

            # for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
            #     all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)
            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)

            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)


        # E:\Dropbox (Personal)\Databases\AnalysisData\MNE_preprocessed\outputs\


        """
        import h5py
        from phoofflineeeganalysis.analysis.EEG_data import EEGComputations, EEGData
        from phoofflineeeganalysis.analysis.SavedSessionsProcessor import XDFDataStreamAccessor

        write_mode = 'a'
        if (not file_path.exists()):
            write_mode = 'w'

        num_sessions: int = len(active_only_out_eeg_raws)
        xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
        # xdf_stream_infos_df.to_hdf(file_path, key='/xdf_stream_infos_df', append=True) ## append=False to overwrite existing
        xdf_stream_infos_df.to_hdf(file_path, key='/xdf_stream_infos_df', append=True)

        flat_annotations = []

        for an_xdf_dataset_idx in np.arange(num_sessions):
            a_raw = active_only_out_eeg_raws[an_xdf_dataset_idx]
            a_meas_date = a_raw.info.get('meas_date')
            a_raw_key: str = a_meas_date.strftime("%Y-%m-%d/%H-%M-%S") # '2025-09-22/21-35-47'

            a_result = results[an_xdf_dataset_idx]
            with h5py.File(file_path, 'a') as f:
                EEGComputations.perform_write_to_hdf(a_result=a_result, f=f, root_key=f'/result/{a_raw_key}')

            # a_stream_info = deepcopy(xdf_stream_infos_df).loc[an_xdf_dataset_idx]    
            # print(f'i: {i}, a_meas_date: {a_meas_date}, a_stream_info: {a_stream_info}\n\n')
            # print(f'i: {an_xdf_dataset_idx}, a_meas_date: {a_meas_date}')
            # a_raw.to_data_frame(time_format='datetime').to_hdf(file_path, key=f'/raw/{a_raw_key}/df', append=True)
            a_raw.to_data_frame(time_format='datetime').to_hdf(file_path, key=f'/raw/{a_raw_key}', append=True)
            # EEGComputations.to_hdf(a_result=a_result, file_path=file_path, root_key=f'/result/{a_raw_key}')
            a_df = a_raw.annotations.to_data_frame(time_format='datetime')
            a_df = a_df[a_df['description'] != 'BAD_motion']
            # a_df['xdf_dataset_idx'] = an_xdf_dataset_idx
            flat_annotations.append(a_df)
                

        flat_annotations = pd.concat(flat_annotations, ignore_index=True)
        flat_annotations['onset_str'] = flat_annotations['onset'].dt.strftime("%Y-%m-%d_%I:%M:%S.%f %p")

        if flat_annotations is not None:
            flat_annotations.to_hdf(file_path, key='/flat_annotations_df', append=True)


        return file_path
    


