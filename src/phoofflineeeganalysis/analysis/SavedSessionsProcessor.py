import time
import re
from datetime import datetime, timezone
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
from datetime import datetime, timezone
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





@define(slots=False)
class LabRecorderXDF:
    """ 
    
    from phoofflineeeganalysis.analysis.SavedSessionsProcessor import LabRecorderXDF, unwrap_single_element_listlike_if_needed
    
    """
    lab_recorder_to_mne_to_type_dict = {'EEG':'eeg', 'ACC':'eeg', 'GYRO':'eeg', 'RAW': 'eeg'} # 'RAW' for eeg quality
    stream_name_to_modality_dict = {'Epoc X': DataModalityType.EEG, 'Epoc X Motion':DataModalityType.MOTION, 'Epoc X eQuality':None, 'TextLogger': DataModalityType.PHO_LOG_TO_LSL}

    datasets: List[mne.io.Raw] = field(default=None)
    
    @classmethod
    def init_from_lab_recorder_xdf_file(cls, a_xdf_file: Path):
        from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
        from phoofflineeeganalysis.analysis.motion_data import MotionData

        # Load .xdf
        streams, header = pyxdf.load_xdf(a_xdf_file)
        file_datetime = datetime.strptime(header['info']['datetime'][0], "%Y-%m-%dT%H:%M:%S%z") # '2025-09-11T17:04:20-0400' -> datetime.datetime(2025, 9, 11, 17, 4, 20, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=72000)))           
        file_datetime = file_datetime.astimezone(timezone.utc)
             
        num_streams: int = len(streams)
        
        stream_infos = []
        raws = []
        raws_dict = {}
        
        for stream in streams:
            name: str = stream['info']['name'][0]
            a_modality: DataModalityType = cls.stream_name_to_modality_dict.get(name, None)
            if a_modality is not None:
                a_modality = a_modality.value
            if a_modality not in raws_dict:
                raws_dict[a_modality] = []
            
            fs = float(stream['info']['nominal_srate'][0])
            stream_info_dict: Dict = {'name': name, 'fs': fs}

            n_samples, n_channels = np.shape(stream['time_series'])
            stream_info_dict.update(**{'n_samples': n_samples, 'n_channels': n_channels})
            ## stream info keys:
            for a_key in ('stream_id', 'effective_srate', 'hostname', 'source_id', 'channel_format', 'type', 'created_at'):
                a_value = stream['info'].get(a_key, None)
                a_value = unwrap_single_element_listlike_if_needed(a_value)
                if a_value is not None:
                    stream_info_dict[a_key] = a_value


            ## stream footer:
            for a_key in ('first_timestamp', 'last_timestamp', 'sample_count'):
                a_value = stream.get('footer', {}).get('info', {}).get(a_key, None)
                a_value = unwrap_single_element_listlike_if_needed(a_value)
                if a_value is not None:
                    stream_info_dict[a_key] = a_value

            stream_infos.append(stream_info_dict)
            
            ## Process Data:
            # stream_info_dict

            if fs == 0:  
                # continue # skip irregular event streams
                ch_names = ['TextLogger_Markers']
                ch_types = ['misc']
                logger_timestamps = stream['time_stamps']
                logger_strings = [unwrap_single_element_listlike_if_needed(v) for v in stream['time_series']]
                assert len(logger_timestamps) == len(logger_strings), f"len(logger_timestamps): {len(logger_timestamps)} != len(logger_strings): {len(logger_strings)}"
                # info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
                # data = np.array(stream['time_series']).T
                # raw = mne.io.RawArray(data, info)
                # pd. logger_timestamps
                converted = file_datetime + pd.to_timedelta(logger_timestamps, unit="ns") ## starts out in nanoseconds (ns) relative to `file_datetime`
                converted = converted - file_datetime ## subtract out the `file_datetime` component
                converted = converted.total_seconds() ## use .total_seconds() to get the value in seconds
                # raw = mne.Annotations(onset=logger_timestamps, duration=([0.0] * len(logger_timestamps)), description=logger_strings, orig_time=file_datetime.astimezone(timezone.utc))
                raw = mne.Annotations(onset=converted, duration=([0.0] * len(logger_timestamps)), description=logger_strings, orig_time=None) ## set orig_time=None
                # raw = mne.Annotations(onset=pd.to_timedelta(logger_timestamps, unit="ns"), duration=([0.0] * len(logger_timestamps)), description=logger_strings, orig_time=file_datetime)     
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
                
                raw = mne.io.RawArray(data, info)   
                raws.append(raw)
                if a_modality is not None:
                    raws_dict[a_modality].append(raw)

        stream_infos: pd.DataFrame = pd.DataFrame.from_records(stream_infos)

        ## set the annotations for the EEG-type modalities

        for an_eeg_ds in raws_dict.get(DataModalityType.EEG.value, []):
            EEGData.set_montage(datasets_EEG=an_eeg_ds)
            
            # ==================================================================================================================================================================================================================================================================================== #
            # Adding `DataModalityType.PHO_LOG_TO_LSL` before `DataModalityType.MOTION` annotations works, while the opposite order seems to lose the MOTION annotations                                                                                                                           #
            # ==================================================================================================================================================================================================================================================================================== #
            for an_annotation_ds in raws_dict.get(DataModalityType.PHO_LOG_TO_LSL.value, []):
                # meas_date = deepcopy(an_eeg_ds.info.get('meas_date'))
                # MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=an_annotation_ds, align_to_Raw_meas_time=True)
                MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=an_annotation_ds, align_to_Raw_meas_time=False)        
                # if (an_eeg_ds.annotations is None) or (len(an_eeg_ds.annotations) < 1):
                #     # an_eeg_ds.annotations = an_annotation_ds
                #     an_eeg_ds.set_annotations(an_annotation_ds)
                # else:
                #     # a_raw: mne.io.Raw = mne.io.Raw(an_eeg_ds)
                #     an_eeg_ds.set_annotations(an_annotation_ds)
                #     # an_eeg_ds.set_annotation(an_annotation_ds)
                # an_eeg_ds.annotations

            # ==================================================================================================================================================================================================================================================================================== #
            # Add Motion Annotations                                                                                                                                                                                                                                                               #
            # ==================================================================================================================================================================================================================================================================================== #
            for an_motion_raw_ds in raws_dict.get(DataModalityType.MOTION.value, []):

                motion_annots: mne.Annotations = MotionData.find_high_accel_periods(an_motion_raw_ds, should_set_bad_period_annotations=True)

                MNEHelpers.merge_annotations(raw=an_eeg_ds, new_annots=motion_annots, align_to_Raw_meas_time=True)



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
                                    should_write_final_merged_eeg_fif: bool = True
                                                          ):

        """ main load function for all XDF files exported by LabRecorder
        """
        from phoofflineeeganalysis.analysis.MNE_helpers import DatasetDatetimeBoundsRenderingMixin, RawArrayExtended, RawExtended, up_convert_raw_objects, up_convert_raw_obj
        from phoofflineeeganalysis.analysis.EEG_data import EEGData
                                       
        assert lab_recorder_output_path.exists()

        lab_recorder_xdf_files: List[Path] = list(lab_recorder_output_path.glob('*.xdf'))
        lab_recorder_xdf_files
        
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
                eeg_raws = raws_dict[DataModalityType.EEG.value]
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
                
                _out_eeg_raw.append(eeg_raw)
                # stream_infos['xdf_dataset_idx'] = a_xdf_file.name ## just the name
                _out_xdf_stream_infos_df.append(stream_infos)
                
            except (ValueError, KeyError) as e:
                print(f'\t failed with error: {e}\n\tskipping file.')
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
    


