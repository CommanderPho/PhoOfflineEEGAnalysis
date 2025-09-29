import time
import re
from datetime import datetime, timezone

import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import h5py
from nptyping import NDArray
from matplotlib import pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.signal import spectrogram
import xarray as xr

import mne
from mne import set_log_level
from copy import deepcopy
from mne.filter import filter_data  # Slightly faster for large data as direct fn

from mne.io import read_raw
import pyedflib ## for creating single EDF+ files containing channel with different sampling rates (e.g. EEG and MOTION data)

mne.viz.set_browser_backend("Matplotlib")

# from ..EegProcessing import bandpower
from numpy.typing import NDArray


set_log_level("WARNING")


class EEGData:
    """ Methods related to processing of motion (gyro/accel/magnet/quaternion/etc) data

    from phoofflineeeganalysis.analysis.EEG_data import EEGData

    (all_data_MOTION, all_times_MOTION), datasets_MOTION, df_MOTION = flat_data_modality_dict['MOTION']  ## Unpacking
    df_MOTION

    """

    @classmethod
    def _perform_process_EEG_session(cls, raw_eeg):
        """ Called by `preprocess` on each rawEEG session
         
            eeg_analysis_results = _perform_process_EEG_session(raw_eeg)
            eeg_analysis_results['eog_epochs'].plot_image(combine="mean")
            eeg_analysis_results['eog_epochs'].average().plot_joint()

            # Visualize the extracted microstates
            nk.microstates_plot(eeg_analysis_results['microstates'], epoch=(0, 500)) # RuntimeError: No digitization points found.

        """
        eeg_analysis_results = {}
        # Apply band-pass filter (1-58Hz) and re-reference the raw signal
        raw_eeg = raw_eeg.copy().filter(1, 58, verbose=False)
        # eeg = nk.eeg_rereference(eeg, "average") ## do not do this, it apparently reduces quality

        # Extract microstates
        # eeg_analysis_results['microstates'] = nk.microstates_segment(raw_eeg, n_microstates=4)

        ## Try EOG -- fials due to lack of digitization points -- # RuntimeError: No digitization points found.
        eeg_analysis_results['eog_epochs'] = mne.preprocessing.create_eog_epochs(raw_eeg, ch_name=['AF3', 'AF4']) ## , baseline=(-0.5, -0.2) use the EEG channels closest to the eyes - eog_epochs 

        return eeg_analysis_results
    

    @classmethod
    def set_montage(cls, datasets_EEG: List):
        """ 
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)

        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking


        """
        from mne.channels.montage import DigMontage
        from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper

        active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
        emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage
        
        if isinstance(datasets_EEG, (mne.io.BaseRaw, mne.io.Raw, mne.io.RawArray)):
            # datasets_EEG = [datasets_EEG] # single element list
            datasets_EEG.set_montage(emotiv_epocX_montage)
        else:
            for i, raw_eeg in enumerate(datasets_EEG):
                # raw_eeg = raw_eeg.pick(["eeg"], verbose=False)
                # raw_eeg.load_data()
                # sampling_rate = raw_eeg.info["sfreq"]  # Store the sampling rate
                raw_eeg.set_montage(emotiv_epocX_montage)


    @classmethod
    def preprocess(cls, datasets_EEG: List, preprocessed_EEG_save_path: Optional[Path]=None, n_most_recent_sessions_to_preprocess: Optional[int] = 5):
        """ 
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)
        
        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking
        
        
        """
        from mne.channels.montage import DigMontage
        from phoofflineeeganalysis.analysis.anatomy_and_electrodes import ElectrodeHelper

        active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
        emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage

        ## BEGIN ANALYSIS of EEG Data
        num_EEG_files: int = len(datasets_EEG)
        eeg_session_IDXs = np.arange(num_EEG_files)

        if (n_most_recent_sessions_to_preprocess is not None) and (n_most_recent_sessions_to_preprocess > 0):
            n_most_recent_sessions_to_preprocess = min(n_most_recent_sessions_to_preprocess, num_EEG_files) ## don't process more than we have
            active_eeg_rec_IDXs = eeg_session_IDXs[-n_most_recent_sessions_to_preprocess:]
        else:
            ## ALL sessions
            active_eeg_rec_IDXs = deepcopy(eeg_session_IDXs)

        analysis_results_EEG = []
        valid_active_IDXs = []

        for i in active_eeg_rec_IDXs:
            eeg_analysis_results = None
            try:
                raw_eeg = datasets_EEG[i]
                raw_eeg = raw_eeg.pick(["eeg"], verbose=False)
                raw_eeg.load_data()
                sampling_rate = raw_eeg.info["sfreq"]  # Store the sampling rate
                raw_eeg.set_montage(emotiv_epocX_montage)
                datasets_EEG[i] = raw_eeg ## update it and put it back
                eeg_analysis_results = cls._perform_process_EEG_session(raw_eeg)
            except ValueError as e:
                print(f'Encountered value error: {e} while trying to processing EEG file {i}/{len(datasets_EEG)}: {raw_eeg}. Skipping')
                datasets_EEG[i] = None ## drop result
                eeg_analysis_results = None ## no analysis result
                pass
            except Exception as e:
                raise e
            
            if eeg_analysis_results is not None:
                analysis_results_EEG.append(eeg_analysis_results)
                valid_active_IDXs.append(i)
                
                if preprocessed_EEG_save_path is not None:
                    if not preprocessed_EEG_save_path.exists():
                        preprocessed_EEG_save_path.mkdir(parents=True, exist_ok=True)
                    a_raw_savepath: Path = preprocessed_EEG_save_path.joinpath(raw_eeg.filenames[0].name).resolve()
                    raw_eeg.save(a_raw_savepath, overwrite=True)
                    
            else:
                # valid_active_IDXs
                print(f'EEG dataset {i} is invalid. Skipping.')
                pass
            

        valid_active_IDXs = np.array(valid_active_IDXs)
        
        ## OUTPUTS: analysis_results_EEG
        ## UPDATES: eeg_session_IDXs
        # return (active_session_IDXs, analysis_results_EEG)
        return (valid_active_IDXs, analysis_results_EEG)
    





class EEGComputations:
    """ 
    
    from phoofflineeeganalysis.analysis.EEG_data import EEGComputations, EEGData
    
        
        
    _all_outputs = EEGComputations.run_all(raw=raw)
    """
    @classmethod
    def all_fcns_dict(cls):
        return {
            'raw_data_topo': cls.raw_data_topo,
            'cwt': cls.raw_morlet_cwt, 
            'spectogram': cls.raw_spectogram_working,
        }


    @classmethod
    def run_all(cls, raw, should_suppress_exceptions: bool=True, **kwargs):
        _all_outputs = {}
        _all_fns = cls.all_fcns_dict()
        for a_fn_name, a_fn in _all_fns.items():
            print(f'running {a_fn_name}...')
            try:
                _all_outputs[a_fn_name] = a_fn(raw, **kwargs)
                print(f'\tdone.')
            except Exception as e:
                print(f'\terror occured in {a_fn_name}: error {e}.')
                if not should_suppress_exceptions:
                    raise
            
        return _all_outputs
    
    @classmethod
    def raw_morlet_cwt(cls, raw: mne.io.Raw, picks=None, wavelet_param=4, num_freq=60, fmax=50, spacing=12.5):
        """Compute continuous Morlet wavelet transform for MNE Raw EEG.

        raw: mne.io.Raw
        picks: channels to use (default: all EEG)
        wavelet_param: number of cycles
        num_freq: number of frequencies
        fmax: highest frequency (Hz)
        spacing: frequency step (Hz) if <1, treated as log spacing factor
        """
        if picks is None:
            picks = mne.pick_types(raw.info, eeg=True, meg=False)

        fs = raw.info["sfreq"]
        data = raw.get_data(picks=picks)

        if spacing < 1:  # logarithmic spacing
            freqs = np.geomspace(fmax/num_freq, fmax, num=num_freq)
        else:  # linear spacing
            freqs = np.arange(spacing, fmax+spacing, spacing)[:num_freq]

        power = mne.time_frequency.tfr_array_morlet(data[np.newaxis], sfreq=fs, freqs=freqs, n_cycles=wavelet_param, output="power")

        return dict(freqs=freqs, power=power[0])
    

    @classmethod
    def raw_data_topo(cls, raw: mne.io.Raw, l_freq=1, h_freq=58, epoch_dur=4,
                       epoch_step: float = 0.250,
                       moving_avg_epochs: int = 32):
        """Compute continuous Morlet wavelet transform for MNE Raw EEG.

        raw: mne.io.Raw
        picks: channels to use (default: all EEG)
        wavelet_param: number of cycles
        num_freq: number of frequencies
        fmax: highest frequency (Hz)
        spacing: frequency step (Hz) if <1, treated as log spacing factor
        
        l_freq=1
        h_freq=58
        epoch_dur=4
        epoch_step=0.025
        moving_avg_epochs=32
        
        epoch_avg = _all_outputs['raw_data_topo']['epoch_avg']
                
        epochs = _all_outputs['raw_data_topo']['epochs']
        
        
        """
        # out_dict = dict(raw_filtered=out_dict['raw_filtered'], topo=out_dict['data_topo'], epoch_avg=out_dict['epoch_avg'], mov_avg=out_dict['mov_avg'], epochs=out_dict['epochs'])
        out_dict = dict(raw_filtered=None, topo=None, epoch_avg=None, mov_avg=None, epochs=None)
        
        # 1. Temporal filter: filter modifies in-place if not copying
        out_dict['raw_filtered'] = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', n_jobs='cuda')

        # 2. Epoching (sliding window)
        sfreq = raw.info['sfreq']
        step_samples: int = int(epoch_step * sfreq)
        window_samples: int  = int(epoch_dur * sfreq)
        data = out_dict['raw_filtered'].get_data()
        n_ch, n_times = data.shape

        print(f'for INPUT PARAMS: epoch_dur: {epoch_dur}, epoch_step: {epoch_step}, moving_avg_epochs: {moving_avg_epochs}')
        print(f'\tstep_samples: {step_samples}, window_samples: {window_samples}\n\tn_ch: {n_ch}, n_times: {n_times}')
        # Use strided view for efficient sliding windows (no explicit python loops)
        def epoch_strided(data, window_samples, step_samples):
            n_epochs = (n_times - window_samples) // step_samples + 1
            s0, s1 = data.strides
            return np.lib.stride_tricks.as_strided(
                data,
                shape=(n_epochs, n_ch, window_samples),
                strides=(step_samples * s1, s0, s1),
                writeable=False
            )

        out_dict['epochs'] = epoch_strided(data, window_samples, step_samples)
        # shape: (n_epochs, n_ch, n_samples)

        # 3. x²
        epochs_squared = np.square(out_dict['epochs'])

        # 4. Moving epoch average (vectorized via convolution)
        # kernel = np.ones(moving_avg_epochs) / moving_avg_epochs
        # pad so output is same shape (len=n_epochs)
        # mov_avg = np.apply_along_axis(
        #     lambda m: np.vstack([np.convolve(m[:, ch, samp], kernel, mode='full')[:len(m)] 
        #                         for ch in range(n_ch) for samp in range(window_samples)]
        #                     ).reshape((n_ch, window_samples, len(m))).transpose(2, 0, 1),
        #     axis=0, arr=epochs_squared[None,...]).squeeze()

        # Alternatively, for moving window average over epochs axis, use:
        # import scipy.ndimage
        out_dict['mov_avg'] = scipy.ndimage.uniform_filter1d(epochs_squared, size=moving_avg_epochs, axis=0, origin=-(moving_avg_epochs//2)).squeeze()

        # 5. Epoch average (over all epochs)
        out_dict['epoch_avg'] = out_dict['mov_avg'].mean(axis=0)  # (n_ch, n_samples)
        out_dict['data_topo'] = out_dict['epoch_avg'].mean(axis=1)  # (n_ch,)
        
        return out_dict

    @classmethod
    def raw_spectogram_working(cls, raw: mne.io.Raw, picks=None, nperseg=1024, noverlap=512):
        """Compute continuous Morlet wavelet transform for MNE Raw EEG.

        raw: mne.io.Raw
        picks: channels to use (default: all EEG)
        wavelet_param: number of cycles
        num_freq: number of frequencies
        fmax: highest frequency (Hz)
        spacing: frequency step (Hz) if <1, treated as log spacing factor


        plt.figure(num='spectrogram', clear=True)
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='auto'); plt.ylim([1,40])


        Unpack like:

            a_spectogram_result: Dict = a_result['spectogram'] 

            ch_names = a_spectogram_result['ch_names']
            fs = a_spectogram_result['fs']
            a_spectogram_result_dict = a_spectogram_result['spectogram_result_dict'] # Dict[channel: Tuple]
            Sxx = a_spectogram_result['Sxx']
            Sxx_avg = a_spectogram_result['Sxx_avg']
            
            for a_ch, a_tuple in a_spectogram_result_dict.items():
                f, t, Sxx = a_tuple ## unpack the tuple
                
        """
        # from scipy.signal import spectrogram        

        if picks is None:
            picks = mne.pick_types(raw.info, eeg=True, meg=False)

        fs: float = raw.info["sfreq"]
        data = raw.get_data(picks=picks)
        ch_names = deepcopy(raw.info.ch_names)
        # raw = deepcopy(raw)
        # raw.down_convert_to_base_type()
        Sxx_list = []
        Sxx_avg_list = []
        
        spectogram_result_dict = {}
        for ch_idx, a_ch in enumerate(raw.info.ch_names):
            f, t, Sxx = spectrogram(data[ch_idx], fs=fs, nperseg=nperseg, noverlap=noverlap) # #TODO 2025-09-28 13:25: - [ ] Convert to newer `ShortTimeFFT.spectrogram`
            spectogram_result_dict[a_ch] = (f, t, Sxx) ## a tuple
            Sxx_list.append(Sxx) # np.shape(Sxx) # (513, 1116) - (n_freqs, n_times)
            Sxx_avg = np.nanmean(Sxx, axis=-1) ## average over all time to get one per session
            Sxx_avg_list.append(Sxx_avg)

        Sxx_avg_list = np.stack(Sxx_avg_list) # (14, 513) - (n_channels, n_freqs)
        Sxx_list = np.stack(Sxx_list) # (14, 513, 1116) - (n_channels, n_freqs, n_times)
            
        Sxx_avg_list = xr.DataArray(Sxx_avg_list, dims=("channels", "freqs"), coords={"channels": ch_names, "freqs": f})
        Sxx_list = xr.DataArray(Sxx_list, dims=("channels", "freqs", "times"), coords={"channels": ch_names, "freqs": f, "times": t})

        # return dict(fs=fs, spectogram_result_dict=spectogram_result_dict)

        return dict(t=t, freqs=f, fs=fs, ch_names=ch_names,
                    spectogram_result_dict=spectogram_result_dict,
                                        Sxx_avg=Sxx_avg_list,
                                        Sxx=Sxx_list,
                                        )
    



    @classmethod
    def perform_write_to_hdf(cls, a_result, f, root_key: str='/', debug_print=True):
        """ 
        EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

        from phoofflineeeganalysis.analysis.EEG_data import EEGComputations

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
        def _perform_write_dict_recurrsively(attribute, a_value):
            if debug_print:
                print(f'attribute: {attribute}')
            if isinstance(a_value, pd.DataFrame):
                a_value.to_hdf(f, key=attribute, append=True)
            elif isinstance(a_value, (xr.DataArray, xr.Dataset)):
                # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
                # f.create_dataset(attribute, data=a_value.values)
                f.create_dataset(attribute, data=a_value)
                # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
            elif isinstance(a_value, np.ndarray):
                f.create_dataset(attribute, data=a_value)
            elif isinstance(a_value, (str, float, int)):
                # f.attrs.create(
                print(f'cannot yet write attributes. Skipping "{attribute}" of type {type(a_value)}')
            elif isinstance(a_value, dict):
                for a_sub_attribute, a_sub_value in a_value.items():
                    ## process each subattribute independently
                    _perform_write_dict_recurrsively(f"{attribute}/{a_sub_attribute}", a_sub_value)

            elif (Path(attribute).parts[-2] == 'spectogram_result_dict') and isinstance(a_value, tuple) and len(a_value) == 3:
                ## unpack tuple
                # freqs, t, Sxx = a_value
                ## convert to dict and pass attribute as-is
                # _perform_write_dict_recurrsively(attribute, {'f': freqs, 't': t, 'Sxx': Sxx})
                pass
            else:
                print(f'error: {attribute} of type {type(a_value)} cannot be written. Skipping')                

        _perform_write_dict_recurrsively(f'{root_key}', a_value=a_result)




    @classmethod
    def to_hdf(cls, a_result, file_path: Path, root_key: str='/', debug_print=True):
        """ 
        EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

        from phoofflineeeganalysis.analysis.EEG_data import EEGComputations

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
        write_mode = 'r+'
        if (not file_path.exists()):
            write_mode = 'w'

        with h5py.File(file_path, write_mode) as f:

            def _perform_write_dict_recurrsively(attribute, a_value):
                if debug_print:
                    print(f'attribute: {attribute}')
                if isinstance(a_value, pd.DataFrame):
                    a_value.to_hdf(file_path, key=attribute, append=True)
                elif isinstance(a_value, (xr.DataArray, xr.Dataset)):
                    # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
                    # f.create_dataset(attribute, data=a_value.values)
                    f.create_dataset(attribute, data=a_value)
                    # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
                elif isinstance(a_value, np.ndarray):
                    f.create_dataset(attribute, data=a_value)
                elif isinstance(a_value, (str, float, int)):
                    # f.attrs.create(
                    print(f'cannot yet write attributes. Skipping "{attribute}" of type {type(a_value)}')
                elif isinstance(a_value, dict):
                    for a_sub_attribute, a_sub_value in a_value.items():
                        ## process each subattribute independently
                        _perform_write_dict_recurrsively(f"{attribute}/{a_sub_attribute}", a_sub_value)
                        
                elif (Path(attribute).parts[-2] == 'spectogram_result_dict') and isinstance(a_value, tuple) and len(a_value) == 3:
                    ## unpack tuple
                    # freqs, t, Sxx = a_value
                    ## convert to dict and pass attribute as-is
                    # _perform_write_dict_recurrsively(attribute, {'f': freqs, 't': t, 'Sxx': Sxx})
                    pass
                else:
                    print(f'error: {attribute} of type {type(a_value)} cannot be written. Skipping')                

            _perform_write_dict_recurrsively(f'{root_key}', a_value=a_result)



    # ==================================================================================================================================================================================================================================================================================== #
    # PLOTTING/VISUALIZATION FUNCTIONS                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #

