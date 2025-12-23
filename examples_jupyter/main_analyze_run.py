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
    plot_all_spectograms, plot_session_spectogram, ZarrSerialization, build_merged
)


def compute_session_summary_metrics(active_only_out_eeg_raws, results, stream_infos_df: Optional[pd.DataFrame], output_folder: Path, freq_min: float = 1.0, freq_max: float = 40.0) -> Path:
    """
    Compute simple per-session summary metrics from the spectrogram outputs and save to CSV.

    Metrics are intended to support quick comparison across sessions and include:
      - Duration, sampling rate, number of channels
      - Bandpower (absolute and relative) for classical bands (delta/theta/alpha/beta)
      - Dominant frequency within [freq_min, freq_max]

    Returns:
        Path to the created CSV file.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    rows = []

    # Precompute simple lookup from dataset index -> xdf_filename (if available)
    dataset_to_filename: Dict[int, str] = {}
    if stream_infos_df is not None:
        try:
            tmp_df = stream_infos_df.reset_index()
            if "xdf_dataset_idx" in tmp_df.columns and "xdf_filename" in tmp_df.columns:
                for ds_idx, grp in tmp_df.groupby("xdf_dataset_idx"):
                    # Use the first filename for this dataset index
                    dataset_to_filename[int(ds_idx)] = str(grp["xdf_filename"].iloc[0])
        except Exception as e:
            print(f"  WARN: failed to derive dataset -> filename mapping: {e}")

    def _band_mean(Sxx_vals: np.ndarray, freqs_vals: np.ndarray, low: float, high: float) -> float:
        band_mask = (freqs_vals >= low) & (freqs_vals < high)
        if not np.any(band_mask):
            return float("nan")
        return float(np.nanmean(Sxx_vals[:, band_mask]))

    for idx, (a_raw, a_result) in enumerate(zip(active_only_out_eeg_raws, results)):
        if a_result is None:
            continue
        if "spectogram" not in a_result:
            continue

        try:
            spect_dict = a_result["spectogram"]
            Sxx_avg = spect_dict["Sxx_avg"]  # xarray.DataArray: (channels, freqs)
            freqs_vals = np.asarray(Sxx_avg.coords["freqs"])
            Sxx_vals = np.asarray(Sxx_avg)  # (n_channels, n_freqs)
        except Exception as e:
            print(f"  WARN: failed to unpack spectrogram for session {idx}: {e}")
            continue

        # Restrict to the main comparison band for global metrics
        main_band_mask = (freqs_vals >= freq_min) & (freqs_vals <= freq_max)
        if not np.any(main_band_mask):
            print(f"  WARN: no frequencies within [{freq_min}, {freq_max}] Hz for session {idx}")
            continue

        Sxx_main = Sxx_vals[:, main_band_mask]
        freqs_main = freqs_vals[main_band_mask]

        # Total power in [freq_min, freq_max]
        total_power = float(np.nanmean(Sxx_main))

        # Classical bands
        delta_power = _band_mean(Sxx_vals, freqs_vals, 1.0, 4.0)
        theta_power = _band_mean(Sxx_vals, freqs_vals, 4.0, 8.0)
        alpha_power = _band_mean(Sxx_vals, freqs_vals, 8.0, 13.0)
        beta_power = _band_mean(Sxx_vals, freqs_vals, 13.0, 30.0)

        # Relative powers and simple ratios
        def _safe_div(n: float, d: float) -> float:
            return float(n / d) if np.isfinite(n) and np.isfinite(d) and d != 0 else float("nan")

        delta_rel = _safe_div(delta_power, total_power)
        theta_rel = _safe_div(theta_power, total_power)
        alpha_rel = _safe_div(alpha_power, total_power)
        beta_rel = _safe_div(beta_power, total_power)

        alpha_theta_ratio = _safe_div(alpha_power, theta_power)
        beta_alpha_ratio = _safe_div(beta_power, alpha_power)

        # Dominant frequency within [freq_min, freq_max] (global across channels)
        try:
            global_psd = np.nanmean(Sxx_main, axis=0)  # (n_freqs_main,)
            peak_idx = int(np.nanargmax(global_psd))
            dominant_freq_hz = float(freqs_main[peak_idx])
        except Exception:
            dominant_freq_hz = float("nan")

        # Basic session info
        meas_date = a_raw.info.get("meas_date")
        if isinstance(meas_date, datetime):
            meas_date_str = meas_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            meas_date_str = str(meas_date)

        try:
            times = a_raw.times
            duration_s = float(times[-1] - times[0]) if times.size > 0 else float("nan")
        except Exception:
            duration_s = float("nan")

        sfreq = float(a_raw.info.get("sfreq", float("nan")))
        n_channels = len(a_raw.info.get("ch_names", []))
        xdf_filename = dataset_to_filename.get(idx, None)

        rows.append(
            {
                "session_idx": idx,
                "xdf_filename": xdf_filename,
                "meas_date": meas_date_str,
                "duration_s": duration_s,
                "sfreq_hz": sfreq,
                "n_channels": n_channels,
                f"total_power_{freq_min:.1f}_to_{freq_max:.1f}_hz": total_power,
                "delta_power": delta_power,
                "theta_power": theta_power,
                "alpha_power": alpha_power,
                "beta_power": beta_power,
                "delta_rel": delta_rel,
                "theta_rel": theta_rel,
                "alpha_rel": alpha_rel,
                "beta_rel": beta_rel,
                "alpha_theta_ratio": alpha_theta_ratio,
                "beta_alpha_ratio": beta_alpha_ratio,
                "dominant_freq_hz": dominant_freq_hz,
            }
        )

    if not rows:
        print("No valid spectrogram-based metrics were computed; skipping CSV export.")
        return output_folder.joinpath("session_summaries_empty.csv")

    metrics_df = pd.DataFrame.from_records(rows)
    csv_path = output_folder.joinpath("session_summaries.csv")
    metrics_df.to_csv(csv_path, index=False)

    print(f"\nSaved per-session summary metrics to: {csv_path.as_posix()}")
    return csv_path


def export_session_spectrograms_html(active_only_out_eeg_raws, results, output_folder: Path, 
                                     freq_min: float = 1.0, freq_max: float = 40.0):
    """
    Export interactive HTML spectrograms for each EEG session using HoloViews.
    
    Creates individual HTML files for each session with:
    - Interactive spectrogram heatmaps for each channel
    - Zoomable/pannable time and frequency axes
    - Hover tooltips showing exact values
    - Session metadata in title
    
    Args:
        active_only_out_eeg_raws: List of mne.io.Raw EEG sessions
        results: List of computation results containing spectrogram data
        output_folder: Path to save HTML files
        freq_min: Minimum frequency to display (Hz)
        freq_max: Maximum frequency to display (Hz)
        
    Returns:
        List of Path objects for created HTML files
        
    Usage:
        html_files = export_session_spectrograms_html(
            active_only_out_eeg_raws, results, 
            outputs_root_folder / "spectrograms_html"
        )
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    html_files = []
    
    for idx, (a_raw, a_result) in enumerate(zip(active_only_out_eeg_raws, results)):
        try:
            # Get session metadata
            meas_date = a_raw.info.get('meas_date')
            if meas_date:
                session_name = meas_date.strftime('%Y-%m-%d_%H-%M-%S')
                session_title = meas_date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                session_name = f"session_{idx:03d}"
                session_title = f"Session {idx}"
            
            # Extract spectrogram data
            spectogram_result_dict = a_result['spectogram']['spectogram_result_dict']
            fs = a_result['spectogram']['fs']
            
            # Create HoloViews plots for each channel
            channel_plots = []
            
            for ch_idx, (ch_name, (f, t, Sxx)) in enumerate(spectogram_result_dict.items()):
                # Convert to dB scale
                Sxx_db = 10 * np.log10(Sxx + 1e-12)
                
                # Filter frequency range
                freq_mask = (f >= freq_min) & (f <= freq_max)
                f_filtered = f[freq_mask]
                Sxx_filtered = Sxx_db[freq_mask, :]
                
                # Create xarray DataArray for easier plotting
                da = xr.DataArray(
                    Sxx_filtered,
                    coords={'frequency': f_filtered, 'time': t},
                    dims=['frequency', 'time'],
                    name=f'{ch_name}_power'
                )
                
                # Create HoloViews image with hvplot
                img = da.hvplot.image(
                    x='time', y='frequency',
                    cmap='viridis',
                    clim=(Sxx_filtered.min(), Sxx_filtered.max()),
                    title=f'{ch_name}',
                    xlabel='Time (s)',
                    ylabel='Frequency (Hz)',
                    width=900,
                    height=150,
                    colorbar=True,
                    tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
                    hover_tooltips=[('Time', '@time{0.2f}s'), 
                                   ('Freq', '@frequency{0.1f}Hz'), 
                                   ('Power', '@image{0.2f}dB')]
                )
                
                channel_plots.append(img)
            
            # Stack all channel plots vertically
            layout = hv.Layout(channel_plots).cols(1)
            
            # Add overall title
            layout = layout.opts(
                title=f'EEG Spectrogram - {session_title}',
                shared_axes=True
            )
            
            # Save to HTML
            html_path = output_folder / f"spectrogram_{session_name}.html"
            hv.save(layout, html_path, backend='bokeh')
            html_files.append(html_path)
            
            print(f'  Saved spectrogram HTML for session {idx+1}/{len(active_only_out_eeg_raws)}: {html_path.name}')
            
        except Exception as e:
            print(f'  ERROR exporting session {idx}: {e}')
            continue
    
    print(f'\nExported {len(html_files)} spectrogram HTML files to: {output_folder}')
    return html_files


def export_combined_spectrograms_html(active_only_out_eeg_raws, results, output_path: Path,
                                      freq_min: float = 1.0, freq_max: float = 40.0,
                                      max_sessions_per_page: int = 10):
    """
    Export a single HTML file with all session spectrograms in a scrollable layout.
    
    Creates one HTML file with:
    - All sessions stacked vertically
    - Session selector dropdown
    - Synchronized time axes across channels
    - Compact view for comparison
    
    Args:
        active_only_out_eeg_raws: List of mne.io.Raw EEG sessions
        results: List of computation results
        output_path: Path for the output HTML file
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
        max_sessions_per_page: Maximum sessions to include (for performance)
        
    Returns:
        Path to created HTML file
        
    Usage:
        html_file = export_combined_spectrograms_html(
            active_only_out_eeg_raws, results,
            outputs_root_folder / "all_spectrograms.html"
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Limit sessions for performance
    n_sessions = min(len(active_only_out_eeg_raws), max_sessions_per_page)
    
    all_session_layouts = []
    
    for idx in range(n_sessions):
        a_raw = active_only_out_eeg_raws[idx]
        a_result = results[idx]
        
        try:
            # Get session metadata
            meas_date = a_raw.info.get('meas_date')
            if meas_date:
                session_title = meas_date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                session_title = f"Session {idx}"
            
            # Extract spectrogram data
            spectogram_result_dict = a_result['spectogram']['spectogram_result_dict']
            
            # Create compact channel plots
            channel_plots = []
            
            for ch_name, (f, t, Sxx) in spectogram_result_dict.items():
                Sxx_db = 10 * np.log10(Sxx + 1e-12)
                
                # Filter frequency
                freq_mask = (f >= freq_min) & (f <= freq_max)
                f_filtered = f[freq_mask]
                Sxx_filtered = Sxx_db[freq_mask, :]
                
                # Create compact plot
                da = xr.DataArray(
                    Sxx_filtered,
                    coords={'frequency': f_filtered, 'time': t},
                    dims=['frequency', 'time']
                )
                
                img = da.hvplot.image(
                    x='time', y='frequency',
                    cmap='viridis',
                    title=f'{ch_name}',
                    xlabel='',
                    ylabel='Hz',
                    width=800,
                    height=80,
                    colorbar=False,
                    tools=['hover', 'pan', 'wheel_zoom', 'reset']
                )
                
                channel_plots.append(img)
            
            # Create session layout
            session_layout = hv.Layout(channel_plots).cols(1).opts(
                title=f'{session_title}'
            )
            
            all_session_layouts.append(session_layout)
            
        except Exception as e:
            print(f'  ERROR processing session {idx}: {e}')
            continue
    
    # Combine all sessions
    combined = hv.Layout(all_session_layouts).cols(1)
    
    # Save to HTML
    hv.save(combined, output_path, backend='bokeh')
    
    print(f'\nExported combined spectrogram HTML with {len(all_session_layouts)} sessions to: {output_path}')
    return output_path


# Configuration
mne.viz.set_browser_backend("qt")
mne.set_config("MNE_BROWSER_BACKEND", "qt")
set_log_level("WARNING")

hv.extension('bokeh', logo=False)
hvplot.extension('bokeh')
pn.extension()

# Jupyter-lab enable printing for any line on its own (instead of just the last one in the cell)
# InteractiveShell.ast_node_interactivity = "all"

# Initialize datasets
datasets = []


# db_root_path = Path('/content/drive/MyDrive/Databases').resolve()
db_root_path = Path('E:/Dropbox (Personal)/Databases').resolve()
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

outputs_root_folder: Path = Path('L:/AITEMP/PhoOfflineEEGAnalysisOutputs').resolve()
assert outputs_root_folder.exists()

lab_recorder_output_path = Path("E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001").resolve()
assert lab_recorder_output_path.exists()


def process_XDFs_main(n_most_recent_sessions_to_preprocess: Optional[int] = 5,
        should_write_final_merged_eeg_fif: bool = True,
        included_xdf_file_names: Optional[List]=None, # Include all
        should_load_preprocessed: bool = False,
    ):
    """ 

        from PhoOfflineEEGAnalysis.examples_jupyter.main_analyze_run import process_XDFs_main

    """


    # SavedSessionProcessor

    sso: SavedSessionsProcessor = SavedSessionsProcessor(eeg_recordings_file_path=eeg_recordings_file_path,
                                                        headset_motion_recordings_file_path=headset_motion_recordings_file_path, WhisperVideoTranscripts_LSL_Converted_file_path=WhisperVideoTranscripts_LSL_Converted, pho_log_to_LSL_recordings_path=pho_log_to_LSL_recordings_path,
                                                        eeg_analyzed_parent_export_path=eeg_analyzed_parent_export_path, 
                                                        n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess, 
                                                        should_load_data=True, should_load_preprocessed=should_load_preprocessed,
                                                        #  should_load_data=True, should_load_preprocessed=True,
                                                        )


    labRecorder_PostProcessed_path: Path = sso.eeg_analyzed_parent_export_path.joinpath(f'LabRecorder_PostProcessed')
    labRecorder_PostProcessed_path.mkdir(exist_ok=True)



    # Parallel XDF file processing
    assert lab_recorder_output_path.exists()

    lab_recorder_xdf_files: list[Path] = list(lab_recorder_output_path.glob('*.xdf'))
    n_total_found_files: int = len(lab_recorder_xdf_files)

    # Filter by included file names if specified
    if included_xdf_file_names is not None:
        print(f'limiting to included_xdf_file_names: {included_xdf_file_names}...')
        lab_recorder_xdf_files = [v for v in lab_recorder_xdf_files if v.name in included_xdf_file_names]
        n_filtered_found_files: int = len(lab_recorder_xdf_files)
        print(f'\tlimited to {n_filtered_found_files}/{n_total_found_files} files')

    # Limit to n_most_recent_sessions_to_preprocess most recent files
    if n_most_recent_sessions_to_preprocess is not None and len(lab_recorder_xdf_files) > n_most_recent_sessions_to_preprocess:
        print(f'Limiting to {n_most_recent_sessions_to_preprocess} most recent XDF files...')
        # Sort by modification time (most recent first)
        lab_recorder_xdf_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        lab_recorder_xdf_files = lab_recorder_xdf_files[:n_most_recent_sessions_to_preprocess]
        print(f'\tLimited to {len(lab_recorder_xdf_files)}/{n_total_found_files} most recent files')

    if (labRecorder_PostProcessed_path is not None) and should_write_final_merged_eeg_fif:
        labRecorder_PostProcessed_path.mkdir(exist_ok=True)

    # Determine optimal number of workers
    max_workers = min(len(lab_recorder_xdf_files), (os.cpu_count() or 4))
    print(f"Processing {len(lab_recorder_xdf_files)} XDF files using {max_workers} parallel workers...")

    # Initialize result containers
    _out_eeg_raw = [None] * len(lab_recorder_xdf_files)
    _out_xdf_stream_infos_df = [None] * len(lab_recorder_xdf_files)
    _out_results = [None] * len(lab_recorder_xdf_files)


    def _subfn_process_single_xdf_file(idx_file_tuple):
        """Process a single XDF file - captures lots of things 
        """
        an_xdf_file_idx, a_xdf_file = idx_file_tuple
        try:
            print(f'  Processing XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"...')

            # Load XDF file
            _obj = LabRecorderXDF.init_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file)
            stream_infos = _obj.stream_infos
            raws = _obj.datasets
            raws_dict = _obj.datasets_dict

            eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])

            if len(eeg_raws) == 0:
                print(f'  WARN: no EEG streams found in "{a_xdf_file.as_posix()}". Skipping file.')
                return an_xdf_file_idx, None, None, None

            # Merge by device so we can handle multiple EEG streams per XDF
            merged_eeg_raws, merge_meta = LabRecorderXDF.merge_eeg_streams_by_device(
                eeg_raws=eeg_raws, strict_merge=False, debug_print=False
            )
            if len(merged_eeg_raws) == 0:
                print(f'  WARN: could not produce any merged EEG datasets for "{a_xdf_file.as_posix()}". Skipping file.')
                return an_xdf_file_idx, None, None, None

            # Save post-processed data if requested (one set per merged dataset)
            exports_dict = None
            if should_write_final_merged_eeg_fif:
                _, exports_dict = LabRecorderXDF.save_post_processed_to_fif(
                    raws_dict=raws_dict,
                    a_xdf_file=a_xdf_file,
                    labRecorder_PostProcessed_path=labRecorder_PostProcessed_path,
                )

            # For the notebook path we currently return only the first merged EEG
            # dataset for downstream plotting, but we still respect multi-stream
            # handling for disk outputs.
            eeg_raw = merged_eeg_raws[0]

            # Add metadata
            stream_infos['lab_recorder_xdf_file_idx'] = an_xdf_file_idx
            stream_infos['xdf_filename'] = a_xdf_file.name
            stream_infos['eeg_device_group_idx'] = 0
            stream_infos['eeg_device_key'] = merge_meta[0].get('device_key', 'device_0')
            stream_infos['n_eeg_segments_in_group'] = merge_meta[0].get('n_segments', 1)

            if exports_dict is not None:
                for a_format, per_idx_dict in exports_dict.items():
                    export_path = per_idx_dict.get(0, None)
                    if export_path is not None:
                        # Ensure we handle both Path and string-like values
                        stream_infos[f'proccessed_{a_format}_filename'] = Path(export_path).name

            # Up-convert and set montage
            eeg_raw = up_convert_raw_obj(eeg_raw)
            EEGData.set_montage(datasets_EEG=[eeg_raw])
            eeg_raw.debug_test_annotations_timestamps()

            # Do post-processing stage
            result = None
            try:
                meas_date = eeg_raw.info.get('meas_date', 'Unknown')
                print(f"  Processing merged EEG dataset for file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)} (meas_date: {meas_date})")
                # Disable BAD_* annotation masking for spectrograms in this pipeline to reduce NaN gaps
                result = EEGComputations.run_all(raw=eeg_raw, mask_bad_annotated_times=False)
                print(f"  Completed merged EEG dataset for file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)} (meas_date: {meas_date})")
            except Exception as e:
                print(f"  ERROR processing merged EEG dataset for file {an_xdf_file_idx+1}: {e}")

            print(f'  Completed XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"')
            return an_xdf_file_idx, eeg_raw, stream_infos, result

        except (ValueError, KeyError, AssertionError, TypeError) as e:
            print(f'  ERROR in XDF file {an_xdf_file_idx+1}: {e}\n  Skipping file.')
            return an_xdf_file_idx, None, None, None

        except Exception as e:
            print(f'  EXCEPTION in XDF file {an_xdf_file_idx+1}: {e}')
            raise


    # ---------------------------------------------------------------------------- #
    #                 Parallel processing using ThreadPoolExecutor                 #
    # ---------------------------------------------------------------------------- #
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(_subfn_process_single_xdf_file, (idx, xdf_file)): idx 
            for idx, xdf_file in enumerate(lab_recorder_xdf_files)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                idx, eeg_raw, stream_infos, result = future.result()
                _out_eeg_raw[idx] = eeg_raw
                _out_xdf_stream_infos_df[idx] = stream_infos
                _out_results[idx] = result
            except Exception as e:
                idx = future_to_idx[future]
                print(f"  EXCEPTION collecting result for file {idx+1}: {e}")
                _out_eeg_raw[idx] = None
                _out_xdf_stream_infos_df[idx] = None
                _out_results[idx] = None



    # Filter out failed files
    valid_indices = [i for i, (raw, info, result) in enumerate(zip(_out_eeg_raw, _out_xdf_stream_infos_df, _out_results)) if raw is not None and info is not None]
    _out_eeg_raw = [_out_eeg_raw[i] for i in valid_indices]
    _out_xdf_stream_infos_df = [_out_xdf_stream_infos_df[i] for i in valid_indices]
    _out_results = [_out_results[i] for i in valid_indices]

    # Add xdf_dataset_idx to stream_infos
    for dataset_idx, stream_infos in enumerate(_out_xdf_stream_infos_df):
        stream_infos['xdf_dataset_idx'] = dataset_idx

    # Combine stream infos
    _out_xdf_stream_infos_df = pd.concat(_out_xdf_stream_infos_df)
    _out_xdf_stream_infos_df = _out_xdf_stream_infos_df.set_index('xdf_dataset_idx')

    # Up-convert and sort (with results following the same order)
    _out_eeg_raw = up_convert_raw_objects(_out_eeg_raw)
    # Create sorting indices
    sort_indices = sorted(range(len(_out_eeg_raw)), key=lambda i: (_out_eeg_raw[i].raw_timerange()[0] is None, _out_eeg_raw[i].raw_timerange()[0]))
    _out_eeg_raw = [_out_eeg_raw[i] for i in sort_indices]
    _out_results = [_out_results[i] for i in sort_indices]

    # Set montage for all datasets
    EEGData.set_montage(datasets_EEG=_out_eeg_raw)  # pyright: ignore[reportUnknownMemberType]

    xdf_dataset_indicies = np.unique(deepcopy(_out_xdf_stream_infos_df).reset_index(drop=False, inplace=False)['xdf_dataset_idx'].to_numpy())
    n_unique_xdf_datasets: int = len(xdf_dataset_indicies)
    print(f'n_unique_xdf_datasets: {n_unique_xdf_datasets}')

    _out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw)
    _out_xdf_stream_infos_df

    ## Use the results already computed during parallel processing
    # No need to call batch_compute_all_eeg_datasets again since we already computed during XDF loading
    active_only_out_eeg_raws = _out_eeg_raw
    results = _out_results

    num_sessions: int = len(results)
    print(f'Processed {num_sessions} sessions with EEG computations')


    ## OUTPUTS: _out_xdf_stream_infos_df, active_only_out_eeg_raws, results, 

    return sso, xdf_dataset_indicies, _out_xdf_stream_infos_df, active_only_out_eeg_raws, results


# xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
# xdf_stream_infos_df










if __name__ == "__main__":

    # n_most_recent_sessions_to_preprocess: int = None # None means all sessions
    # n_most_recent_sessions_to_preprocess: int = 35
    n_most_recent_sessions_to_preprocess: int = 15

    should_load_preprocessed: bool = False
    # should_load_preprocessed: bool = True

    should_write_final_merged_eeg_fif: bool = False
    # should_write_final_merged_eeg_fif: bool = True

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


    sso, xdf_dataset_indicies, _out_xdf_stream_infos_df, active_only_out_eeg_raws, results = process_XDFs_main(included_xdf_file_names=included_xdf_file_names, 
                                                                                                                n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess,
                                                                                                                should_write_final_merged_eeg_fif=should_write_final_merged_eeg_fif,
                                                                                                                should_load_preprocessed=should_load_preprocessed,
    )

    ## Extract comments/notes/annotations/etc from the outputs
    _extracted_comments = []
    ignored_comment_descriptions = ['BAD_motion', '']
    for a_raw in active_only_out_eeg_raws:
        an_annotations = a_raw.annotations
        if (an_annotations is not None) and (len(an_annotations) > 0):
            an_annotation_df = an_annotations.to_data_frame(time_format='datetime')
            an_annotation_df = an_annotation_df[np.logical_not(np.isin(an_annotation_df['description'], ignored_comment_descriptions))]
            _extracted_comments.append(an_annotation_df)

    if _extracted_comments:
        extracted_comments_df: pd.DataFrame = pd.concat(_extracted_comments)
        extracted_comments_df = extracted_comments_df.rename(columns={'onset':'time', 'description':'text'}, inplace=False)
        print(f'Extracted {len(extracted_comments_df)} comments/annotations')
    else:
        print('No comments/annotations found')

    # ------------------------------------------------------------------------- #
    #                Compute and save per-session summary metrics              #
    # ------------------------------------------------------------------------- #
    print('\nComputing per-session summary metrics from spectrograms...')
    summary_output_folder = outputs_root_folder.joinpath('session_summaries')
    summary_csv_path = compute_session_summary_metrics(
        active_only_out_eeg_raws=active_only_out_eeg_raws,
        results=results,
        stream_infos_df=_out_xdf_stream_infos_df,
        output_folder=summary_output_folder,
        freq_min=1.0,
        freq_max=40.0,
    )

    # ## Save results to Zarr format
    # # Create a simple day_status_dict (you can customize this based on your needs)
    # day_status_dict = {}
    # for a_raw in active_only_out_eeg_raws:
    #     a_meas_date = a_raw.info.get('meas_date')
    #     if a_meas_date:
    #         a_raw_key: str = a_meas_date.strftime("%Y-%m-%d/%H-%M-%S")
    #         day_status_dict[a_raw_key] = 'cog_UNLABELED'  # Default status

    # # Save to Zarr
    # zarr_out_path = outputs_root_folder.joinpath(f"2025-11-18_all_sessions_{len(active_only_out_eeg_raws)}_files.zarr").resolve()
    # print(f'Saving {len(active_only_out_eeg_raws)} sessions to Zarr: "{zarr_out_path.as_posix()}"...')
    # zarr_out_path = ZarrSerialization.save_sessions_as_zarr(
    #     active_only_out_eeg_raws=active_only_out_eeg_raws, 
    #     results=results, 
    #     day_status_dict=day_status_dict, 
    #     out_path=str(zarr_out_path)
    # )
    # zarr_out_path = Path(zarr_out_path)  # Convert back to Path
    # print(f'Successfully saved to: "{zarr_out_path.as_posix()}"')

    # # Build merged dataset for NetCDF export
    # print('Building merged spectogram dataset...')
    # combined_spectogram_ds, combined_spectogram_da = build_merged(
    #     active_only_out_eeg_raws=active_only_out_eeg_raws, 
    #     results=results, 
    #     day_status_dict=day_status_dict,
    #     only_include_sessions_with_status_entries=False
    # )

    # # Save to NetCDF
    # netcdf_save_path: Path = outputs_root_folder.joinpath(f"2025-11-18_saved_spectogram_{len(active_only_out_eeg_raws)}_files.nc").resolve()
    # print(f'Saving spectogram to NetCDF: "{netcdf_save_path.as_posix()}"...')
    # combined_spectogram_da.to_netcdf(netcdf_save_path)
    # print(f'Successfully saved spectogram to: "{netcdf_save_path.as_posix()}"')

    # Export interactive HTML spectrograms
    print('\nExporting interactive HTML spectrograms...')
    html_output_folder = outputs_root_folder.joinpath('spectrograms_html')
    print(f'\texporting to "{html_output_folder}"...')
    html_files = export_session_spectrograms_html(
        active_only_out_eeg_raws=active_only_out_eeg_raws,
        results=results,
        output_folder=html_output_folder,
        freq_min=1.0,
        freq_max=40.0
    )
    print(f'\tdone.')
    # # Export combined HTML view
    # combined_html_path = outputs_root_folder.joinpath(f"2025-11-18_all_spectrograms_{len(active_only_out_eeg_raws)}_sessions.html")
    # combined_html_file = export_combined_spectrograms_html(
    #     active_only_out_eeg_raws=active_only_out_eeg_raws,
    #     results=results,
    #     output_path=combined_html_path,
    #     freq_min=1.0,
    #     freq_max=40.0,
    #     max_sessions_per_page=10
    # )

    print(f'\n=== Processing Complete ===')
    print(f'Total sessions processed: {len(active_only_out_eeg_raws)}')
    # print(f'Zarr output: {zarr_out_path}')
    # print(f'NetCDF output: {netcdf_save_path}')
    print(f'Individual HTML spectrograms: {html_output_folder} ({len(html_files)} files)')
    print(f'Session summary metrics CSV: {summary_csv_path}')
    # print(f'Combined HTML spectrogram: {combined_html_file}')