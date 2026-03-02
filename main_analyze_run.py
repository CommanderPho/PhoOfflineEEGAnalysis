# Standard library imports
import hashlib
import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import h5py
import numpy as np
import pandas as pd
import xarray as xr

# MNE imports
import mne
from mne import set_log_level

# Visualization imports
import holoviews as hv
import hvplot.xarray
import panel as pn

# IPython imports

# Project-specific imports
from phoofflineeeganalysis.analysis.MNE_helpers import (
    up_convert_raw_objects, up_convert_raw_obj
)
from phopymnehelper.EEG_data import EEGComputations, EEGData
from phoofflineeeganalysis.analysis.SavedSessionsProcessor import SavedSessionsProcessor, DataModalityType
from phoofflineeeganalysis.analysis.xdf_files import LabRecorderXDF, XDFDataStreamAccessor

COMPUTATION_HISTORY_COLUMNS = ["cache_key_hex", "xdf_path", "xdf_mtime", "params_json", "result_path", "fif_filename", "computed_at"]


def _compute_computation_cache_key(xdf_path: Path, params: Dict[str, Any], mtime: Optional[float] = None) -> str:
    """Compute a deterministic cache key from XDF path and run_all params; optionally include mtime for invalidation when file changes. Returns 16-char hex digest."""
    canonical_path = str(Path(xdf_path).resolve())
    params_json = json.dumps(params, sort_keys=True)
    if mtime is not None:
        payload = f"{canonical_path}|{mtime}|{params_json}"
    else:
        payload = f"{canonical_path}|{params_json}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_computation_history(cache_root: Path) -> pd.DataFrame:
    """Load the computation history table from CSV. Returns empty DataFrame with correct columns if missing."""
    cache_root = Path(cache_root)
    history_path = cache_root / "computation_history.csv"
    if not history_path.exists():
        return pd.DataFrame(columns=COMPUTATION_HISTORY_COLUMNS)
    try:
        df = pd.read_csv(history_path)
        if list(df.columns) != COMPUTATION_HISTORY_COLUMNS:
            return pd.DataFrame(columns=COMPUTATION_HISTORY_COLUMNS)
        return df
    except Exception:
        return pd.DataFrame(columns=COMPUTATION_HISTORY_COLUMNS)


def _lookup_cached_result(cache_key_hex: str, cache_root: Path, history_df: pd.DataFrame, xdf_path: Optional[Path] = None) -> Optional[Tuple[Path, Optional[str]]]:
    """If key is in history and result blob exists, return (result_path, fif_filename); else None. If xdf_path is given and key lookup misses, try matching by canonical path so entries stored with mtime in the key still hit."""
    if history_df is None or history_df.empty:
        return None

    def _row_to_result(row: pd.Series) -> Optional[Tuple[Path, Optional[str]]]:
        result_path = Path(row["result_path"])
        if not result_path.is_absolute():
            result_path = (cache_root / result_path).resolve()
        if not result_path.exists():
            return None
        fif_filename = row.get("fif_filename")
        if pd.isna(fif_filename) or fif_filename == "":
            fif_filename = None
        else:
            fif_filename = str(fif_filename)
        return (result_path, fif_filename)

    rows = history_df.loc[history_df["cache_key_hex"] == cache_key_hex]
    if not rows.empty:
        out = _row_to_result(rows.iloc[0])
        if out is not None:
            return out
    if xdf_path is not None:
        canonical = str(Path(xdf_path).resolve())
        path_match = history_df.loc[history_df["xdf_path"].astype(str).str.strip() == canonical]
        if not path_match.empty:
            out = _row_to_result(path_match.iloc[0])
            if out is not None:
                return out
    return None


def _load_result_from_cache(result_path: Path) -> dict:
    """Load a serialized result dict from a pickle blob."""
    with open(result_path, "rb") as f:
        return pickle.load(f)


def _save_result_to_cache(cache_key_hex: str, result: dict, xdf_path: Path, mtime: float, params: Dict[str, Any], fif_filename: Optional[str], cache_root: Path, history_lock: threading.Lock) -> None:
    """Write result blob and append one row to the computation history CSV (thread-safe)."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    result_path = cache_root / f"{cache_key_hex}.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Cache save: wrote result to {result_path.resolve().as_posix()}")
    params_json = json.dumps(params, sort_keys=True)
    xdf_path_str = str(Path(xdf_path).resolve())
    result_path_str = str(result_path.resolve())
    fif_str = fif_filename if fif_filename else ""
    computed_at = datetime.now(timezone.utc).isoformat()
    history_path = cache_root / "computation_history.csv"
    new_row = pd.DataFrame([{"cache_key_hex": cache_key_hex, "xdf_path": xdf_path_str, "xdf_mtime": mtime, "params_json": params_json, "result_path": result_path_str, "fif_filename": fif_str, "computed_at": computed_at}])
    with history_lock:
        if history_path.exists():
            new_row.to_csv(history_path, mode="a", header=False, index=False)
        else:
            new_row.to_csv(history_path, mode="w", header=True, index=False)


def compute_session_summary_metrics(active_only_out_eeg_raws, results, stream_infos_df: Optional[pd.DataFrame], output_folder: Path, freq_min: float = 1.0, freq_max: float = 40.0, filename_prefix: str = "") -> Path:
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
        return output_folder.joinpath(f"{filename_prefix}session_summaries_empty.csv")

    metrics_df = pd.DataFrame.from_records(rows)
    csv_path = output_folder.joinpath(f"{filename_prefix}session_summaries.csv")
    metrics_df.to_csv(csv_path, index=False)

    print(f"\nSaved per-session summary metrics to: {csv_path.as_posix()}")
    return csv_path


def _get_channel_bad_intervals(raw: mne.io.BaseRaw, channel_names: List[str]) -> Dict[str, List[Tuple[float, float]]]:
    """Build per-channel bad (onset, end) intervals in seconds from raw.annotations (BAD_* descriptions, optional ch_names)."""
    out: Dict[str, List[Tuple[float, float]]] = {ch: [] for ch in channel_names}
    if raw.annotations is None or len(raw.annotations) == 0:
        return out
    ch_names_attr = getattr(raw.annotations, "ch_names", None)
    for i in range(len(raw.annotations)):
        desc = raw.annotations.description[i]
        if not (isinstance(desc, str) and desc.upper().startswith("BAD_")):
            continue
        onset = float(raw.annotations.onset[i])
        duration = float(raw.annotations.duration[i])
        end = onset + duration
        seg_ch_names = ch_names_attr[i] if (ch_names_attr is not None and i < len(ch_names_attr)) else None
        if not isinstance(seg_ch_names, (list, tuple)) or len(seg_ch_names) == 0:
            for ch in channel_names:
                out[ch].append((onset, end))
        else:
            for ch in seg_ch_names:
                if ch in out:
                    out[ch].append((onset, end))
    return out


def export_session_spectrograms_html(active_only_out_eeg_raws, results, output_folder: Path,
                                     freq_min: float = 1.0, freq_max: float = 40.0, filename_prefix: str = ""):
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
            channel_bad_intervals = _get_channel_bad_intervals(a_raw, list(spectogram_result_dict.keys()))
            
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
                
                bad_intervals = channel_bad_intervals.get(ch_name, [])
                t_min, t_max = float(np.min(t)), float(np.max(t))
                f_min, f_max = float(f_filtered.min()), float(f_filtered.max())
                if bad_intervals:
                    rect_data = []
                    for start, end in bad_intervals:
                        start_c = max(start, t_min)
                        end_c = min(end, t_max)
                        if start_c < end_c:
                            rect_data.append((start_c, f_min, end_c, f_max))
                    if rect_data:
                        rects = hv.Rectangles(rect_data, kdims=['time', 'frequency', 'time', 'frequency']).opts(alpha=0.35, color='red', line_alpha=0)
                        channel_plots.append(img * rects)
                    else:
                        channel_plots.append(img)
                else:
                    channel_plots.append(img)
            
            # Stack all channel plots vertically
            layout = hv.Layout(channel_plots).cols(1)
            
            # Add overall title
            layout = layout.opts(
                title=f'EEG Spectrogram - {session_title}',
                shared_axes=True
            )
            
            # Save to HTML
            html_path = output_folder / f"{filename_prefix}spectrogram_{session_name}.html"
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
            channel_bad_intervals = _get_channel_bad_intervals(a_raw, list(spectogram_result_dict.keys()))
            
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
                
                bad_intervals = channel_bad_intervals.get(ch_name, [])
                t_min, t_max = float(np.min(t)), float(np.max(t))
                f_min, f_max = float(f_filtered.min()), float(f_filtered.max())
                if bad_intervals:
                    rect_data = []
                    for start, end in bad_intervals:
                        start_c = max(start, t_min)
                        end_c = min(end, t_max)
                        if start_c < end_c:
                            rect_data.append((start_c, f_min, end_c, f_max))
                    if rect_data:
                        rects = hv.Rectangles(rect_data, kdims=['time', 'frequency', 'time', 'frequency']).opts(alpha=0.35, color='red', line_alpha=0)
                        channel_plots.append(img * rects)
                    else:
                        channel_plots.append(img)
                else:
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


def _safe_meas_date_sec(raw: Optional[mne.io.BaseRaw]) -> float:
    if raw is None:
        return float("nan")
    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        return float("nan")
    if getattr(meas_date, "tzinfo", None) is None:
        meas_date = meas_date.replace(tzinfo=timezone.utc)
    return float(meas_date.timestamp())


def _extract_raw_export_payload(raw: Optional[mne.io.BaseRaw]) -> Dict[str, Any]:
    if raw is None:
        return {"has_data": False, "raw_data": np.empty((0, 0), dtype=np.float64), "raw_times_sec": np.empty((0,), dtype=np.float64), "channel_names": [], "sfreq_hz": float("nan"), "time_origin_meas_date_sec": float("nan")}
    try:
        raw_data = np.asarray(raw.get_data(), dtype=np.float64)
        raw_times_sec = np.asarray(raw.times, dtype=np.float64)
        channel_names = [str(v) for v in raw.info.get("ch_names", [])]
        sfreq_hz = float(raw.info.get("sfreq", float("nan")))
        return {"has_data": True, "raw_data": raw_data, "raw_times_sec": raw_times_sec, "channel_names": channel_names, "sfreq_hz": sfreq_hz, "time_origin_meas_date_sec": _safe_meas_date_sec(raw)}
    except Exception:
        return {"has_data": False, "raw_data": np.empty((0, 0), dtype=np.float64), "raw_times_sec": np.empty((0,), dtype=np.float64), "channel_names": [], "sfreq_hz": float("nan"), "time_origin_meas_date_sec": float("nan")}


def _compute_nearest_spectrogram_bin_idx(sample_times_sec: np.ndarray, spectrogram_times_sec: np.ndarray, max_samples_for_map: int = 2_000_000) -> Tuple[np.ndarray, bool]:
    sample_times_sec = np.asarray(sample_times_sec, dtype=np.float64)
    spectrogram_times_sec = np.asarray(spectrogram_times_sec, dtype=np.float64)
    if sample_times_sec.size == 0:
        return np.empty((0,), dtype=np.int64), True
    if spectrogram_times_sec.size == 0:
        return np.full(sample_times_sec.shape, -1, dtype=np.int64), True
    if sample_times_sec.size > max_samples_for_map:
        return np.empty((0,), dtype=np.int64), False
    right_idx = np.searchsorted(spectrogram_times_sec, sample_times_sec, side="left")
    right_idx = np.clip(right_idx, 0, spectrogram_times_sec.size - 1)
    left_idx = np.clip(right_idx - 1, 0, spectrogram_times_sec.size - 1)
    left_dist = np.abs(sample_times_sec - spectrogram_times_sec[left_idx])
    right_dist = np.abs(sample_times_sec - spectrogram_times_sec[right_idx])
    nearest = np.where(right_dist < left_dist, right_idx, left_idx).astype(np.int64)
    return nearest, True


def _build_alignment_export_payload(spectrogram_times_sec: np.ndarray, eeg_payload: Dict[str, Any], motion_payload: Dict[str, Any]) -> Dict[str, Any]:
    spectrogram_times_sec = np.asarray(spectrogram_times_sec, dtype=np.float64)
    eeg_nearest_idx, eeg_has_nearest_map = _compute_nearest_spectrogram_bin_idx(eeg_payload["raw_times_sec"], spectrogram_times_sec)
    motion_nearest_idx, motion_has_nearest_map = _compute_nearest_spectrogram_bin_idx(motion_payload["raw_times_sec"], spectrogram_times_sec)
    if spectrogram_times_sec.size > 0:
        spectrogram_start_time_sec = float(spectrogram_times_sec[0])
        spectrogram_end_time_sec = float(spectrogram_times_sec[-1])
    else:
        spectrogram_start_time_sec = float("nan")
        spectrogram_end_time_sec = float("nan")
    return {"alignment_method": "raw_plus_map", "spectrogram_times_sec": spectrogram_times_sec, "spectrogram_start_time_sec": spectrogram_start_time_sec, "spectrogram_end_time_sec": spectrogram_end_time_sec, "eeg_time_origin_meas_date_sec": float(eeg_payload["time_origin_meas_date_sec"]), "motion_time_origin_meas_date_sec": float(motion_payload["time_origin_meas_date_sec"]), "eeg_nearest_spectrogram_bin_idx": eeg_nearest_idx, "motion_nearest_spectrogram_bin_idx": motion_nearest_idx, "eeg_has_nearest_map": bool(eeg_has_nearest_map), "motion_has_nearest_map": bool(motion_has_nearest_map)}


def export_spectrograms_for_rerun(active_only_out_eeg_raws, results, output_dir: Path, freq_min: float = 1.0, freq_max: float = 40.0, filename_prefix: str = "", active_only_out_motion_raws: Optional[List[Optional[mne.io.BaseRaw]]] = None) -> List[Path]:
    """
    Export spectrograms and session datetimes to one NumPy .npz file per session under output_dir for later viewing in Rerun (e.g. via view_spectrograms_rerun.py).

    Each file contains: freq_min, freq_max, session_indices=[0], s0_meas_date_sec, s0_channel_names, s0_freqs, s0_times, s0_Sxx.
    Frequency range is restricted to [freq_min, freq_max]. Each .npz can be passed to view_spectrograms_rerun.py independently.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    used_session_ids: set = set()
    for idx, (a_raw, a_result) in enumerate(zip(active_only_out_eeg_raws, results)):
        try:
            if a_result is None or "spectogram" not in a_result:
                continue
            meas_date = a_raw.info.get("meas_date")
            if meas_date is None:
                meas_date_sec = float("nan")
                session_id = f"session_{idx:03d}"
            else:
                if getattr(meas_date, "tzinfo", None) is None:
                    meas_date = meas_date.replace(tzinfo=timezone.utc)
                meas_date_sec = meas_date.timestamp()
                session_id = meas_date.strftime("%Y-%m-%dT%H-%M-%S")
            base_id = session_id
            while session_id in used_session_ids:
                suffix = sum(1 for s in used_session_ids if s == base_id or (isinstance(s, str) and s.startswith(base_id + "_")))
                session_id = f"{base_id}_{suffix}"
            used_session_ids.add(session_id)
            spectogram_result_dict = a_result["spectogram"]["spectogram_result_dict"]
            channel_names = np.array(list(spectogram_result_dict.keys()), dtype=object)
            f0, t0, Sxx0 = next(iter(spectogram_result_dict.values()))
            freq_mask = (f0 >= freq_min) & (f0 <= freq_max)
            freqs = np.asarray(f0)[freq_mask]
            times = np.asarray(t0)
            Sxx_list = []
            for ch_name in channel_names:
                f, t, Sxx = spectogram_result_dict[ch_name]
                Sxx_list.append(np.asarray(Sxx)[freq_mask, :])
            Sxx_stack = np.stack(Sxx_list, axis=0)
            eeg_payload = _extract_raw_export_payload(a_raw)
            motion_raw = active_only_out_motion_raws[idx] if (active_only_out_motion_raws is not None and idx < len(active_only_out_motion_raws)) else None
            motion_payload = _extract_raw_export_payload(motion_raw)
            alignment_payload = _build_alignment_export_payload(times, eeg_payload, motion_payload)
            export_dict = {"freq_min": np.array(freq_min), "freq_max": np.array(freq_max), "session_indices": np.array([0])}
            export_dict["s0_meas_date_sec"] = np.array(meas_date_sec)
            export_dict["s0_channel_names"] = channel_names
            export_dict["s0_freqs"] = freqs
            export_dict["s0_times"] = times
            export_dict["s0_Sxx"] = Sxx_stack
            export_dict["s0_eeg_raw_data"] = eeg_payload["raw_data"]
            export_dict["s0_eeg_raw_times_sec"] = eeg_payload["raw_times_sec"]
            export_dict["s0_eeg_channel_names"] = np.array(eeg_payload["channel_names"], dtype=object)
            export_dict["s0_has_motion"] = np.array(bool(motion_payload["has_data"]))
            export_dict["s0_motion_raw_data"] = motion_payload["raw_data"]
            export_dict["s0_motion_raw_times_sec"] = motion_payload["raw_times_sec"]
            export_dict["s0_motion_channel_names"] = np.array(motion_payload["channel_names"], dtype=object)
            export_dict["s0_alignment_method"] = np.array(alignment_payload["alignment_method"], dtype=object)
            export_dict["s0_spectrogram_times_sec"] = alignment_payload["spectrogram_times_sec"]
            export_dict["s0_spectrogram_start_time_sec"] = np.array(alignment_payload["spectrogram_start_time_sec"])
            export_dict["s0_spectrogram_end_time_sec"] = np.array(alignment_payload["spectrogram_end_time_sec"])
            export_dict["s0_eeg_time_origin_meas_date_sec"] = np.array(alignment_payload["eeg_time_origin_meas_date_sec"])
            export_dict["s0_motion_time_origin_meas_date_sec"] = np.array(alignment_payload["motion_time_origin_meas_date_sec"])
            export_dict["s0_eeg_has_nearest_map"] = np.array(alignment_payload["eeg_has_nearest_map"])
            export_dict["s0_motion_has_nearest_map"] = np.array(alignment_payload["motion_has_nearest_map"])
            export_dict["s0_eeg_nearest_spectrogram_bin_idx"] = alignment_payload["eeg_nearest_spectrogram_bin_idx"]
            export_dict["s0_motion_nearest_spectrogram_bin_idx"] = alignment_payload["motion_nearest_spectrogram_bin_idx"]
            session_filename = f"{filename_prefix}spectrograms_{session_id}.npz"
            out_path = output_dir / session_filename
            np.savez_compressed(out_path, **export_dict)
            written_paths.append(out_path)
        except Exception as e:
            print(f"  WARN: export_spectrograms_for_rerun skipped session {idx}: {e}")
            continue
    if written_paths:
        print(f"Exported {len(written_paths)} spectrogram .npz file(s) for Rerun to: {output_dir.as_posix()}")
    return written_paths


def export_spectrograms_hdf5(active_only_out_eeg_raws, results, output_path: Path, freq_min: float = 1.0, freq_max: float = 40.0, stream_infos_df: Optional[pd.DataFrame] = None, active_only_out_motion_raws: Optional[List[Optional[mne.io.BaseRaw]]] = None) -> Path:
    """
    Export spectrograms, timestamps, and recording metadata to a single HDF5 file for interchange.

    The output can be read by Python (h5py), MATLAB, R, Julia, and other tools. Each session is stored
    under /sessions/session_XXX with datasets: freqs (Hz), times (s), channel_names, spectrogram
    (n_channels x n_freqs x n_times), and group attributes: meas_date_iso, meas_date_sec, sfreq_hz,
    duration_s, n_channels; if stream_infos_df is provided, xdf_filename is attached per session.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_to_filename: Dict[int, str] = {}
    if stream_infos_df is not None:
        try:
            tmp_df = stream_infos_df.reset_index()
            if "xdf_dataset_idx" in tmp_df.columns and "xdf_filename" in tmp_df.columns:
                for ds_idx, grp in tmp_df.groupby("xdf_dataset_idx"):
                    dataset_to_filename[int(ds_idx)] = str(grp["xdf_filename"].iloc[0])
        except Exception as e:
            print(f"  WARN: export_spectrograms_hdf5 failed to derive dataset -> filename mapping: {e}")
    session_indices: List[int] = []
    with h5py.File(output_path, "w") as f:
        f.attrs["format_version"] = "1.0"
        f.attrs["freq_min"] = float(freq_min)
        f.attrs["freq_max"] = float(freq_max)
        sessions_grp = f.create_group("sessions")
        for idx, (a_raw, a_result) in enumerate(zip(active_only_out_eeg_raws, results)):
            try:
                if a_result is None or "spectogram" not in a_result:
                    continue
                meas_date = a_raw.info.get("meas_date")
                if meas_date is None:
                    meas_date_sec = float("nan")
                    meas_date_iso = ""
                else:
                    if getattr(meas_date, "tzinfo", None) is None:
                        meas_date = meas_date.replace(tzinfo=timezone.utc)
                    meas_date_sec = meas_date.timestamp()
                    meas_date_iso = meas_date.isoformat()
                spectogram_result_dict = a_result["spectogram"]["spectogram_result_dict"]
                channel_names = list(spectogram_result_dict.keys())
                f0, t0, Sxx0 = next(iter(spectogram_result_dict.values()))
                freq_mask = (f0 >= freq_min) & (f0 <= freq_max)
                freqs = np.asarray(f0)[freq_mask]
                times = np.asarray(t0)
                Sxx_list = []
                for ch_name in channel_names:
                    f_ch, t_ch, Sxx = spectogram_result_dict[ch_name]
                    Sxx_list.append(np.asarray(Sxx)[freq_mask, :])
                Sxx_stack = np.stack(Sxx_list, axis=0)
                eeg_payload = _extract_raw_export_payload(a_raw)
                motion_raw = active_only_out_motion_raws[idx] if (active_only_out_motion_raws is not None and idx < len(active_only_out_motion_raws)) else None
                motion_payload = _extract_raw_export_payload(motion_raw)
                alignment_payload = _build_alignment_export_payload(times, eeg_payload, motion_payload)
                try:
                    duration_s = float(a_raw.times[-1] - a_raw.times[0]) if a_raw.times.size > 0 else float("nan")
                except Exception:
                    duration_s = float("nan")
                sfreq_hz = float(a_raw.info.get("sfreq", float("nan")))
                n_channels = len(channel_names)
                sgrp = sessions_grp.create_group(f"session_{idx:03d}")
                sgrp.create_dataset("freqs", data=freqs, dtype=np.float64)
                sgrp.create_dataset("times", data=times, dtype=np.float64)
                dt_str = h5py.special_dtype(vlen=str)
                ch_dset = sgrp.create_dataset("channel_names", (len(channel_names),), dtype=dt_str)
                ch_dset[:] = channel_names
                sgrp.create_dataset("spectrogram", data=Sxx_stack, dtype=np.float64)
                eeg_grp = sgrp.create_group("eeg")
                eeg_grp.create_dataset("raw_data", data=eeg_payload["raw_data"], dtype=np.float64)
                eeg_grp.create_dataset("raw_times_sec", data=eeg_payload["raw_times_sec"], dtype=np.float64)
                eeg_ch_dset = eeg_grp.create_dataset("channel_names", (len(eeg_payload["channel_names"]),), dtype=dt_str)
                if len(eeg_payload["channel_names"]) > 0:
                    eeg_ch_dset[:] = eeg_payload["channel_names"]
                eeg_grp.attrs["sfreq_hz"] = float(eeg_payload["sfreq_hz"])
                eeg_grp.attrs["time_origin_meas_date_sec"] = float(eeg_payload["time_origin_meas_date_sec"])
                motion_grp = sgrp.create_group("motion")
                motion_grp.attrs["has_motion"] = bool(motion_payload["has_data"])
                motion_grp.create_dataset("raw_data", data=motion_payload["raw_data"], dtype=np.float64)
                motion_grp.create_dataset("raw_times_sec", data=motion_payload["raw_times_sec"], dtype=np.float64)
                motion_ch_dset = motion_grp.create_dataset("channel_names", (len(motion_payload["channel_names"]),), dtype=dt_str)
                if len(motion_payload["channel_names"]) > 0:
                    motion_ch_dset[:] = motion_payload["channel_names"]
                motion_grp.attrs["sfreq_hz"] = float(motion_payload["sfreq_hz"])
                motion_grp.attrs["time_origin_meas_date_sec"] = float(motion_payload["time_origin_meas_date_sec"])
                alignment_grp = sgrp.create_group("alignment")
                alignment_grp.attrs["alignment_method"] = alignment_payload["alignment_method"]
                alignment_grp.attrs["eeg_has_nearest_map"] = bool(alignment_payload["eeg_has_nearest_map"])
                alignment_grp.attrs["motion_has_nearest_map"] = bool(alignment_payload["motion_has_nearest_map"])
                alignment_grp.attrs["spectrogram_start_time_sec"] = float(alignment_payload["spectrogram_start_time_sec"])
                alignment_grp.attrs["spectrogram_end_time_sec"] = float(alignment_payload["spectrogram_end_time_sec"])
                alignment_grp.create_dataset("spectrogram_times_sec", data=alignment_payload["spectrogram_times_sec"], dtype=np.float64)
                alignment_grp.create_dataset("eeg_nearest_spectrogram_bin_idx", data=alignment_payload["eeg_nearest_spectrogram_bin_idx"], dtype=np.int64)
                alignment_grp.create_dataset("motion_nearest_spectrogram_bin_idx", data=alignment_payload["motion_nearest_spectrogram_bin_idx"], dtype=np.int64)
                sgrp.attrs["meas_date_iso"] = meas_date_iso
                sgrp.attrs["meas_date_sec"] = meas_date_sec
                sgrp.attrs["sfreq_hz"] = sfreq_hz
                sgrp.attrs["duration_s"] = duration_s
                sgrp.attrs["n_channels"] = n_channels
                sgrp.attrs["has_motion"] = bool(motion_payload["has_data"])
                xdf_filename = dataset_to_filename.get(idx)
                if xdf_filename is not None:
                    sgrp.attrs["xdf_filename"] = str(xdf_filename)
                session_indices.append(idx)
            except Exception as e:
                print(f"  WARN: export_spectrograms_hdf5 skipped session {idx}: {e}")
                continue
        f.attrs["n_sessions"] = len(session_indices)
    print(f"Exported spectrograms to HDF5: {output_path.as_posix()}")
    return output_path


def export_spectrograms_netcdf(active_only_out_eeg_raws, results, output_path: Path, freq_min: float = 1.0, freq_max: float = 40.0, stream_infos_df: Optional[pd.DataFrame] = None, active_only_out_motion_raws: Optional[List[Optional[mne.io.BaseRaw]]] = None) -> Path:
    """
    Export spectrograms, timestamps, and recording metadata to a single CF-friendly NetCDF file for interchange.

    The output can be read by Python (xarray/netCDF4), R, Julia, MATLAB, and other tools. Stores spectrograms
    (session, channel, freq, time), time/freq axes with NaN padding for variable-length sessions, and
    session-level metadata (meas_date_iso, meas_date_sec, sfreq_hz, duration_s, n_channels, xdf_filename).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_to_filename: Dict[int, str] = {}
    if stream_infos_df is not None:
        try:
            tmp_df = stream_infos_df.reset_index()
            if "xdf_dataset_idx" in tmp_df.columns and "xdf_filename" in tmp_df.columns:
                for ds_idx, grp in tmp_df.groupby("xdf_dataset_idx"):
                    dataset_to_filename[int(ds_idx)] = str(grp["xdf_filename"].iloc[0])
        except Exception as e:
            print(f"  WARN: export_spectrograms_netcdf failed to derive dataset -> filename mapping: {e}")
    session_rows: List[Dict[str, Any]] = []
    for idx, (a_raw, a_result) in enumerate(zip(active_only_out_eeg_raws, results)):
        try:
            if a_result is None or "spectogram" not in a_result:
                continue
            meas_date = a_raw.info.get("meas_date")
            if meas_date is None:
                meas_date_sec = float("nan")
                meas_date_iso = ""
            else:
                if getattr(meas_date, "tzinfo", None) is None:
                    meas_date = meas_date.replace(tzinfo=timezone.utc)
                meas_date_sec = meas_date.timestamp()
                meas_date_iso = meas_date.isoformat()
            spectogram_result_dict = a_result["spectogram"]["spectogram_result_dict"]
            channel_names = list(spectogram_result_dict.keys())
            f0, t0, Sxx0 = next(iter(spectogram_result_dict.values()))
            freq_mask = (f0 >= freq_min) & (f0 <= freq_max)
            freqs = np.asarray(f0)[freq_mask]
            times = np.asarray(t0)
            Sxx_list = []
            for ch_name in channel_names:
                f, t, Sxx = spectogram_result_dict[ch_name]
                Sxx_list.append(np.asarray(Sxx)[freq_mask, :])
            Sxx_stack = np.stack(Sxx_list, axis=0)
            eeg_payload = _extract_raw_export_payload(a_raw)
            motion_raw = active_only_out_motion_raws[idx] if (active_only_out_motion_raws is not None and idx < len(active_only_out_motion_raws)) else None
            motion_payload = _extract_raw_export_payload(motion_raw)
            alignment_payload = _build_alignment_export_payload(times, eeg_payload, motion_payload)
            try:
                duration_s = float(a_raw.times[-1] - a_raw.times[0]) if a_raw.times.size > 0 else float("nan")
            except Exception:
                duration_s = float("nan")
            sfreq_hz = float(a_raw.info.get("sfreq", float("nan")))
            xdf_filename = dataset_to_filename.get(idx)
            session_rows.append({"session_idx": idx, "meas_date_iso": meas_date_iso, "meas_date_sec": meas_date_sec, "sfreq_hz": sfreq_hz, "duration_s": duration_s, "n_channels": len(channel_names), "xdf_filename": str(xdf_filename) if xdf_filename is not None else "", "channel_names": channel_names, "freqs": freqs, "times": times, "spectrogram": Sxx_stack, "eeg": eeg_payload, "motion": motion_payload, "alignment": alignment_payload})
        except Exception as e:
            print(f"  WARN: export_spectrograms_netcdf skipped session {idx}: {e}")
            continue
    if not session_rows:
        print("  WARN: export_spectrograms_netcdf had no valid sessions; writing empty NetCDF.")
        ds_empty = xr.Dataset(attrs={"format_version": "1.1", "freq_min": float(freq_min), "freq_max": float(freq_max), "n_sessions": 0})
        ds_empty.to_netcdf(output_path)
        print(f"Exported spectrograms to NetCDF: {output_path.as_posix()}")
        return output_path
    n_sessions = len(session_rows)
    n_channels = max(r["spectrogram"].shape[0] for r in session_rows)
    n_freq = max(r["freqs"].size for r in session_rows)
    n_time = max(r["times"].size for r in session_rows)
    n_eeg_channels = max((r["eeg"]["raw_data"].shape[0] for r in session_rows), default=0)
    n_eeg_samples = max((r["eeg"]["raw_times_sec"].size for r in session_rows), default=0)
    n_motion_channels = max((r["motion"]["raw_data"].shape[0] for r in session_rows), default=0)
    n_motion_samples = max((r["motion"]["raw_times_sec"].size for r in session_rows), default=0)
    spectrogram_padded = np.full((n_sessions, n_channels, n_freq, n_time), np.nan, dtype=np.float64)
    freqs_padded = np.full((n_sessions, n_freq), np.nan, dtype=np.float64)
    times_padded = np.full((n_sessions, n_time), np.nan, dtype=np.float64)
    eeg_raw_data_padded = np.full((n_sessions, n_eeg_channels, n_eeg_samples), np.nan, dtype=np.float64)
    eeg_raw_times_padded = np.full((n_sessions, n_eeg_samples), np.nan, dtype=np.float64)
    motion_raw_data_padded = np.full((n_sessions, n_motion_channels, n_motion_samples), np.nan, dtype=np.float64)
    motion_raw_times_padded = np.full((n_sessions, n_motion_samples), np.nan, dtype=np.float64)
    eeg_nearest_idx_padded = np.full((n_sessions, n_eeg_samples), -1, dtype=np.int64)
    motion_nearest_idx_padded = np.full((n_sessions, n_motion_samples), -1, dtype=np.int64)
    channel_names_padded = np.full((n_sessions, n_channels), "", dtype=object)
    eeg_channel_names_padded = np.full((n_sessions, n_eeg_channels), "", dtype=object)
    motion_channel_names_padded = np.full((n_sessions, n_motion_channels), "", dtype=object)
    for i, r in enumerate(session_rows):
        nc, nf, nt = r["spectrogram"].shape
        spectrogram_padded[i, :nc, :nf, :nt] = r["spectrogram"]
        freqs_padded[i, :r["freqs"].size] = r["freqs"]
        times_padded[i, :r["times"].size] = r["times"]
        eeg_data = r["eeg"]["raw_data"]
        eeg_times = r["eeg"]["raw_times_sec"]
        if eeg_data.ndim == 2 and eeg_data.size > 0:
            eeg_raw_data_padded[i, :eeg_data.shape[0], :eeg_data.shape[1]] = eeg_data
        if eeg_times.size > 0:
            eeg_raw_times_padded[i, :eeg_times.size] = eeg_times
        eeg_idx = r["alignment"]["eeg_nearest_spectrogram_bin_idx"]
        if r["alignment"]["eeg_has_nearest_map"] and eeg_idx.size > 0:
            eeg_nearest_idx_padded[i, :eeg_idx.size] = eeg_idx
        motion_data = r["motion"]["raw_data"]
        motion_times = r["motion"]["raw_times_sec"]
        if motion_data.ndim == 2 and motion_data.size > 0:
            motion_raw_data_padded[i, :motion_data.shape[0], :motion_data.shape[1]] = motion_data
        if motion_times.size > 0:
            motion_raw_times_padded[i, :motion_times.size] = motion_times
        motion_idx = r["alignment"]["motion_nearest_spectrogram_bin_idx"]
        if r["alignment"]["motion_has_nearest_map"] and motion_idx.size > 0:
            motion_nearest_idx_padded[i, :motion_idx.size] = motion_idx
        for j, name in enumerate(r["channel_names"]):
            if j < n_channels:
                channel_names_padded[i, j] = name
        for j, name in enumerate(r["eeg"]["channel_names"]):
            if j < n_eeg_channels:
                eeg_channel_names_padded[i, j] = name
        for j, name in enumerate(r["motion"]["channel_names"]):
            if j < n_motion_channels:
                motion_channel_names_padded[i, j] = name
    meas_date_iso_arr = np.array([r["meas_date_iso"] for r in session_rows], dtype=object)
    meas_date_sec_arr = np.array([r["meas_date_sec"] for r in session_rows], dtype=np.float64)
    sfreq_hz_arr = np.array([r["sfreq_hz"] for r in session_rows], dtype=np.float64)
    duration_s_arr = np.array([r["duration_s"] for r in session_rows], dtype=np.float64)
    n_channels_arr = np.array([r["n_channels"] for r in session_rows], dtype=np.int64)
    xdf_filename_arr = np.array([r["xdf_filename"] for r in session_rows], dtype=object)
    has_motion_arr = np.array([1 if r["motion"]["has_data"] else 0 for r in session_rows], dtype=np.int8)
    alignment_method_arr = np.array([r["alignment"]["alignment_method"] for r in session_rows], dtype=object)
    spectrogram_start_time_sec_arr = np.array([r["alignment"]["spectrogram_start_time_sec"] for r in session_rows], dtype=np.float64)
    spectrogram_end_time_sec_arr = np.array([r["alignment"]["spectrogram_end_time_sec"] for r in session_rows], dtype=np.float64)
    eeg_time_origin_meas_date_sec_arr = np.array([r["alignment"]["eeg_time_origin_meas_date_sec"] for r in session_rows], dtype=np.float64)
    motion_time_origin_meas_date_sec_arr = np.array([r["alignment"]["motion_time_origin_meas_date_sec"] for r in session_rows], dtype=np.float64)
    eeg_has_nearest_map_arr = np.array([1 if r["alignment"]["eeg_has_nearest_map"] else 0 for r in session_rows], dtype=np.int8)
    motion_has_nearest_map_arr = np.array([1 if r["alignment"]["motion_has_nearest_map"] else 0 for r in session_rows], dtype=np.int8)
    ds = xr.Dataset(
        {
            "spectrogram": (("session", "channel", "freq", "time"), spectrogram_padded),
            "freqs": (("session", "freq"), freqs_padded),
            "times": (("session", "time"), times_padded),
            "meas_date_iso": (("session",), meas_date_iso_arr),
            "meas_date_sec": (("session",), meas_date_sec_arr),
            "sfreq_hz": (("session",), sfreq_hz_arr),
            "duration_s": (("session",), duration_s_arr),
            "n_channels": (("session",), n_channels_arr),
            "xdf_filename": (("session",), xdf_filename_arr),
            "channel_names": (("session", "channel"), channel_names_padded),
            "eeg_raw_data": (("session", "eeg_channel", "eeg_sample"), eeg_raw_data_padded),
            "eeg_raw_times_sec": (("session", "eeg_sample"), eeg_raw_times_padded),
            "eeg_channel_names": (("session", "eeg_channel"), eeg_channel_names_padded),
            "has_motion": (("session",), has_motion_arr),
            "motion_raw_data": (("session", "motion_channel", "motion_sample"), motion_raw_data_padded),
            "motion_raw_times_sec": (("session", "motion_sample"), motion_raw_times_padded),
            "motion_channel_names": (("session", "motion_channel"), motion_channel_names_padded),
            "alignment_method": (("session",), alignment_method_arr),
            "spectrogram_start_time_sec": (("session",), spectrogram_start_time_sec_arr),
            "spectrogram_end_time_sec": (("session",), spectrogram_end_time_sec_arr),
            "eeg_time_origin_meas_date_sec": (("session",), eeg_time_origin_meas_date_sec_arr),
            "motion_time_origin_meas_date_sec": (("session",), motion_time_origin_meas_date_sec_arr),
            "eeg_has_nearest_map": (("session",), eeg_has_nearest_map_arr),
            "motion_has_nearest_map": (("session",), motion_has_nearest_map_arr),
            "eeg_nearest_spectrogram_bin_idx": (("session", "eeg_sample"), eeg_nearest_idx_padded),
            "motion_nearest_spectrogram_bin_idx": (("session", "motion_sample"), motion_nearest_idx_padded),
        },
        coords={"session": np.arange(n_sessions), "channel": np.arange(n_channels), "freq": np.arange(n_freq), "time": np.arange(n_time), "eeg_channel": np.arange(n_eeg_channels), "eeg_sample": np.arange(n_eeg_samples), "motion_channel": np.arange(n_motion_channels), "motion_sample": np.arange(n_motion_samples)},
        attrs={"format_version": "1.1", "freq_min": float(freq_min), "freq_max": float(freq_max), "n_sessions": n_sessions},
    )
    ds.to_netcdf(output_path)
    print(f"Exported spectrograms to NetCDF: {output_path.as_posix()}")
    return output_path


def export_spectrograms_parquet(active_only_out_eeg_raws, results, output_path: Path, freq_min: float = 1.0, freq_max: float = 40.0, stream_infos_df: Optional[pd.DataFrame] = None, active_only_out_motion_raws: Optional[List[Optional[mne.io.BaseRaw]]] = None) -> Path:
    """
    Export spectrograms, timestamps, and recording metadata to a single Parquet file for interchange and analytics.

    One row per session with scalar metadata and list/array columns for channel_names, freqs, times, and
    spectrogram (3D as list of list of list). Readable with pandas/PyArrow in Python, R, and other tools.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_to_filename: Dict[int, str] = {}
    if stream_infos_df is not None:
        try:
            tmp_df = stream_infos_df.reset_index()
            if "xdf_dataset_idx" in tmp_df.columns and "xdf_filename" in tmp_df.columns:
                for ds_idx, grp in tmp_df.groupby("xdf_dataset_idx"):
                    dataset_to_filename[int(ds_idx)] = str(grp["xdf_filename"].iloc[0])
        except Exception as e:
            print(f"  WARN: export_spectrograms_parquet failed to derive dataset -> filename mapping: {e}")
    rows: List[Dict[str, Any]] = []
    for idx, (a_raw, a_result) in enumerate(zip(active_only_out_eeg_raws, results)):
        try:
            if a_result is None or "spectogram" not in a_result:
                continue
            meas_date = a_raw.info.get("meas_date")
            if meas_date is None:
                meas_date_sec = float("nan")
                meas_date_iso = ""
            else:
                if getattr(meas_date, "tzinfo", None) is None:
                    meas_date = meas_date.replace(tzinfo=timezone.utc)
                meas_date_sec = meas_date.timestamp()
                meas_date_iso = meas_date.isoformat()
            spectogram_result_dict = a_result["spectogram"]["spectogram_result_dict"]
            channel_names = list(spectogram_result_dict.keys())
            f0, t0, Sxx0 = next(iter(spectogram_result_dict.values()))
            freq_mask = (f0 >= freq_min) & (f0 <= freq_max)
            freqs = np.asarray(f0)[freq_mask]
            times = np.asarray(t0)
            Sxx_list = []
            for ch_name in channel_names:
                f, t, Sxx = spectogram_result_dict[ch_name]
                Sxx_list.append(np.asarray(Sxx)[freq_mask, :])
            Sxx_stack = np.stack(Sxx_list, axis=0)
            eeg_payload = _extract_raw_export_payload(a_raw)
            motion_raw = active_only_out_motion_raws[idx] if (active_only_out_motion_raws is not None and idx < len(active_only_out_motion_raws)) else None
            motion_payload = _extract_raw_export_payload(motion_raw)
            alignment_payload = _build_alignment_export_payload(times, eeg_payload, motion_payload)
            try:
                duration_s = float(a_raw.times[-1] - a_raw.times[0]) if a_raw.times.size > 0 else float("nan")
            except Exception:
                duration_s = float("nan")
            sfreq_hz = float(a_raw.info.get("sfreq", float("nan")))
            xdf_filename = dataset_to_filename.get(idx)
            spectrogram_nested = Sxx_stack.tolist()
            rows.append({"session_idx": idx, "meas_date_iso": meas_date_iso, "meas_date_sec": meas_date_sec, "xdf_filename": str(xdf_filename) if xdf_filename is not None else "", "sfreq_hz": sfreq_hz, "duration_s": duration_s, "n_channels": len(channel_names), "channel_names": channel_names, "freqs": freqs.tolist(), "times": times.tolist(), "spectrogram": spectrogram_nested, "eeg_raw_data": eeg_payload["raw_data"].tolist(), "eeg_raw_times_sec": eeg_payload["raw_times_sec"].tolist(), "eeg_channel_names": eeg_payload["channel_names"], "has_motion": bool(motion_payload["has_data"]), "motion_raw_data": motion_payload["raw_data"].tolist(), "motion_raw_times_sec": motion_payload["raw_times_sec"].tolist(), "motion_channel_names": motion_payload["channel_names"], "alignment_method": alignment_payload["alignment_method"], "spectrogram_times_sec": alignment_payload["spectrogram_times_sec"].tolist(), "spectrogram_start_time_sec": float(alignment_payload["spectrogram_start_time_sec"]), "spectrogram_end_time_sec": float(alignment_payload["spectrogram_end_time_sec"]), "eeg_time_origin_meas_date_sec": float(alignment_payload["eeg_time_origin_meas_date_sec"]), "motion_time_origin_meas_date_sec": float(alignment_payload["motion_time_origin_meas_date_sec"]), "eeg_has_nearest_map": bool(alignment_payload["eeg_has_nearest_map"]), "motion_has_nearest_map": bool(alignment_payload["motion_has_nearest_map"]), "eeg_nearest_spectrogram_bin_idx": alignment_payload["eeg_nearest_spectrogram_bin_idx"].tolist(), "motion_nearest_spectrogram_bin_idx": alignment_payload["motion_nearest_spectrogram_bin_idx"].tolist()})
        except Exception as e:
            print(f"  WARN: export_spectrograms_parquet skipped session {idx}: {e}")
            continue
    df = pd.DataFrame.from_records(rows)
    df.to_parquet(output_path, index=False)
    print(f"Exported spectrograms to Parquet: {output_path.as_posix()}")
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


# db_root_path = Path('/content/drive/MyDrive/Databases')
db_root_path = Path('E:/Dropbox (Personal)/Databases')
assert db_root_path.exists(), f"'{db_root_path.as_posix()}' does not exist!"

# eeg_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif')
# headset_motion_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif')

# assert eeg_recordings_file_path.exists()
# assert headset_motion_recordings_file_path.exists()

eeg_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEpocX_EEGRecordings/fif')
flutter_eeg_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEEG_FlutterRecordings')
flutter_motion_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEEG_FlutterRecordings/MOTION_RECORDINGS')
flutter_GENERIC_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEEG_FlutterRecordings/GENERIC_RECORDINGS')

headset_motion_recordings_file_path: Path = db_root_path.joinpath('UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif')
WhisperVideoTranscripts_LSL_Converted = db_root_path.joinpath('UnparsedData/WhisperVideoTranscripts_LSL_Converted')
pho_log_to_LSL_recordings_path: Path = db_root_path.joinpath('UnparsedData/PhoLogToLabStreamingLayer_logs')
## These contain little LSL .fif files with names like: '20250808_062814_log.fif',

eeg_analyzed_parent_export_path = db_root_path.joinpath('AnalysisData/MNE_preprocessed')
pickled_data_path = db_root_path.joinpath('AnalysisData/MNE_preprocessed/PICKLED_COLLECTION')
pickled_data_path.mkdir(exist_ok=True)
assert pickled_data_path.exists(), f"'{pickled_data_path.as_posix()}' does not exist!"


outputs_root_folder: Path = Path('L:/AITEMP/PhoOfflineEEGAnalysisOutputs')
assert outputs_root_folder.exists(), f"'{outputs_root_folder.as_posix()}' does not exist!"

lab_recorder_output_path = Path("E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001")
assert lab_recorder_output_path.exists(), f"'{lab_recorder_output_path.as_posix()}' does not exist!"


def process_XDFs_main(n_most_recent_sessions_to_preprocess: Optional[int] = 5,
        should_write_final_merged_eeg_fif: bool = True,
        included_xdf_file_names: Optional[List]=None, # Include all
        should_load_preprocessed: bool = False,
        use_computation_cache: bool = True,
        cache_root: Optional[Path] = None,
        use_mtime_in_cache_key: bool = False,
        absolute_max_workers: int = 2,
    ):
    """ Processes a single .xdf file independently to produce all exports

        from PhoOfflineEEGAnalysis.examples_jupyter.main_analyze_run import process_XDFs_main

    """
    if cache_root is None:
        cache_root = pickled_data_path / "computation_cache"
    cache_root = Path(cache_root)
    computation_history_df: pd.DataFrame = _load_computation_history(cache_root) if use_computation_cache else pd.DataFrame(columns=COMPUTATION_HISTORY_COLUMNS)
    history_append_lock: threading.Lock = threading.Lock() if use_computation_cache else None
    run_all_params: Dict[str, Any] = {"mask_bad_annotated_times": False}
    if use_computation_cache:
        history_path = cache_root / "computation_history.csv"
        n_entries = len(computation_history_df)
        print(f"Computation cache: root={cache_root.resolve().as_posix()}, history exists={history_path.exists()}, entries={n_entries}")

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
    cpu_count: int = (os.cpu_count() or 2)
    absolute_max_workers = min(absolute_max_workers, cpu_count)
    max_workers = min(len(lab_recorder_xdf_files), absolute_max_workers)
    print(f"Processing {len(lab_recorder_xdf_files)} XDF files using {max_workers} parallel workers...")

    # Initialize result containers
    _out_eeg_raw = [None] * len(lab_recorder_xdf_files)
    _out_motion_raw = [None] * len(lab_recorder_xdf_files)
    _out_xdf_stream_infos_df = [None] * len(lab_recorder_xdf_files)
    _out_results = [None] * len(lab_recorder_xdf_files)


    def _subfn_process_single_xdf_file(idx_file_tuple):
        """Process a single XDF file - captures lots of things 
        """
        an_xdf_file_idx, a_xdf_file = idx_file_tuple
        try:
            print(f'  Processing XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"...')

            if use_computation_cache:
                try:
                    xdf_mtime = a_xdf_file.stat().st_mtime
                    cache_key_hex = _compute_computation_cache_key(a_xdf_file, run_all_params, mtime=xdf_mtime if use_mtime_in_cache_key else None)
                    looked_up = _lookup_cached_result(cache_key_hex, cache_root, computation_history_df, xdf_path=a_xdf_file)
                    if looked_up is not None:
                        result_path, _ = looked_up
                        print(f'  Cache hit: loading result from {result_path.as_posix()}')
                        result = _load_result_from_cache(result_path)
                        _obj = LabRecorderXDF.init_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file)
                        stream_infos = _obj.stream_infos
                        raws_dict = _obj.datasets_dict
                        eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])
                        motion_raws = raws_dict.get(DataModalityType.MOTION.value, [])
                        if len(eeg_raws) == 0:
                            print(f'  WARN: no EEG streams (cache hit path) in "{a_xdf_file.as_posix()}". Skipping file.')
                            return an_xdf_file_idx, None, None, None, None
                        merged_eeg_raws, merge_meta = LabRecorderXDF.merge_eeg_streams_by_device(eeg_raws=eeg_raws, strict_merge=False, debug_print=False)
                        if len(merged_eeg_raws) == 0:
                            print(f'  WARN: could not produce merged EEG (cache hit path) for "{a_xdf_file.as_posix()}". Skipping file.')
                            return an_xdf_file_idx, None, None, None, None
                        eeg_raw = merged_eeg_raws[0]
                        motion_raw = motion_raws[0] if len(motion_raws) > 0 else None
                        stream_infos['lab_recorder_xdf_file_idx'] = an_xdf_file_idx
                        stream_infos['xdf_filename'] = a_xdf_file.name
                        stream_infos['eeg_device_group_idx'] = 0
                        stream_infos['eeg_device_key'] = merge_meta[0].get('device_key', 'device_0')
                        stream_infos['n_eeg_segments_in_group'] = merge_meta[0].get('n_segments', 1)
                        eeg_raw = up_convert_raw_obj(eeg_raw)
                        if motion_raw is not None:
                            motion_raw = up_convert_raw_obj(motion_raw)
                        EEGData.set_montage(datasets_EEG=[eeg_raw])
                        eeg_raw.debug_test_annotations_timestamps()
                        print(f'  Cache hit for XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"')
                        return an_xdf_file_idx, eeg_raw, motion_raw, stream_infos, result
                except Exception as e:
                    print(f'  Cache lookup/load failed for file {an_xdf_file_idx+1}, recomputing: {e}')

            # Load XDF file
            _obj = LabRecorderXDF.init_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file)
            stream_infos = _obj.stream_infos
            raws = _obj.datasets
            raws_dict = _obj.datasets_dict

            eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])

            if len(eeg_raws) == 0:
                print(f'  WARN: no EEG streams found in "{a_xdf_file.as_posix()}". Skipping file.')
                return an_xdf_file_idx, None, None, None, None

            # Merge by device so we can handle multiple EEG streams per XDF
            merged_eeg_raws, merge_meta = LabRecorderXDF.merge_eeg_streams_by_device(
                eeg_raws=eeg_raws, strict_merge=False, debug_print=False
            )
            if len(merged_eeg_raws) == 0:
                print(f'  WARN: could not produce any merged EEG datasets for "{a_xdf_file.as_posix()}". Skipping file.')
                return an_xdf_file_idx, None, None, None, None

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
            motion_raws = raws_dict.get(DataModalityType.MOTION.value, [])
            motion_raw = motion_raws[0] if len(motion_raws) > 0 else None

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
            if motion_raw is not None:
                motion_raw = up_convert_raw_obj(motion_raw)
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

            if use_computation_cache and result is not None and history_append_lock is not None:
                try:
                    xdf_mtime = a_xdf_file.stat().st_mtime
                    cache_key_hex = _compute_computation_cache_key(a_xdf_file, run_all_params, mtime=xdf_mtime if use_mtime_in_cache_key else None)
                    fif_filename = None
                    for k, v in stream_infos.items():
                        if isinstance(k, str) and "proccessed" in k and "filename" in k and v is not None and str(v).strip() != "":
                            fif_filename = str(v).strip()
                            break
                    _save_result_to_cache(cache_key_hex, result, a_xdf_file, xdf_mtime, run_all_params, fif_filename, cache_root, history_append_lock)
                except Exception as e:
                    print(f"  WARN: failed to save result to cache for file {an_xdf_file_idx+1}: {e}")

            print(f'  Completed XDF file {an_xdf_file_idx+1}/{len(lab_recorder_xdf_files)}: "{a_xdf_file.name}"')
            return an_xdf_file_idx, eeg_raw, motion_raw, stream_infos, result

        except (ValueError, KeyError, AssertionError, TypeError) as e:
            print(f'  ERROR in XDF file {an_xdf_file_idx+1}: {e}\n  Skipping file.')
            return an_xdf_file_idx, None, None, None, None

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
                idx, eeg_raw, motion_raw, stream_infos, result = future.result()
                _out_eeg_raw[idx] = eeg_raw
                _out_motion_raw[idx] = motion_raw
                _out_xdf_stream_infos_df[idx] = stream_infos
                _out_results[idx] = result
            except Exception as e:
                idx = future_to_idx[future]
                print(f"  EXCEPTION collecting result for file {idx+1}: {e}")
                _out_eeg_raw[idx] = None
                _out_motion_raw[idx] = None
                _out_xdf_stream_infos_df[idx] = None
                _out_results[idx] = None



    # Filter out failed files
    valid_indices = [i for i, (raw, info, result) in enumerate(zip(_out_eeg_raw, _out_xdf_stream_infos_df, _out_results)) if raw is not None and info is not None]
    _out_eeg_raw = [_out_eeg_raw[i] for i in valid_indices]
    _out_motion_raw = [_out_motion_raw[i] for i in valid_indices]
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
    _out_motion_raw = [_out_motion_raw[i] for i in sort_indices]
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


    ## OUTPUTS: _out_xdf_stream_infos_df, active_only_out_eeg_raws, active_only_out_motion_raws, results, 

    active_only_out_motion_raws = _out_motion_raw
    return sso, xdf_dataset_indicies, _out_xdf_stream_infos_df, active_only_out_eeg_raws, active_only_out_motion_raws, results


# xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
# xdf_stream_infos_df


if __name__ == "__main__":

    # n_most_recent_sessions_to_preprocess: int = None # None means all sessions
    # n_most_recent_sessions_to_preprocess: int = 55
    # n_most_recent_sessions_to_preprocess: int = 15
    n_most_recent_sessions_to_preprocess: int = 5

    should_load_preprocessed: bool = False
    # should_load_preprocessed: bool = True

    should_write_final_merged_eeg_fif: bool = False
    # should_write_final_merged_eeg_fif: bool = True

    should_export_html_histograms: bool = True

    absolute_max_workers: int = 4

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

    export_date_prefix = datetime.now().strftime("%Y-%m-%d_")

    sso, xdf_dataset_indicies, _out_xdf_stream_infos_df, active_only_out_eeg_raws, active_only_out_motion_raws, results = process_XDFs_main(included_xdf_file_names=included_xdf_file_names, 
                                                                                                                n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess,
                                                                                                                should_write_final_merged_eeg_fif=should_write_final_merged_eeg_fif,
                                                                                                                should_load_preprocessed=should_load_preprocessed,
                                                                                                                absolute_max_workers=absolute_max_workers,
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
        filename_prefix=export_date_prefix,
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

    html_output_folder = None
    if should_export_html_histograms:
        # Export interactive HTML spectrograms
        print('\nExporting interactive HTML spectrograms...')
        html_output_folder = outputs_root_folder.joinpath('spectrograms_html')
        print(f'\texporting to "{html_output_folder}"...')
        html_files = export_session_spectrograms_html(
            active_only_out_eeg_raws=active_only_out_eeg_raws,
            results=results,
            output_folder=html_output_folder,
            freq_min=1.0,
            freq_max=40.0,
            filename_prefix=export_date_prefix,
        )
        print(f'\tdone.')


    # Export spectrograms + datetime for Rerun (view in another process: python view_spectrograms_rerun.py <path.npz> or rerun <path.rrd>)
    spectrograms_npz_dir = outputs_root_folder
    spectrograms_npz_paths = None
    spectrograms_h5_path = None
    spectrograms_nc_path = None
    spectrograms_parquet_path = None
    try:
        spectrograms_npz_paths = export_spectrograms_for_rerun(active_only_out_eeg_raws=active_only_out_eeg_raws, results=results, output_dir=spectrograms_npz_dir, freq_min=1.0, freq_max=40.0, filename_prefix=export_date_prefix, active_only_out_motion_raws=active_only_out_motion_raws)
    except Exception as e:
        print(f"  Export failed (Rerun .npz): {e}")
        spectrograms_npz_paths = None
    try:
        spectrograms_h5_path = outputs_root_folder.joinpath(f"{export_date_prefix}spectrograms_export.h5")
        export_spectrograms_hdf5(active_only_out_eeg_raws=active_only_out_eeg_raws, results=results, output_path=spectrograms_h5_path, freq_min=1.0, freq_max=40.0, stream_infos_df=_out_xdf_stream_infos_df, active_only_out_motion_raws=active_only_out_motion_raws)
    except Exception as e:
        print(f"  Export failed (HDF5): {e}")
        spectrograms_h5_path = None
    try:
        spectrograms_nc_path = outputs_root_folder.joinpath(f"{export_date_prefix}spectrograms_export.nc")
        export_spectrograms_netcdf(active_only_out_eeg_raws=active_only_out_eeg_raws, results=results, output_path=spectrograms_nc_path, freq_min=1.0, freq_max=40.0, stream_infos_df=_out_xdf_stream_infos_df, active_only_out_motion_raws=active_only_out_motion_raws)
    except Exception as e:
        print(f"  Export failed (NetCDF): {e}")
        spectrograms_nc_path = None
    try:
        spectrograms_parquet_path = outputs_root_folder.joinpath(f"{export_date_prefix}spectrograms_export.parquet")
        export_spectrograms_parquet(active_only_out_eeg_raws=active_only_out_eeg_raws, results=results, output_path=spectrograms_parquet_path, freq_min=1.0, freq_max=40.0, stream_infos_df=_out_xdf_stream_infos_df, active_only_out_motion_raws=active_only_out_motion_raws)
    except Exception as e:
        print(f"  Export failed (Parquet): {e}")
        spectrograms_parquet_path = None

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
    if should_export_html_histograms:
        print(f'Individual HTML spectrograms: {html_output_folder} ({len(html_files)} files)')
    print(f'Session summary metrics CSV: {summary_csv_path}')

    # 'python .\\view_spectrograms_rerun.py "L:\AITEMP\PhoOfflineEEGAnalysisOutputs\2026-02-19_spectrograms_2026-02-18T20-43-55.npz" --spawn'
    # print(f'Spectrograms for Rerun: {spectrograms_npz_dir if spectrograms_npz_paths else "failed"}{f" ({len(spectrograms_npz_paths)} .npz files)" if spectrograms_npz_paths else ""} (run: uv run --project rerun -- python rerun/view_spectrograms_rerun.py "{spectrograms_npz_dir if spectrograms_npz_paths}" then open the .rrd with rerun)')
    if spectrograms_npz_paths:
        print(f'Spectrograms for Rerun: {spectrograms_npz_paths} {f" ({len(spectrograms_npz_paths)} .npz files)" if (len(spectrograms_npz_paths) > 1) else ""}')
        print(f'\tuv run --project rerun -- python rerun/view_spectrograms_rerun.py "{spectrograms_npz_dir}"')
    else:
        print(f'Spectrograms for Rerun: failed')

    print(f'Spectrograms HDF5 (interchange): {spectrograms_h5_path if spectrograms_h5_path else "failed"}')
    print(f'Spectrograms NetCDF (interchange): {spectrograms_nc_path if spectrograms_nc_path else "failed"}')
    print(f'Spectrograms Parquet (interchange): {spectrograms_parquet_path if spectrograms_parquet_path else "failed"}')
    # print(f'Combined HTML spectrogram: {combined_html_file}')
