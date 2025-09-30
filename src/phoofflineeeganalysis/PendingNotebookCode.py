from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from phoofflineeeganalysis.analysis.EEG_data import EEGComputations, EEGData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["figure.titlesize"] = 8
plt.rcParams["axes.titlepad"] = 0
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.constrained_layout.h_pad"] = 0.0
plt.rcParams["figure.constrained_layout.w_pad"] = 0.0
plt.rcParams["figure.constrained_layout.hspace"] = 0.0
plt.rcParams["figure.constrained_layout.wspace"] = 0.0
plt.rcParams["figure.subplot.wspace"] = 0.0
plt.rcParams["figure.subplot.hspace"] = 0.0
plt.rcParams["figure.subplot.wspace"] = 0.0
plt.rcParams["figure.subplot.hspace"] = 0.0



def plot_scrollable_spectogram(ds_disk, channels_to_select=None, fig_export_path="spectrogram_sessions.html"):
    """ 
    Plot EEG spectrograms with scrollable/zoomable time axis,
    bandpower histograms, and selected channel spectrograms.

    Usage:
        from phoofflineeeganalysis.PendingNotebookCode import plot_scrollable_spectogram

        _out = plot_scrollable_spectogram(ds_disk=partial_sess_ds, channels_to_select = ['AF3', 'F7', 'F3', 'FC5'], fig_export_path="spectrogram_sessions.html")
        _out.show()

    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if channels_to_select is None:
        channels_to_select = []
    else:
        # Ensure all requested channels exist in the dataset
        available_channels = set(str(ch) for ch in ds_disk['channels'].values)
        missing = [ch for ch in channels_to_select if ch not in available_channels]
        if missing:
            raise ValueError(f"Channels not found in dataset: {missing}")

    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-13 Hz)": (8, 13),
        "Beta (13-30 Hz)": (13, 30),
        "Gamma (30-64 Hz)": (30, 64)
    }

    n_sessions = len(ds_disk.session)
    n_extra_channels = len(channels_to_select)
    n_extra_cols = 2 * n_extra_channels
    # The column_widths must match the number of columns
    # Always: [0.6, 0.2] for avg, then for each channel: [0.6, 0.2]
    column_widths = [0.6, 0.2] + [0.6, 0.2] * n_extra_channels

    fig = make_subplots(
        rows=n_sessions, cols=(2 + n_extra_cols),
        column_widths=column_widths,
        shared_xaxes=False,
        subplot_titles=[
            f"{s} ({ds_disk['cognitive_status'].values[i]})"
            for i, s in enumerate(ds_disk.session.values)
        ]
    )

    # Extract data variable (power values)
    data = ds_disk["__xarray_dataarray_variable__"]

    for i, session in enumerate(ds_disk.session.values):
        avg_spectrogram = data.sel(session=session).mean(dim="channels")
        time = ds_disk["times"].values
        freqs = ds_disk["freqs"].values
        Z = 10 * np.log10(avg_spectrogram.values)

        # --- 1) Add spectrogram (all channels avg)
        fig.add_trace(
            go.Heatmap(
                z=Z, x=time, y=freqs,
                colorscale="Viridis",
                colorbar=dict(title="Power (dB)"),
            ),
            row=i+1, col=1
        )

        # --- 2) Bandpower histograms
        band_means = []
        for low, high in bands.values():
            band_data = avg_spectrogram.sel(freqs=slice(low, high)).mean().values
            band_means.append(10 * np.log10(band_data))

        fig.add_trace(
            go.Bar(
                x=band_means, y=list(bands.keys()),
                orientation="h", marker_color="steelblue"
            ),
            row=i+1, col=2
        )

        # --- 3) Individual channel spectrograms (optional)
        for ch_idx, ch in enumerate(channels_to_select):
            # Defensive: ensure channel exists
            if ch not in ds_disk['channels'].values:
                raise ValueError(f"Channel '{ch}' not found in dataset channels: {ds_disk['channels'].values}")

            ch_spec = data.sel(session=session, channels=ch)
            Z_ch = 10 * np.log10(ch_spec.values)
            # Each channel gets two columns: spectrogram, then bandpower
            ch_col_spec = 3 + 2 * ch_idx
            ch_col_band = 3 + 2 * ch_idx + 1

            fig.add_trace(
                go.Heatmap(
                    z=Z_ch, x=time, y=freqs,
                    colorscale="Viridis",
                    showscale=False,  # hide duplicate colorbars
                    name=f"{ch} ({session})"
                ),
                row=i+1, col=ch_col_spec
            )

            # --- Bandpower histograms for this channel
            band_means = []
            for low, high in bands.values():
                band_data = ch_spec.sel(freqs=slice(low, high)).mean().values
                band_means.append(10 * np.log10(band_data))

            fig.add_trace(
                go.Bar(
                    x=band_means, y=list(bands.keys()),
                    orientation="h", marker_color="steelblue"
                ),
                row=i+1, col=ch_col_band
            )

    # Layout adjustments
    fig.update_layout(
        height=900, width=1800,
        title="EEG Session Spectrograms with Bandpower + Selected Channels",
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
    )

    if fig_export_path is not None:
        print(f'exporting figure to "{fig_export_path}"...')
        fig.write_html(fig_export_path)

    return fig



def batch_compute_all_eeg_datasets(eeg_raws, limit_num_items: Optional[int]=None, max_workers: Optional[int]=None):
    """ 
    Compute EEG computations for all datasets in parallel using ThreadPoolExecutor.

    Args:
        eeg_raws: List of MNE Raw EEG datasets
        limit_num_items: Limit processing to the last N items (None for all)
        max_workers: Maximum number of parallel threads (None for auto-detect)

    Returns:
        List of computation results for each dataset

    Usage:
        # Basic parallel processing with automatic worker detection:
        from phoofflineeeganalysis.PendingNotebookCode import compute_all_eeg_comps
        results = compute_all_eeg_comps(eeg_raws=my_eeg_datasets)
        print(f"Processed {len(results)} datasets")

        # Process only the last 5 datasets using 4 workers:
        results = compute_all_eeg_comps(
            eeg_raws=my_eeg_datasets, 
            limit_num_items=5, 
            max_workers=4
        )

        # Performance comparison with sequential version:
        import time
        start = time.time()
        par_results = compute_all_eeg_comps(eeg_datasets, max_workers=4)
        par_time = time.time() - start
        print(f"Parallel processing: {par_time:.2f}s")

        # Error handling and validation:
        results = compute_all_eeg_comps(eeg_datasets)
        successful_count = sum(1 for r in results if r is not None)
        print(f"Successfully processed {successful_count}/{len(results)} datasets")
        failed_indices = [i for i, r in enumerate(results) if r is None]
        if failed_indices:
            print(f"Failed computations at indices: {failed_indices}")
    """
    ## INPUT: fixed_len_epochs
    freqs = np.arange(5., 40., 1.0)
    # Define frequencies and number of cycles
    # freqs = np.logspace(*np.log10([2, 40]), num=20)
    n_cycles = freqs / 2.0 # A common approach is to use a fixed number of cycles or a value that increases with frequency.

    [a_raw.info.get('meas_date') for a_raw in eeg_raws]

    if limit_num_items is not None:
        limit_num_items = min(limit_num_items, len(eeg_raws))
        active_only_out_eeg_raws = eeg_raws[-limit_num_items:] ## get only the last 3 items
    else:
        active_only_out_eeg_raws = eeg_raws ## get all items
    # ## OUT: _active_only_out_eeg_raw
    # [a_raw.info.get('meas_date') for a_raw in _active_only_out_eeg_raw]

    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(len(active_only_out_eeg_raws), (os.cpu_count() or 4))
    
    print(f"Processing {len(active_only_out_eeg_raws)} EEG datasets using {max_workers} parallel workers...")

    # Parallel processing using ThreadPoolExecutor
    active_all_outputs_dict = [None] * len(active_only_out_eeg_raws)
    
    def process_single_eeg(idx_raw_tuple):
        """Process a single EEG dataset"""
        idx, a_raw = idx_raw_tuple
        try:
            meas_date = a_raw.info.get('meas_date', 'Unknown')
            print(f"  Processing dataset {idx+1}/{len(active_only_out_eeg_raws)} (meas_date: {meas_date})")
            result = EEGComputations.run_all(raw=a_raw)
            print(f"  Completed dataset {idx+1}/{len(active_only_out_eeg_raws)} (meas_date: {meas_date})")
            return idx, result
        except Exception as e:
            print(f"  ERROR processing dataset {idx+1}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_eeg, (idx, raw)): idx 
            for idx, raw in enumerate(active_only_out_eeg_raws)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                idx, result = future.result()
                active_all_outputs_dict[idx] = result
            except Exception as e:
                idx = future_to_idx[future]
                print(f"  EXCEPTION in dataset {idx+1}: {e}")
                active_all_outputs_dict[idx] = None

    print(f"Completed processing all {len(active_only_out_eeg_raws)} EEG datasets.")
    return active_only_out_eeg_raws, active_all_outputs_dict



def save_all_to_HDF5(_active_only_out_eeg_raw, _active_all_outputs_dict, hdf5_out_path: Path):
    """
        hdf5_out_path: Path = Path('E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/outputs').joinpath('2025-09-22_eegComputations.h5').resolve()
    hdf5_out_path
    """

    for idx, (a_raw, a_raw_outputs) in enumerate(zip(_active_only_out_eeg_raw, _active_all_outputs_dict)):
        # a_path: Path = Path(a_raw.filenames[0])
        # basename: str = a_path.stem
        # basename: str = a_raw.info.get('meas_date')
        src_file_path: Path = Path(a_raw.info.get('description')).resolve()
        basename: str = src_file_path.stem

        print(f'basename: {basename}')

        for an_output_key, an_output_dict in a_raw_outputs.items():
            for an_output_subkey, an_output_value in an_output_dict.items():
                final_data_key: str = '/'.join([basename, an_output_key, an_output_subkey])
                print(f'\tfinal_data_key: "{final_data_key}"')
                # all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)

        # spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
        # fs = a_raw_outputs['spectogram']['fs']

        # for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
        #     all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)
        #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)

        #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)


def plot_session_spectogram(a_raw, a_raw_outputs, sync_to_mne_raw_fig = None):
    """ Plots the spectograms computed in `compute_all_eeg_comps(...)` for each raw EEG session in a separate matplotlib figure, with each channel being a separate row subplot. 

    Usage:
        ## Plot a synchronized EEG Raw data and Spectogram Figure together:
        mne_raw_fig = active_only_out_eeg_raws[0].plot(time_format='datetime', scalings='auto') # MNEBrowseFigure
        fig, axs = plot_session_spectogram(active_only_out_eeg_raws[0], results[0], sync_to_mne_raw_fig=mne_raw_fig)

    """


    basename: str = a_raw.info.get('meas_date')
    fig_identifier: str = f'spectrogram_all_channels[{basename}]'

    fig, axs = plt.subplots(nrows=len(a_raw.info.ch_names), ncols=1, num='spectrogram_all_channels', sharey=True, sharex=True, clear=True)

    spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
    fs = a_raw_outputs['spectogram']['fs']

    for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
        f, t, Sxx = a_ch_spect_result_tuple
        an_ax = axs[ch_idx]
        an_ax.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='auto')
        # plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='auto', ax=an_ax)
        # an_ax.ylims([1, 40])
        # ch_label: str = f"{a_ch}[{ch_idx}]"
        ch_label: str = f"{a_ch}"
        # plt.ylabel(f"{a_ch}[{ch_idx}]", ax=an_ax)
        an_ax.set_ylabel(ch_label)
        
    fig.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)  # remove spacing
    
    if sync_to_mne_raw_fig is not None:
        def on_mne_xlim_changed(event_ax):
            xlim = event_ax.get_xlim()
            ## only need to sync one because share_x=True
            spectogram_ax = axs[0]
            # for ax in axs:
            spectogram_ax.set_xlim(xlim)
            spectogram_ax.figure.canvas.draw_idle()


        # connect to each MNE channel axis
        # for ch_ax in mne_raw_fig.axes:  # mne_fig is your MNEBrowseFigure instance
        #     ch_ax.callbacks.connect('xlim_changed', on_mne_xlim_changed)

        ## only need to do one again because they're synced
        ch_ax = sync_to_mne_raw_fig.axes[0]
        ch_ax.callbacks.connect('xlim_changed', on_mne_xlim_changed)
        print(f'set-up synchronization between MNE raw plot and spectogram axes')

    return fig, axs


def plot_all_spectograms(_active_only_out_eeg_raw, _active_all_outputs_dict):
    """ Plots the spectograms computed in `compute_all_eeg_comps(...)` for each raw EEG session in a separate matplotlib figure, with each channel being a separate row subplot. 

    Usage:
        plot_all_spectograms(active_only_out_eeg_raws, results)
    """
    _out_figs_dict = {}

    plt.close('all')

    for idx, (a_raw, a_raw_outputs) in enumerate(zip(_active_only_out_eeg_raw, _active_all_outputs_dict)):
        # a_path: Path = Path(a_raw.filenames[0])
        # basename: str = a_path.stem
        basename: str = a_raw.info.get('meas_date')
        fig_identifier: str = f'spectrogram_all_channels[{idx}]: {basename}'

        fig, axs = plt.subplots(nrows=len(a_raw.info.ch_names), ncols=2, num=fig_identifier,
                                sharey='row', width_ratios=[12, 1])
        _out_figs_dict[fig_identifier] = {'fig': fig, 'axs': axs}

        spectogram_result_dict =a_raw_outputs['spectogram']['spectogram_result_dict']
        fs =a_raw_outputs['spectogram']['fs']

        for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
            f, t, Sxx = a_ch_spect_result_tuple
            an_ax = axs[ch_idx][0]
            v =  10*np.log10(Sxx+1e-12)
            an_ax.pcolormesh(t, f, v, shading='auto')
            # plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='auto', ax=an_ax)
            plt.ylim([1, 40])
            # plt.ylabel(f"{a_ch}[{ch_idx}]", ax=an_ax)
            an_ax.set_ylabel(f"{a_ch}[{ch_idx}]")

            a_summary_ax = axs[ch_idx][1]
            summary_v = np.atleast_2d(np.nansum(v, axis=0))
            a_summary_ax.imshow(summary_v)        
            # a_summary_ax.pcolormesh([0.0, 1.0], f, summary_v, shading='auto')



def render_all_spectograms_to_high_quality_pdfs(
    _active_only_out_eeg_raw,
    _active_all_outputs_dict,
    output_parent_folder: Optional[Path] = None,
    mode: str = "paged",
    seconds_per_page: float = 120.0,
    freq_min_hz: float = 1.0,
    freq_max_hz: float = 40.0,
    cmap: str = "viridis",
    dpi: int = 300,
    include_channel_summary: bool = True,
    page_size_inches: Tuple[float, float] = (14.0, 8.5),
    tight_layout: bool = True,
    annotate_file_info: bool = True,
    debug_print: bool = False,
):
    """
    Render spectrograms for each EEG recording into high-quality PDFs suitable for tablet viewing.

    Produces either:
    - Paged PDFs (default): Each page covers a time window (seconds_per_page) with all channels stacked.
    - Long single-page PDFs (mode="long"): Horizontal layouts can be very large; use with caution for filesize/performance.

    Args:
        _active_only_out_eeg_raw: List of mne.io.Raw for the EEG sessions rendered in order
        _active_all_outputs_dict: List of per-session results from compute_all_eeg_comps, each containing spectrogram outputs
        output_parent_folder: Parent folder to write PDFs (default: <cwd>/spectrogram_exports/yyyy-mm-dd)
        mode: "paged" (default) or "long"
        seconds_per_page: Window length per page when in paged mode
        freq_min_hz: Minimum frequency rendered
        freq_max_hz: Maximum frequency rendered
        cmap: Matplotlib colormap name
        dpi: Render DPI for figures
        include_channel_summary: If True, add a thin side panel per channel with the time-collapsed energy
        page_size_inches: Tuple(width, height) in inches
        tight_layout: If True, call tight_layout() per page
        annotate_file_info: If True, add filename, meas_date, timezone, and time range to headers
        debug_print: Verbose printing

    Returns:
        List[Path]: Paths to written PDF files (one per session)

    Usage:
        # Compute results first
        active_only_out_eeg_raws, results = compute_all_eeg_comps(eeg_raws=my_eeg_datasets, limit_num_items=3)

        # Render to PDFs (paged)
        from pathlib import Path
        out_paths = render_all_spectograms_to_high_quality_pdfs(
            active_only_out_eeg_raws,
            results,
            output_parent_folder=Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exports"),
            mode="paged",
            seconds_per_page=180.0,
            freq_min_hz=1.0,
            freq_max_hz=40.0,
            dpi=300
        )
        print(f"Wrote {len(out_paths)} PDF(s)")

        # Render as a single long page (may produce large files)
        out_paths_long = render_all_spectograms_to_high_quality_pdfs(
            active_only_out_eeg_raws,
            results,
            mode="long",
            seconds_per_page=0,  # ignored in long mode
        )

    """
    if output_parent_folder is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_parent_folder = Path.cwd().joinpath("spectrogram_exports", today).resolve()
    output_parent_folder.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []

    for idx, (a_raw, a_raw_outputs) in enumerate(zip(_active_only_out_eeg_raw, _active_all_outputs_dict)):
        if a_raw_outputs is None:
            if debug_print:
                print(f"Skipping idx {idx}: no outputs present")
            continue

        spectogram_result_dict = a_raw_outputs.get('spectogram', {}).get('spectogram_result_dict', None)
        fs = a_raw_outputs.get('spectogram', {}).get('fs', None)
        if (spectogram_result_dict is None) or (fs is None):
            if debug_print:
                print(f"Skipping idx {idx}: spectrogram results missing")
            continue

        # Build session identifiers
        try:
            src_file_path: Optional[Path] = None
            description_val = a_raw.info.get('description', None)
            if description_val is not None:
                try:
                    src_file_path = Path(description_val).resolve()
                except Exception:
                    src_file_path = None
            basename: str = src_file_path.stem if (src_file_path is not None and src_file_path.exists()) else f"EEG_{idx:03d}"
        except Exception:
            basename = f"EEG_{idx:03d}"

        meas_date = a_raw.info.get('meas_date', None)
        tz = None
        if isinstance(meas_date, datetime):
            tz = meas_date.tzinfo or timezone.utc
        # Fallbacks
        safe_meas_date_str = str(meas_date) if isinstance(meas_date, datetime) else "Unknown"

        # Create output path per session
        out_pdf_path: Path = output_parent_folder.joinpath(f"{basename}_spectrogram.pdf").resolve()

        # Build time axis from one channel's spectrogram (assume consistent t across channels)
        # Choose the first entry deterministically
        first_key = next(iter(spectogram_result_dict.keys()))
        f_base, t_base, Sxx_base = spectogram_result_dict[first_key]

        # Frequency mask
        f_mask = (f_base >= float(freq_min_hz)) & (f_base <= float(freq_max_hz))
        f_plot = f_base[f_mask]

        # Map samples to absolute times if meas_date is available
        def format_time_label(seconds_since_start: float) -> str:
            if isinstance(meas_date, datetime):
                actual_ts = meas_date + timedelta(seconds=float(seconds_since_start))
                return actual_ts.astimezone(tz or timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            else:
                return f"t+{seconds_since_start:.2f}s"

        # Determine paging
        total_duration_s = float(t_base[-1] - t_base[0]) if len(t_base) > 1 else float(t_base[-1])
        if mode.lower() == "paged":
            if seconds_per_page is None or seconds_per_page <= 0:
                seconds_per_page = 120.0
            num_pages = max(1, int(np.ceil(total_duration_s / float(seconds_per_page))))
        else:
            num_pages = 1  # long

        if debug_print:
            print(f"Session {idx}: basename={basename}, duration={total_duration_s:.2f}s, pages={num_pages}, mode={mode}")

        with PdfPages(out_pdf_path.as_posix()) as pdf:
            for page_idx in range(num_pages):
                # Page bounds in seconds (relative to start of t_base)
                if mode.lower() == "paged":
                    start_s = page_idx * seconds_per_page
                    end_s = min(total_duration_s, (page_idx + 1) * seconds_per_page)
                else:  # long
                    start_s = 0.0
                    end_s = total_duration_s

                # Time mask for current page
                t_mask = (t_base >= (t_base[0] + start_s)) & (t_base <= (t_base[0] + end_s))
                t_plot = t_base[t_mask] - t_base[0]  # normalize to 0 at start of session

                # Prepare figure
                n_channels = len(spectogram_result_dict)
                ncols = 2 if include_channel_summary else 1
                width_ratios = [12, 1] if include_channel_summary else [1]

                fig = plt.figure(figsize=page_size_inches, dpi=dpi)
                axs = fig.subplots(nrows=n_channels, ncols=ncols, sharey='row', width_ratios=width_ratios)
                if n_channels == 1:
                    axs = np.atleast_2d(axs)  # unify indexing

                # Header text (file info)
                if annotate_file_info:
                    header_lines = [f"Session: {basename}"]
                    header_lines.append(f"Measured: {safe_meas_date_str}")
                    if mode.lower() == "paged":
                        header_lines.append(f"Page {page_idx+1}/{num_pages}, Window: {format_time_label(start_s)} → {format_time_label(end_s)}")
                    else:
                        header_lines.append(f"Full Duration: 0.00s → {total_duration_s:.2f}s")
                    fig.suptitle(" | ".join(header_lines), fontsize=12)

                # Render each channel
                for ch_row_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
                    f, t, Sxx = a_ch_spect_result_tuple
                    # Apply masks
                    f_m = (f >= float(freq_min_hz)) & (f <= float(freq_max_hz))
                    f_use = f[f_m]
                    t_m = (t >= (t_base[0] + start_s)) & (t <= (t_base[0] + end_s))
                    t_use = t[t_m] - t_base[0]
                    S = Sxx[f_m][:, t_m]

                    power_db = 10.0 * np.log10(S + 1e-12)

                    ax_main = axs[ch_row_idx][0] if include_channel_summary else axs[ch_row_idx]
                    mesh = ax_main.pcolormesh(t_use, f_use, power_db, shading='auto', cmap=cmap)
                    ax_main.set_ylim([float(freq_min_hz), float(freq_max_hz)])
                    ax_main.set_ylabel(f"{a_ch}")
                    if ch_row_idx == n_channels - 1:
                        # Label times only on the last channel row to save space
                        ax_main.set_xlabel("Time (s)")
                    # Convert a few tick labels to human readable times if possible
                    if isinstance(meas_date, datetime):
                        xticks = ax_main.get_xticks()
                        ax_main.set_xticklabels([format_time_label(float(x)) for x in xticks], rotation=0, fontsize=8)

                    # Optional summary side panel
                    if include_channel_summary:
                        ax_side = axs[ch_row_idx][1]
                        # collapse across frequency (sum of power over freq per time)
                        summary_v = np.atleast_2d(np.nansum(power_db, axis=0))
                        ax_side.imshow(summary_v, aspect='auto', cmap=cmap)
                        ax_side.set_xticks([])
                        ax_side.set_yticks([])

                # Colorbar for the last axis
                cbar = fig.colorbar(mesh, ax=axs.ravel().tolist(), shrink=0.98, pad=0.01)
                cbar.set_label('Power (dB)')

                if tight_layout:
                    plt.tight_layout(rect=[0, 0, 1, 0.95] if annotate_file_info else None)

                pdf.savefig(fig)
                plt.close(fig)

        written_paths.append(out_pdf_path)
        if debug_print:
            print(f"Wrote: {out_pdf_path.as_posix()}")

    return written_paths
