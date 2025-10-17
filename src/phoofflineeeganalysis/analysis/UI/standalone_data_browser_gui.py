# standalone_data_browser_gui.py
"""
Standalone results/data browser using Panel + HoloViews (+ Datashader when appropriate).

This app renders a four-panel layout:
- Left sidebar: session pager (list + prev/next) for browsing sessions
- Right sidebar: display/plot options
- Bottom panel: comments file viewer with time-linked seeking
- Main panel: synchronized timeline with raw EEG, chosen result plots, and a comment track

Usage examples:
    panel serve standalone_data_browser_gui.py --show

Programmatic:
    from phoofflineeeganalysis.analysis.UI.standalone_data_browser_gui import make_app
    app = make_app(ds=dataset, comments_df=comments)
    pn.serve(app, show=True)
"""

import numpy as np
import xarray as xr
import dask.array as da
import holoviews as hv
import holoviews.operation.datashader as hd
import panel as pn
import param
import pandas as pd
from typing import Optional, List, Tuple
from holoviews import opts
from holoviews.streams import Selection1D

hv.extension('bokeh')
pn.extension()

DEFAULT_BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma (30-64 Hz)": (30, 64),
}

def _to_db(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(x, floor))

class StandaloneResultsBrowserApp(param.Parameterized):
    """
    Four-panel standalone browser for reviewing EEG sessions and derived results.
    Expects an xarray Dataset with dims at least:
      - session
      - channels
      - times (seconds)

    And optionally data variables for results (e.g., spectrograms) shaped with freqs x times.
    Comments are provided as a pandas DataFrame with columns at minimum: ['time', 'text'].
    """

    ds = param.Parameter(doc="xarray Dataset with at least dims (session, channels, times)")
    comments_df = param.DataFrame(precedence=-1)

    # UI state
    session = param.ObjectSelector(doc="Active session")
    channels = param.List(doc="Selected channels for raw display")
    time_window = param.Tuple(default=(0.0, 10.0), doc="Visible time window (s)")
    colormap = param.ObjectSelector(default='Viridis', objects=['Viridis', 'Inferno', 'Magma', 'Plasma', 'Turbo'])
    show_results = param.ListSelector(default=['Spectrogram'], objects=['Spectrogram', 'Bandpower'])
    export_filename = param.String(default='spectrogram_sessions.html')

    def __init__(self, ds: xr.Dataset, comments_df: Optional[pd.DataFrame] = None, **params):
        super().__init__(ds=ds, comments_df=comments_df if comments_df is not None else pd.DataFrame(columns=['time','text']), **params)

        # Identify/prepare data variable for spectrograms if present; else raw-only
        if isinstance(ds, xr.Dataset):
            if "__xarray_dataarray_variable__" in ds.data_vars:
                self.data_var = ds["__xarray_dataarray_variable__"]
            else:
                try:
                    self.data_var = ds.to_dataarray()
                except Exception:
                    # Fallback: create zero-sized DataArray; visuals will hide spectrogram
                    self.data_var = xr.DataArray(np.empty((0,0)), dims=('freqs','times'))
        else:
            raise ValueError("ds must be an xarray Dataset")

        # Ensure dask-chunked for large data sets
        if hasattr(self.data_var, 'data') and not isinstance(self.data_var.data, da.Array):
            try:
                chunk_sizes = {
                    dim: (self.data_var.sizes[dim] if dim != 'times' else min(512, self.data_var.sizes[dim]))
                    for dim in getattr(self.data_var, 'dims', [])
                }
                self.data_var = self.data_var.chunk(chunk_sizes)
            except Exception:
                pass

        # Params initialization from dataset
        self.param.session.objects = list(self.ds.session.values)
        self.session = self.param.session.objects[0] if len(self.param.session.objects) > 0 else None
        self.param.channels = list(self.ds.channels.values)
        self.channels = list(self.ds.channels.values)

        times = self.ds.times.values
        self.time_window = (float(times.min()), float(times.max())) if times.size else (0.0, 0.0)

        # Build widgets
        self._build_widgets()

        # Initialize comment points element and selection stream for main panel linkage
        try:
            base_df = self.comments_df[['time']].copy()
            base_df['y'] = 0.0
        except Exception:
            base_df = pd.DataFrame({ 'time': [], 'y': [] })
        self.comment_points = hv.Points(base_df, kdims=['time','y']).opts(size=6, color='orange', height=60, width=900, tools=['tap'])
        self.comment_selection = Selection1D(source=self.comment_points)

        def _on_comment_select(indices):
            try:
                if not indices:
                    return
                idx = indices[0]
                if idx < 0 or idx >= len(self.comments_df):
                    return
                ts = float(self.comments_df.iloc[idx]['time'])
                # Update table selection and seek
                self.comments_table.selection = [idx]
                self._seek_to_time(ts)
            except Exception:
                pass

        self.comment_selection.add_subscriber(_on_comment_select)

    # ---------- Data helpers ----------
    def _slice_time(self, arr: xr.DataArray) -> xr.DataArray:
        try:
            return arr.sel(times=slice(self.time_window[0], self.time_window[1]))
        except Exception:
            return arr

    def _get_raw_eeg_for_channels(self, channel_list: List[str]) -> Optional[xr.DataArray]:
        try:
            arr = self.ds['raw'] if 'raw' in self.ds.data_vars else None
            if arr is None:
                return None
            arr = arr.sel(session=self.session, channels=channel_list)
            return self._slice_time(arr)
        except Exception:
            return None

    def _get_spectrogram_avg(self) -> Optional[xr.DataArray]:
        if getattr(self, 'data_var', None) is None or self.data_var.size == 0:
            return None
        try:
            arr = self.data_var.sel(session=self.session, channels=self.channels)
            arr = arr.mean(dim='channels') if 'channels' in arr.dims else arr
            return self._slice_time(arr)
        except Exception:
            return None

    def _compute_bandpowers(self) -> Optional[pd.DataFrame]:
        # Compute simple bandpowers using spectrogram average if available
        arr = self._get_spectrogram_avg()
        if arr is None:
            return None
        try:
            # arr dims: freqs x times
            freqs = arr['freqs'].values if 'freqs' in arr.coords else None
            if freqs is None:
                return None
            band_vals = {}
            for name, (lo, hi) in DEFAULT_BANDS.items():
                sub = arr.sel(freqs=slice(lo, hi))
                val = sub.mean(dim=('freqs','times'))
                val = 10 * np.log10(np.maximum(val, 1e-12))
                try:
                    band_vals[name] = float(val.compute().item())
                except Exception:
                    band_vals[name] = float(val.values) if np.ndim(val.values) == 0 else float(np.nan)
            df = pd.DataFrame({'band': list(band_vals.keys()), 'power_db': list(band_vals.values())})
            return df
        except Exception:
            return None

    # ---------- UI widgets ----------
    def _build_widgets(self):
        # Left: sessions pager
        self.session_selector = pn.widgets.Select(name='Session', options=list(self.ds.session.values), value=self.session)
        self.prev_button = pn.widgets.Button(name='Prev', button_type='primary')
        self.next_button = pn.widgets.Button(name='Next', button_type='primary')

        # Right: options
        self.channel_selector = pn.widgets.MultiChoice(name='Channels', value=self.channels, options=list(self.ds.channels.values))
        times = self.ds.times.values
        self.time_range_slider = pn.widgets.RangeSlider(name='Time window (s)', start=float(times.min()) if times.size else 0.0,
                                                        end=float(times.max()) if times.size else 0.0,
                                                        value=self.time_window,
                                                        step=(float(times[1]-times[0]) if times.size > 1 else 1.0))
        self.results_selector = pn.widgets.CheckButtonGroup(name='Results', value=['Spectrogram'], options=['Spectrogram','Bandpower'])
        self.colormap_selector = pn.widgets.Select(name='Colormap', options=['Viridis','Inferno','Magma','Plasma','Turbo'], value=self.colormap)
        self.export_button = pn.widgets.Button(name='Export HTML', button_type='success')

        # Bottom: comments
        # Expect columns: time (float seconds), text (str)
        if 'time' not in self.comments_df.columns:
            self.comments_df['time'] = []
        if 'text' not in self.comments_df.columns:
            self.comments_df['text'] = []
        self.comments_table = pn.widgets.DataFrame(self.comments_df, name='Comments', autosize_mode='fit_viewport', height=200) # , selectable=True

        # Wire events
        def _on_session(evt):
            self.session = evt.new
        def _on_prev(evt):
            if self.session is None:
                return
            items = list(self.ds.session.values)
            if not items:
                return
            idx = items.index(self.session)
            self.session = items[(idx - 1) % len(items)]
            self.session_selector.value = self.session
        def _on_next(evt):
            if self.session is None:
                return
            items = list(self.ds.session.values)
            if not items:
                return
            idx = items.index(self.session)
            self.session = items[(idx + 1) % len(items)]
            self.session_selector.value = self.session
        def _on_channels(evt):
            self.channels = list(evt.new)
        def _on_time_window(evt):
            try:
                start, end = evt.new
                self.time_window = (float(start), float(end))
            except Exception:
                pass
        def _on_export(evt):
            self._export_html()

        self.session_selector.param.watch(_on_session, 'value')
        self.prev_button.on_click(_on_prev)
        self.next_button.on_click(_on_next)
        self.channel_selector.param.watch(_on_channels, 'value')
        self.time_range_slider.param.watch(_on_time_window, 'value')
        self.export_button.on_click(_on_export)

        # Comments selection -> seek
        def _on_comments_select(event):
            try:
                # DataFrame widget exposes .selection as list of row indices
                sel = self.comments_table.selection
                if not sel:
                    return
                idx = sel[0]
                ts = float(self.comments_df.iloc[idx]['time'])
                self._seek_to_time(ts)
            except Exception:
                pass

        self.comments_table.param.watch(lambda e: _on_comments_select(e), 'selection')

    # ---------- Interaction helpers ----------
    def _seek_to_time(self, ts: float):
        try:
            start, end = self.time_window
            width = end - start
            if width <= 0:
                return
            new_start = max(ts - width * 0.5, float(self.ds.times.values.min()))
            new_end = new_start + width
            tmin, tmax = float(self.ds.times.values.min()), float(self.ds.times.values.max())
            if new_end > tmax:
                new_end = tmax
                new_start = max(tmin, new_end - width)
            self.time_window = (new_start, new_end)
            self.time_range_slider.value = self.time_window
        except Exception:
            pass

    # ---------- Panels ----------
    @pn.depends('session', 'channels', 'time_window', 'colormap', 'show_results')
    def main_timeline_panel(self):
        plots = []

        # Raw EEG plot (if available)
        raw_arr = self._get_raw_eeg_for_channels(self.channels)
        if raw_arr is not None:
            try:
                # Expect dims: channels x times OR times x channels; normalize
                dims = list(raw_arr.dims)
                if 'times' in dims and 'channels' in dims:
                    # Convert to DataFrame for hv.Curve overlay per channel
                    df_list = []
                    for ch in raw_arr['channels'].values:
                        y = raw_arr.sel(channels=ch)
                        df_list.append(pd.DataFrame({'times': y['times'].values, 'value': y.values, 'channel': str(ch)}))
                    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=['times','value','channel'])
                    curve = hv.Overlay([
                        hv.Curve(df[df['channel']==ch], kdims=['times'], vdims=['value']).opts(title='Raw EEG', height=250, width=900)
                        for ch in df['channel'].unique()
                    ]) if not df.empty else hv.Curve([])
                    plots.append(curve)
            except Exception:
                pass

        # Spectrogram average
        if 'Spectrogram' in self.show_results:
            arr = self._get_spectrogram_avg()
            if arr is not None:
                try:
                    img = hv.Image(arr)
                except Exception:
                    try:
                        z = arr.values
                        img = hv.Image((arr['times'].values, arr.get('freqs', arr['times']).values, z), ['times','freqs'], '__val__')
                    except Exception:
                        img = hv.Image([])
                shaded = hd.datashade(img, cmap=self.colormap)
                shaded = shaded.opts(height=300, width=900)
                plots.append(shaded)

        # Bandpower bars
        if 'Bandpower' in self.show_results:
            bp = self._compute_bandpowers()
            if bp is not None and not bp.empty:
                bars = hv.Bars(bp, kdims=['band'], vdims=['power_db']).opts(height=200, width=400, xlabel='Band', ylabel='Power (dB)')
                plots.append(bars)

        # Comment track: reuse element with selection stream
        if not self.comments_df.empty:
            plots.append(self.comment_points)

        if not plots:
            return pn.pane.Markdown("No data to display")

        layout = hv.Layout(plots).cols(1)
        return pn.pane.HoloViews(layout)

    def left_sidebar_panel(self):
        return pn.Column(
            pn.pane.Markdown("### Sessions"),
            self.session_selector,
            pn.Row(self.prev_button, self.next_button),
            sizing_mode='stretch_width'
        )

    def right_sidebar_panel(self):
        return pn.Column(
            pn.pane.Markdown("### Display Options"),
            self.channel_selector,
            self.results_selector,
            self.colormap_selector,
            self.export_button,
            sizing_mode='stretch_width'
        )

    def bottom_comments_panel(self):
        return pn.Column(
            pn.pane.Markdown("### Comments"),
            self.time_range_slider,
            self.comments_table,
            sizing_mode='stretch_width'
        )

    def panel(self):
        header = pn.pane.Markdown("# Standalone Results Browser")
        # 4-panel layout: left | main | right stacked above bottom
        top = pn.Row(self.left_sidebar_panel(), self.main_timeline_panel, self.right_sidebar_panel())
        layout = pn.Column(header, top, self.bottom_comments_panel())
        return layout

    # ---------- Export ----------
    def _export_html(self):
        try:
            layout = self.panel()
            layout.save(self.export_filename, embed=True)
        except Exception as e:
            print(f"Export failed: {e}")


def export_results_browser(ds: xr.Dataset, output_path: str, comments_df: Optional[pd.DataFrame] = None) -> str:
    """
    Build the standalone results browser and save it as a self-contained HTML.
    Returns the output path on success.
    """
    app = StandaloneResultsBrowserApp(ds=ds, comments_df=comments_df)
    layout = app.panel()
    layout.save(output_path, embed=True)
    return output_path


def make_app(ds: xr.Dataset, comments_df: Optional[pd.DataFrame] = None):
    """
    Factory for the standalone results browser. Returns a Panel layout.
    """
    app = StandaloneResultsBrowserApp(ds=ds, comments_df=comments_df)
    return app.panel()


def _get_default_panel():
    try:
        ds = globals().get('ds_disk', None)
        comments = globals().get('comments_df', None)
        if ds is None:
            return pn.Column(pn.pane.Markdown("No dataset `ds_disk` found. Use `make_app(ds, comments_df)`."))
        return make_app(ds, comments_df=comments)
    except Exception:
        return pn.Column(pn.pane.Markdown("Error while creating default app. Use `make_app(ds, comments_df)`."))


pn_app = _get_default_panel()


