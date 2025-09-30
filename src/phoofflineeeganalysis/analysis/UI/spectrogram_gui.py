# spectrogram_gui.py
"""
Spectrogram analysis GUI using Panel + HoloViews + Datashader + Dask.

Save this file and run:
    panel serve spectrogram_gui.py --show

Or import `make_app(ds, channels_to_select)` from this file in a running notebook/process.
"""

import numpy as np
import xarray as xr
import dask.array as da
import holoviews as hv
import datashader as ds
import datashader.transfer_functions as tf
import holoviews.operation.datashader as hd
import panel as pn
import param
import pandas as pd
from holoviews import opts
from bokeh.models import HoverTool

hv.extension('bokeh')
pn.extension()

# Define EEG bands
DEFAULT_BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma (30-64 Hz)": (30, 64),
}

# Helpful util: safe dB conversion
def _to_db(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(x, floor))


class SpectrogramApp(param.Parameterized):
    """
    Panel/HoloViews application for spectrogram visualization with datashader + dask.

    Instantiate with:
        app = SpectrogramApp(ds=ds_disk, channels_to_select=['AF3','F7','F3','FC5'])
        pn.serve(app.panel(), show=True)
    or inside a script, call: panel serve spectrogram_gui.py --show
    """

    ds = param.Parameter(doc="xarray Dataset containing spectrogram with dims (session, channels, freqs, times)")
    channels_to_select = param.List([], doc="Channels to show individually")
    session = param.ObjectSelector(doc="Selected session")
    channels = param.List(doc="Selected channels (for averaging)")
    time_window = param.Tuple(default=(0.0, 10.0), doc="Start/stop time window (s)")
    colormap = param.ObjectSelector(default='Viridis', objects=['Viridis', 'Inferno', 'Magma', 'Plasma', 'Turbo'])
    use_db = param.Boolean(default=True, doc="Display power in dB (True) or linear")
    agg_func = param.ObjectSelector(default='mean', objects=['mean', 'median', 'max'], doc="Aggregation function for datashader")
    export_filename = param.String(default='spectrogram_export.html')

    def __init__(self, ds: xr.Dataset, channels_to_select=None, **params):
        super().__init__(ds=ds, channels_to_select=channels_to_select or [], **params)

        # Basic dataset validation & conversion
        if "__xarray_dataarray_variable__" not in ds.data_vars:
            raise ValueError("Dataset must contain a data variable named '__xarray_dataarray_variable__' (power data).")

        self.data_var = ds["__xarray_dataarray_variable__"]

        # Ensure the array is dask-backed (lazy) to handle large data:
        if not isinstance(self.data_var.data, da.Array):
            # Convert to dask with reasonable chunking (chunk time, keep others whole)
            chunk_sizes = {
                dim: (self.data_var.sizes[dim] if dim != 'times'
                    else min(512, self.data_var.sizes[dim]))
                for dim in self.data_var.dims
            }
            try:
                self.data_var = self.data_var.chunk(chunk_sizes)
            except Exception as e:
                print(f"Chunking failed: {e}")


        # Fill param widgets with dataset content
        self.param.session.objects = list(self.ds.session.values)
        self.session = self.param.session.objects[0]

        self.param['channels'].objects = list(self.ds.channels.values)


        # default to all channels selected for the averaged plot
        self.channels = list(self.ds.channels.values)

        # Set a wide default time window (full range)
        times = self.ds.times.values
        self.time_window = (float(times.min()), float(times.max()))

        # Panel widgets
        self._build_widgets()

    def _build_widgets(self):
        self.session_selector = pn.widgets.Select(name='Session', options=list(self.ds.session.values),
                                                 value=self.session)
        self.channel_selector = pn.widgets.MultiChoice(name='Channels (avg)', value=self.channels,
                                                       options=list(self.ds.channels.values))
        self.channel_select_indiv = pn.widgets.MultiChoice(name='Channels (individual)',
                                                           value=self.channels_to_select,
                                                           options=list(self.ds.channels.values))
        times = self.ds.times.values
        self.time_range_slider = pn.widgets.RangeSlider(name='Time window (s)',
                                                        start=float(times.min()), end=float(times.max()),
                                                        value=self.time_window, step=max(1.0, float((times[1]-times[0]))))
        self.colormap_selector = pn.widgets.Select(name='Colormap', options=['Viridis','Inferno','Magma','Plasma','Turbo'],
                                                  value=self.colormap)
        self.db_toggle = pn.widgets.Checkbox(name='Show dB', value=self.use_db)
        self.export_button = pn.widgets.Button(name='Export current view to HTML', button_type='primary')

        # link widget events to param values
        pn.bind(self._set_param_from_widget, self.session_selector, self.channel_selector,
                self.channel_select_indiv, self.time_range_slider, self.colormap_selector, self.db_toggle)

        self.export_button.on_click(self._export_current_view)

    def _set_param_from_widget(self, session, channels, channels_indiv, time_window, colormap, db_toggle):
        # update parameters
        self.session = session
        self.channels = list(channels)
        self.channels_to_select = list(channels_indiv)
        self.time_window = tuple(time_window)
        self.colormap = colormap
        self.use_db = db_toggle
        return

    # Data access helpers
    def _get_avg_spectrogram_dask(self):
        """Return an xarray DataArray (dask-backed) averaged over selected channels for the chosen session."""
        sel = dict(session=self.session, channels=self.channels)
        arr = self.data_var.sel(**sel).mean(dim='channels')
        # slice time window
        arr = arr.sel(times=slice(self.time_window[0], self.time_window[1]))
        return arr  # dims: freqs x times

    def _get_channel_spectrogram_dask(self, ch):
        sel = dict(session=self.session, channels=ch)
        arr = self.data_var.sel(**sel)
        arr = arr.sel(times=slice(self.time_window[0], self.time_window[1]))
        return arr  # dims: freqs x times

    def _get_freqs_times(self, arr):
        freqs = self.ds.freqs.values
        times = self.ds.times.values
        # apply same slicing as arr (arr coords will reflect selection)
        return arr['freqs'].values, arr['times'].values

    # HoloViews plotting helpers
    def _make_datashaded_image(self, arr_xr: xr.DataArray, cmap='Viridis', use_db=True, aggregator='mean'):
        """
        arr_xr: xr.DataArray with dims ('freqs','times') or ('times','freqs') ideally (we will map y->freqs, x->times)
        Return: hv.Image that is datashaded for fast interactive viewing.
        """
        # Ensure dims order: (times, freqs) for datashader (x,y)
        # We'll create hv.Image with (x=times, y=freqs)
        # Convert dask-backed array to 2D numpy via holoviews+datashader pipeline (datashader will do aggregation)
        # Create an hv.Image using the xarray values lazily via hv.Dataset
        # Create DataFrame points for datashader to aggregate -> but that's heavy for very big arrays.
        # Instead we use hd.datashade on hv.Image which accepts xarray-backed images.

        # Convert to 2D numpy-backed hv.Image but rely on datashader for aggregation
        freqs = arr_xr['freqs'].values
        times = arr_xr['times'].values

        # Create HV Image: dims are (times, freqs) -> (x,y)
        # hv.Image expects Z shape (ny, nx) with x and y coords. We'll transpose to match
        # but to preserve lazy compute we create hv.Image from numpy metadata and Dask array
        z = arr_xr.data  # this is a dask array with shape (freqs, times)
        # datashader expects coordinates: x->times, y->freqs. We'll transpose z to (times,freqs) for hv.Image mapping
        # hv.Image expects z shape (ny, nx) where x corresponds to columns (times), y to rows (freqs)
        # To avoid unnecessary compute, we will make a small helper that returns a function to compute when needed.
        # Simpler: create a small hv.Dataset by stacking coords into columns (this will generate points but is acceptable
        # because datashader will aggregate). For very large arrays consider using rasterization of pre-aggregated tiles.

        # We'll create a pandas DataFrame lazily via dask if needed; but to keep code concise, we use hd.datashade on an
        # hv.Image constructed from np.zeros (placeholder) with explicit extents, and use the aggregate function to draw from arr_xr when exporting.
        # Instead, HoloViews supports hv.Image(arr_xr) with xarray input for recent versions.
        try:
            # Try hv.Image from xarray directly (preserves coords)
            img = hv.Image(arr_xr, kdims=['times', 'freqs'], vdims=['__val__'])
        except Exception:
            # fallback: convert small chunk to numpy (this will compute) - this is acceptable because datashader will handle large arrays in interactive usage
            z_np = _to_db(arr_xr.values) if use_db else arr_xr.values
            img = hv.Image((times, freqs, z_np), ['times', 'freqs'], '__val__')

        # Apply dB conversion lazily if possible
        if use_db:
            # If we have a numeric numpy array inside, convert now; otherwise leave as is and let hv pipeline handle it
            try:
                img = img.apply(lambda x: 10 * np.log10(np.maximum(x, 1e-12)))
            except Exception:
                pass

        # Use datashader to rasterize
        if aggregator == 'mean':
            agg = ds.reductions.mean
        elif aggregator == 'median':
            agg = ds.reductions.median
        elif aggregator == 'max':
            agg = ds.reductions.max
        else:
            agg = ds.reductions.mean

        # Use holoviews operation: datashade (returns an RGB Element)
        shaded = hd.datashade(img, aggregator=agg, cmap=cmap, dynamic=False)
        # Convert to hv.RGB for display; allow link to hover tool by using Rasterize when needed
        return shaded

    # Bandpower computation (lazy, via Dask)
    def compute_bandpowers(self):
        arr = self._get_avg_spectrogram_dask()  # dims freq x times
        freqs = self.ds.freqs.values
        band_vals = {}
        for name, (lo, hi) in DEFAULT_BANDS.items():
            # select frequencies in band
            band = arr.sel(freqs=slice(lo, hi))
            # compute mean across freqs and times (global band power for this session/channels/time window)
            # Keep lazy: use .mean() returns xarray object (with dask compute only when .compute())
            val = band.mean(dim=('freqs', 'times'))
            # convert to dB if requested - do lazily by mapping over dask array
            if self.use_db:
                # val is a zero-dim xarray DataArray; we can attach dB conversion after compute
                band_vals[name] = (10 * np.log10(val)).compute().item()
            else:
                band_vals[name] = val.compute().item()
        return band_vals

    # Panel reactive views
    @pn.depends('session', 'channels', 'time_window', 'colormap', 'use_db', 'agg_func', watch=True)
    def _update_internal(self):
        # Called when parameters change; placeholder for potential cached computations
        pass

    def hv_spectrogram_panel(self):
        """Return the main datashaded averaged spectrogram (HoloViews pane)"""
        arr = self._get_avg_spectrogram_dask()  # xr.DataArray dims ('freqs','times')
        # Try to create hv.Image directly from xarray if supported
        try:
            # transpose to hv convention (times x freqs) if needed by hv.Image; hv.Image(xarray) handles dims order
            img = hv.Image(arr)
        except Exception:
            # fallback make small numpy compute (not ideal for large arrays)
            z = arr.values
            if self.use_db:
                z = _to_db(z)
            img = hv.Image((arr['times'].values, arr['freqs'].values, z), ['times', 'freqs'], '__val__')
        # datashade it using holoviews.operation.datashader
        shaded = hd.datashade(img, aggregator=getattr(ds.reductions, self.agg_func), cmap=self.colormap)
        # Add some options & return as a Panel pane
        shaded = shaded.opts(height=300, width=900, tools=['hover'], active_tools=['wheel_zoom'])
        return pn.pane.HoloViews(shaded)

    def hv_individual_channels_panel(self):
        """Return a column of datashaded spectrograms for each selected individual channel."""
        panels = []
        for ch in self.channels_to_select:
            arr = self._get_channel_spectrogram_dask(ch).squeeze()
            # build hv.Image
            try:
                img = hv.Image(arr)
            except Exception:
                z = arr.values
                if self.use_db:
                    z = _to_db(z)
                img = hv.Image((arr['times'].values, arr['freqs'].values, z), ['times', 'freqs'], '__val__')
            shaded = hd.datashade(img, aggregator=getattr(ds.reductions, self.agg_func), cmap=self.colormap)
            shaded = shaded.opts(title=f"{ch}", height=150, width=300)
            panels.append(shaded)
        if panels:
            return pn.Column(*[pn.pane.HoloViews(p) for p in panels], sizing_mode='stretch_width')
        else:
            return pn.pane.Markdown("No individual channels selected")

    def bandpower_panel(self):
        """Return a small bar chart of bandpowers computed with Dask (compute happens lazily when rendering)."""
        try:
            band_vals = self.compute_bandpowers()
        except Exception as e:
            # If compute fails (e.g., too big), attempt incremental compute with coarse downsampling
            pn.state.notifications.warning("Bandpower computation failed; try reducing time window or channels.")
            band_vals = {k: np.nan for k in DEFAULT_BANDS.keys()}

        df = pd.DataFrame({'band': list(band_vals.keys()), 'power': list(band_vals.values())})
        bar = hv.Bar(df, 'band', 'power').opts(height=250, width=300, xlabel='Band', ylabel='Power (dB)' if self.use_db else 'Power')
        return pn.pane.HoloViews(bar)

    def _export_current_view(self, event=None):
        """Export the current composed view to an HTML file using hv.save or pn.panel.save."""
        # Compose a layout similar to the main panel and save to HTML
        layout = pn.Row(self.main_view(), self.side_view())
        try:
            layout.save(self.export_filename, embed=True)
            pn.state.notifications.success(f"Saved to {self.export_filename}")
        except Exception as e:
            pn.state.notifications.error(f"Export failed: {e}")

    # Compose main layout pieces
    def main_view(self):
        return pn.Column(
            pn.Row(pn.Column(self.session_selector, self.channel_selector, self.channel_select_indiv, self.time_range_slider,
                             pn.Row(self.colormap_selector, self.db_toggle, self.export_button)),
                   pn.Spacer(width=20),
                   pn.Column(self.hv_spectrogram_panel, self.bandpower_panel)),
            sizing_mode='stretch_width'
        )

    def side_view(self):
        return pn.Column(pn.pane.Markdown("### Individual channels"),
                         self.hv_individual_channels_panel(),
                         sizing_mode='stretch_width')

    def panel(self):
        # high-level composition
        header = pn.pane.Markdown("# EEG Spectrogram Explorer (Datashader + Dask)")
        layout = pn.Column(header, pn.Row(self.main_view(), self.side_view()))
        return layout


# Factory function to create and return a Panel app
def make_app(ds: xr.Dataset, channels_to_select=None):
    """
    Create the SpectrogramApp instance and return the Panel layout (call .show() or serve).
    Example:
        from phoofflineeeganalysis.analysis.UI.spectrogram_gui import make_app

        app = make_app(ds_disk, channels_to_select=['AF3','F7','F3','FC5'])
        pn.serve(app, show=True)   # if using programmatic server
    """
    app = SpectrogramApp(ds=ds, channels_to_select=channels_to_select or ['AF3','F7','F3','FC5'])
    return app.panel()


# If run with `panel serve spectrogram_gui.py` Panel will look for `pn` at the module level.
# We expose a default `pn` object that Panel will serve
def _get_default_panel():
    # Attempt to look for a variable named `ds_disk` in the global scope (useful when running in notebook)
    try:
        import inspect, sys
        # If ds_disk is already defined globally (like in your session) use it; otherwise user must import and create app manually.
        ds = globals().get('ds_disk', None)
        if ds is None:
            # Fallback: create an empty markdown telling the user to construct the app manually.
            return pn.Column(pn.pane.Markdown("No dataset `ds_disk` found in module globals. "
                                             "Please create the app programmatically with `make_app(ds)`"))
        else:
            return make_app(ds, channels_to_select=['AF3','F7','F3','FC5'])
    except Exception:
        return pn.Column(pn.pane.Markdown("Error while creating default app. Use `make_app(ds)`."))


# Expose Panel app object for `panel serve` convenience
pn_app = _get_default_panel()
