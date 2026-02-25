# Rerun examples

Examples that stream PhoOfflineEEGAnalysis data (e.g. from XDF files) into [Rerun](https://rerun.io/) for visualization.

## XDF IMU-style (`xdf_imu_style.py`)

Logs XDF stream data in the same style as the [Rerun IMU signals example](https://rerun.io/examples/feature-showcase/imu_signals): one entity per stream with multiple scalar columns via `rr.send_columns()` and `rr.Scalars.columns()`. The viewer shows one time-series panel per stream with multiple lines (channels).

**Run from project root:**

```bash
uv run python rerun/xdf_imu_style.py path/to/recording.xdf
```

Options: `--save` / `-o` (output .rrd path), `--no-spawn` (do not open viewer), `--step N` (downsample to every Nth sample).

## XDF data loader plugin (`rerun_loader_xdf.py`)

External data loader that lets you **open .xdf files directly in the Rerun Viewer** (open dialog, drag-and-drop, or `rerun file.xdf`). Logs each XDF stream as one entity with multi-channel scalars (IMU-style).

**Run standalone from project root:**

```bash
uv run --project rerun -- python rerun/rerun_loader_xdf.py path/to/recording.xdf
```

**Install as `rerun-loader-xdf` so the Viewer discovers it:** Any executable on your PATH whose name starts with `rerun-loader-` is invoked by the Viewer when opening a file. To enable “open .xdf in Rerun”:

- **Symlink (Unix):** `ln -s /path/to/PhoOfflineEEGAnalysis/rerun/rerun_loader_xdf.py /path/on/PATH/rerun-loader-xdf` (and make it executable, or use a small wrapper that runs `uv run --project rerun -- python /path/to/rerun_loader_xdf.py "$@"`).
- **Windows:** Create a batch file or script named `rerun-loader-xdf.cmd` (or `rerun-loader-xdf`) on PATH that runs `uv run --project c:\path\to\PhoOfflineEEGAnalysis\rerun -- python c:\path\to\PhoOfflineEEGAnalysis\rerun\rerun_loader_xdf.py %*`.

After installation, opening an .xdf file in the Rerun Viewer (e.g. drag-and-drop or File > Open) will use this loader and display the streams.
