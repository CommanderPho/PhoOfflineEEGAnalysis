# Rerun examples

Examples that stream PhoOfflineEEGAnalysis data (e.g. from XDF files) into [Rerun](https://rerun.io/) for visualization.

## XDF IMU-style (`xdf_imu_style.py`)

Logs XDF stream data in the same style as the [Rerun IMU signals example](https://rerun.io/examples/feature-showcase/imu_signals): one entity per stream with multiple scalar columns via `rr.send_columns()` and `rr.Scalars.columns()`. The viewer shows one time-series panel per stream with multiple lines (channels).

**Run from project root:**

```bash
uv run python rerun/xdf_imu_style.py path/to/recording.xdf
```

Options: `--save` / `-o` (output .rrd path), `--no-spawn` (do not open viewer), `--step N` (downsample to every Nth sample).
