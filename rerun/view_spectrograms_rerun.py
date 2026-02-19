"""
View EEG spectrogram export in Rerun (run in another process).

Loads a .npz file produced by export_spectrograms_for_rerun() from main_analyze_run.py,
logs each session's spectrograms with datetime to Rerun (one frame per time step per channel,
so the timeline is scrollable), sends a blueprint for a fixed 10s trailing window on the
session_time timeline, then saves to .rrd and optionally spawns the viewer.
Requires rerun-sdk in a separate env (main project does not depend on it).

Usage (from repo root, using Rerun subproject):
  uv run --project rerun -- python rerun/view_spectrograms_rerun.py <path/to/spectrograms_export.npz> [--spawn] [--out path.rrd] [--max-frames N]

Or with pip: pip install rerun-sdk, then:
  python rerun/view_spectrograms_rerun.py <path.npz> [--spawn] [--out path.rrd] [--max-frames N]

Then open the generated .rrd in another process: rerun path/to/spectrograms.rrd
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def _sanitize_entity_name(name: str) -> str:
    """Replace characters that are invalid in Rerun entity paths."""
    return re.subn(r"[\/\\]", "_", name)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="View exported EEG spectrograms in Rerun.")
    parser.add_argument("npz_path", type=Path, help="Path to spectrograms_export.npz from export_spectrograms_for_rerun()")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer after logging (otherwise only save .rrd)")
    parser.add_argument("--out", type=Path, default=None, help="Output .rrd path (default: same stem as npz, next to npz)")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames per channel for timeline scrubbing (default: 500)")
    args = parser.parse_args()
    npz_path = args.npz_path.resolve()
    if not npz_path.exists():
        print(f"ERROR: File not found: {npz_path}", file=sys.stderr)
        return 1
    out_path = args.out
    if out_path is None:
        out_path = npz_path.with_suffix(".rrd")
    else:
        out_path = Path(out_path).resolve()

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError:
        print("ERROR: rerun-sdk not installed. Install in a Rerun-capable env (e.g. uv run --project rerun, or pip install rerun-sdk).", file=sys.stderr)
        return 1

    data = np.load(npz_path, allow_pickle=True)
    session_indices = data["session_indices"]
    if session_indices.size == 0:
        print("No sessions in export file.", file=sys.stderr)
        return 1

    rr.init("pho_eeg_spectrograms")

    for idx in session_indices:
        idx = int(idx)
        meas_date_sec = float(data[f"s{idx}_meas_date_sec"].item())
        channel_names = data[f"s{idx}_channel_names"]
        if channel_names.ndim == 0:
            channel_names = np.array([str(channel_names.item())])
        else:
            channel_names = [str(c) for c in channel_names]
        Sxx = data[f"s{idx}_Sxx"]
        n_ch, n_freq, n_time = Sxx.shape
        times = np.asarray(data[f"s{idx}_times"]).flatten()
        if times.shape[0] != n_time:
            print(f"ERROR: s{idx}_times length {times.shape[0]} != Sxx time dimension {n_time}", file=sys.stderr)
            return 1
        meas_date_is_nan = np.isnan(meas_date_sec)
        max_frames = max(1, args.max_frames)
        step = max(1, n_time // max_frames)
        time_indices = list(range(0, n_time, step))
        if n_time > 1 and time_indices[-1] != n_time - 1:
            time_indices.append(n_time - 1)
        for ch_idx, ch_name in enumerate(channel_names):
            Sxx_ch = np.asarray(Sxx[ch_idx], dtype=np.float64)
            Sxx_db = 10.0 * np.log10(Sxx_ch + 1e-12)
            vmin, vmax = np.nanmin(Sxx_db), np.nanmax(Sxx_db)
            entity_path = f"sessions/session_{idx}/spectrograms/{_sanitize_entity_name(ch_name)}"
            if n_time <= 1:
                if vmax > vmin:
                    img = (Sxx_db - vmin) / (vmax - vmin)
                else:
                    img = np.zeros_like(Sxx_db)
                img = np.clip(img, 0.0, 1.0).astype(np.float32)
                if meas_date_is_nan:
                    rr.set_time("session_time", duration=float(times[0]) if n_time > 0 else 0.0)
                else:
                    rr.set_time("session_time", timestamp=meas_date_sec + (float(times[0]) if n_time > 0 else 0.0))
                rr.log(entity_path, rr.Image(img))
            else:
                half_win = min(250, n_time // 4)
                for ti in time_indices:
                    if meas_date_is_nan:
                        rr.set_time("session_time", duration=float(times[ti]))
                    else:
                        rr.set_time("session_time", timestamp=meas_date_sec + float(times[ti]))
                    j_lo = max(0, ti - half_win)
                    j_hi = min(n_time, ti + half_win + 1)
                    Sxx_ch_window = Sxx_db[:, j_lo:j_hi]
                    if vmax > vmin:
                        img_window = (Sxx_ch_window - vmin) / (vmax - vmin)
                    else:
                        img_window = np.zeros_like(Sxx_ch_window)
                    img_window = np.clip(img_window, 0.0, 1.0).astype(np.float32)
                    rr.log(entity_path, rr.Image(img_window))

    rr.send_blueprint(rrb.Spatial2DView(origin="/", time_ranges=rrb.VisibleTimeRange(timeline="session_time", start=rrb.TimeRangeBoundary.cursor_relative(), end=rrb.TimeRangeBoundary.cursor_relative(seconds=60.0))))
    rr.save(str(out_path))
    print(f"Saved Rerun recording to: {out_path}")
    if args.spawn:
        rr.spawn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
