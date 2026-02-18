"""
View EEG spectrogram export in Rerun (run in another process).

Loads a .npz file produced by export_spectrograms_for_rerun() from main_analyze_run.py,
logs each session's spectrograms with datetime to Rerun, then saves to .rrd and optionally
spawns the viewer. The NPZ may also contain raw EEG, motion, and text log; when present,
they are logged to Rerun with the same session time alignment. Requires rerun-sdk in a
separate env (main project does not depend on it).

Usage (from repo root, using Rerun subproject):
  uv run --project rerun -- python rerun/view_spectrograms_rerun.py <path/to/spectrograms_export.npz> [--spawn] [--out path.rrd]

Or with pip: pip install rerun-sdk, then:
  python rerun/view_spectrograms_rerun.py <path.npz> [--spawn] [--out path.rrd]

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
    except ImportError:
        print("ERROR: rerun-sdk not installed. Install in a Rerun-capable env (e.g. uv run --project rerun, or pip install rerun-sdk).", file=sys.stderr)
        return 1

    data = np.load(npz_path, allow_pickle=True)
    session_indices = data["session_indices"]
    if session_indices.size == 0:
        print("No sessions in export file.", file=sys.stderr)
        return 1

    rr.init("pho_eeg_spectrograms")
    keys = set(data.files)

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
        for ch_idx, ch_name in enumerate(channel_names):
            Sxx_ch = np.asarray(Sxx[ch_idx], dtype=np.float64)
            Sxx_db = 10.0 * np.log10(Sxx_ch + 1e-12)
            vmin, vmax = np.nanmin(Sxx_db), np.nanmax(Sxx_db)
            if vmax > vmin:
                img = (Sxx_db - vmin) / (vmax - vmin)
            else:
                img = np.zeros_like(Sxx_db)
            img = np.clip(img, 0.0, 1.0).astype(np.float32)
            entity_path = f"sessions/session_{idx}/spectrograms/{_sanitize_entity_name(ch_name)}"
            rr.set_time("session_time", timestamp=meas_date_sec)
            rr.log(entity_path, rr.Image(img))

        # Raw EEG time series (if present)
        eeg_data_key = f"s{idx}_eeg_data"
        if eeg_data_key in keys:
            eeg_data = np.asarray(data[eeg_data_key], dtype=np.float64)
            eeg_times = np.asarray(data[f"s{idx}_eeg_times"], dtype=np.float64)
            eeg_ch_names = data[f"s{idx}_eeg_channel_names"]
            if eeg_ch_names.ndim == 0:
                eeg_ch_names = [str(eeg_ch_names.item())]
            else:
                eeg_ch_names = [str(c) for c in eeg_ch_names]
            n_eeg_ch, n_eeg_time = eeg_data.shape
            for t_idx in range(n_eeg_time):
                t_sec = meas_date_sec + float(eeg_times[t_idx])
                rr.set_time("session_time", timestamp=t_sec)
                for ch_idx, ch_name in enumerate(eeg_ch_names):
                    if ch_idx < n_eeg_ch:
                        rr.log(f"sessions/session_{idx}/eeg_raw/{_sanitize_entity_name(ch_name)}", rr.Scalars(float(eeg_data[ch_idx, t_idx])))

        # Motion time series (if present)
        motion_data_key = f"s{idx}_motion_data"
        if motion_data_key in keys:
            motion_data = np.asarray(data[motion_data_key], dtype=np.float64)
            motion_times = np.asarray(data[f"s{idx}_motion_times"], dtype=np.float64)
            motion_ch_names = data[f"s{idx}_motion_ch_names"]
            if motion_ch_names.ndim == 0:
                motion_ch_names = [str(motion_ch_names.item())]
            else:
                motion_ch_names = [str(c) for c in motion_ch_names]
            n_motion_ch, n_motion_time = motion_data.shape
            for t_idx in range(n_motion_time):
                t_sec = meas_date_sec + float(motion_times[t_idx])
                rr.set_time("session_time", timestamp=t_sec)
                for ch_idx, ch_name in enumerate(motion_ch_names):
                    if ch_idx < n_motion_ch:
                        rr.log(f"sessions/session_{idx}/motion/{_sanitize_entity_name(ch_name)}", rr.Scalars(float(motion_data[ch_idx, t_idx])))

        # Text log (if present)
        text_onset_key = f"s{idx}_text_onset"
        if text_onset_key in keys:
            text_onset = np.asarray(data[text_onset_key], dtype=np.float64)
            text_desc = data[f"s{idx}_text_description"]
            if text_desc.ndim == 0:
                text_desc = [str(text_desc.item())]
            else:
                text_desc = [str(x) for x in text_desc]
            for i in range(len(text_onset)):
                t_sec = meas_date_sec + float(text_onset[i])
                rr.set_time("session_time", timestamp=t_sec)
                rr.log(f"sessions/session_{idx}/text_log", rr.TextLog(str(text_desc[i]) if i < len(text_desc) else ""))

    rr.save(str(out_path))
    print(f"Saved Rerun recording to: {out_path}")
    if args.spawn:
        rr.spawn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
