#!/usr/bin/env python3
"""
Rerun external data loader for .xdf files.

Enables opening .xdf files directly in the Rerun Viewer (open dialog, drag-and-drop,
or `rerun file.xdf`). Implements the external-loader contract: when invoked as
an executable named `rerun-loader-xdf` on PATH, receives filepath and optional
--recording-id, exits with EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE for
non-.xdf files, and logs XDF streams to stdout for the Viewer to ingest.

Installation (so the Rerun Viewer discovers the loader):
  - Symlink: ensure a script named `rerun-loader-xdf` (or any name starting with
    `rerun-loader-`) is on your PATH and invokes this script, e.g. on Unix:
    ln -s /path/to/rerun_loader_xdf.py /path/on/PATH/rerun-loader-xdf
  - Or from repo root with uv: uv run --project rerun -- python rerun/rerun_loader_xdf.py <path>
    (Viewer will not auto-detect this; you must install/symlink for that.)

Run from repo root (standalone): uv run --project rerun -- python rerun/rerun_loader_xdf.py path/to/file.xdf
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyxdf
import rerun as rr


def _file_datetime_from_header(header: dict | None) -> datetime | None:
    """Parse XDF header session datetime (same as LabRecorderXDF). Returns UTC datetime or None."""
    if not header:
        return None
    try:
        info = header.get("info") or {}
        dt_list = info.get("datetime")
        if not dt_list or not isinstance(dt_list, (list, tuple)) or len(dt_list) < 1:
            return None
        dt_str = dt_list[0]
        if not isinstance(dt_str, str) or not dt_str.strip():
            return None
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%dT%H:%M:%S%z")
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, IndexError):
        return None


def _sanitize_entity_name(name: str) -> str:
    """Replace characters that are invalid in Rerun entity paths."""
    if not name:
        return "stream"
    return re.sub(r"[\/\\]", "_", name)


def _unwrap_label(val) -> str:
    """Unwrap single-element list/tuple to match LabRecorderXDF (e.g. ['AF3'] -> 'AF3')."""
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return str(val[0]) if val[0] is not None else ""
    if val is None:
        return ""
    return str(val)


def _channel_labels_from_stream(stream: dict, n_channels: int) -> list[str]:
    """Derive channel labels from XDF stream desc (same as LabRecorderXDF), with unwrap; fallback ch_0, ch_1, ..."""
    try:
        desc = stream.get("info", {}).get("desc", [None])
        if not desc or desc[0] is None:
            return [f"ch_{i}" for i in range(n_channels)]
        channels = desc[0].get("channels", [None])
        if not channels or channels[0] is None:
            return [f"ch_{i}" for i in range(n_channels)]
        ch_list = channels[0].get("channel", [])
        if not isinstance(ch_list, list):
            return [f"ch_{i}" for i in range(n_channels)]
        labels = []
        for i in range(n_channels):
            if i < len(ch_list):
                c = ch_list[i]
                if isinstance(c, dict):
                    raw = c.get("label", c)
                    label = _unwrap_label(raw).strip() or f"ch_{i}"
                else:
                    label = _unwrap_label(c).strip() or f"ch_{i}"
                labels.append(label)
            else:
                labels.append(f"ch_{i}")
        return labels
    except Exception:
        return [f"ch_{i}" for i in range(n_channels)]


# Stream name -> modality for EEG/MOTION/TEXT (aligned with xdf_files.stream_name_to_modality_dict). Other names are not logged.
STREAM_NAME_TO_MODALITY = {"Epoc X": "EEG", "Epoc X Motion": "MOTION", "TextLogger": "TEXT", "EventBoard": "TEXT"}


def _stream_modality(stream: dict) -> str | None:
    """Return 'EEG', 'MOTION', 'TEXT', or None from stream name (no external deps)."""
    return STREAM_NAME_TO_MODALITY.get(_stream_name(stream))


def _stream_name(stream: dict) -> str:
    """Get a short stream name for entity path."""
    try:
        name = stream.get("info", {}).get("name", [None])
        if name and name[0] is not None:
            return str(name[0]).strip() or "stream"
    except Exception:
        pass
    return "stream"


def _global_t0_from_streams(streams: list) -> float:
    """Minimum first timestamp across all streams that have time_stamps (for shared timeline)."""
    t0_candidates = []
    for stream in streams:
        if not stream:
            continue
        ts = stream.get("time_stamps")
        if ts is None or (hasattr(ts, "__len__") and len(ts) == 0):
            continue
        ts = np.asarray(ts).flatten()
        if ts.size > 0:
            t0_candidates.append(float(ts[0]))
    return min(t0_candidates) if t0_candidates else 0.0


def _log_xdf_streams_imu_style(streams: list, entity_path_prefix: str, t0: float) -> None:
    """Log first EEG at xdf/EEG/<channel> (one stacked 1D plot per channel) and first MOTION at xdf/MOTION (one panel)."""
    prefix = f"{entity_path_prefix}xdf/".replace("//", "/")
    logged = {"EEG": False, "MOTION": False}
    for stream in streams:
        if not stream or len(stream.get("time_series", [])) == 0:
            continue
        time_series = np.asarray(stream["time_series"])
        time_stamps = np.asarray(stream["time_stamps"]).flatten()
        n_samples, n_channels = time_series.shape
        if time_stamps.size != n_samples:
            continue
        if not np.issubdtype(time_series.dtype, np.number):
            continue
        modality = _stream_modality(stream)
        if modality not in ("EEG", "MOTION") or logged[modality]:
            continue
        time_sec = (time_stamps - t0).astype(np.float64)
        channel_labels = _channel_labels_from_stream(stream, n_channels)
        if modality == "EEG":
            for i in range(n_channels):
                ch_path = f"{prefix}EEG/{_sanitize_entity_name(channel_labels[i])}"
                ch_scalars = time_series[:, i : i + 1]
                rr.send_columns(ch_path, indexes=[rr.TimeColumn("time_sec", duration=time_sec)], columns=rr.Scalars.columns(scalars=ch_scalars))
                rr.log(ch_path, rr.SeriesLines(names=[channel_labels[i]]), static=True)
        else:
            path = f"{prefix}{modality}"
            rr.send_columns(path, indexes=[rr.TimeColumn("time_sec", duration=time_sec)], columns=rr.Scalars.columns(scalars=time_series))
            rr.log(path, rr.SeriesLines(names=channel_labels), static=True)
        logged[modality] = True
        if logged["EEG"] and logged["MOTION"]:
            break


def _log_xdf_text_streams_merged(streams: list, entity_path_prefix: str, t0: float, text_stream_names: list[str] | None = None) -> None:
    """Merge all TextLogger/EventBoard streams into one entity; log (t_sec, text) sorted by time to xdf/Text."""
    if text_stream_names is None:
        text_stream_names = ["TextLogger", "EventBoard"]
    path = f"{entity_path_prefix}xdf/Text".replace("//", "/")
    entries = []
    for stream in streams:
        if not stream or _stream_name(stream) not in text_stream_names:
            continue
        time_series = stream.get("time_series")
        if time_series is None or (hasattr(time_series, "__len__") and len(time_series) == 0):
            continue
        time_series = np.asarray(time_series)
        if time_series.ndim > 1:
            time_series = time_series.ravel()
        time_stamps = np.asarray(stream["time_stamps"]).flatten()
        n = min(len(time_stamps), len(time_series))
        for i in range(n):
            t_sec = float(time_stamps[i] - t0)
            val = time_series[i]
            text = str(val) if np.isscalar(val) else (str(val.tolist()) if hasattr(val, "tolist") else str(val))
            entries.append((t_sec, text))
    entries.sort(key=lambda x: x[0])
    for t_sec, text in entries:
        rr.set_time("time_sec", duration=t_sec)
        rr.log(path, rr.TextLog(text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun external data loader for .xdf files. Exit with code 66 if file is not .xdf.")
    parser.add_argument("filepath", type=str, nargs="?", default=None, help="Path to the file to load (positional)")
    parser.add_argument("--recording-id", type=str, default=None, help="Recommended recording ID from Rerun Viewer")
    parser.add_argument("--application-id", type=str, default=None, help="Recommended application ID from Rerun Viewer")
    parser.add_argument("--opened-recording-id", type=str, default=None, help="Recording ID currently opened in the viewer")
    parser.add_argument("--opened-application-id", type=str, default=None, help="Application ID currently opened in the viewer")
    parser.add_argument("--entity-path-prefix", type=str, default="", help="Prefix for entity paths")
    args = parser.parse_args()

    filepath = args.filepath
    if not filepath and args.opened_application_id and os.path.isfile(args.opened_application_id) and args.opened_application_id.lower().endswith(".xdf"):
        filepath = args.opened_application_id
        args.opened_application_id = None
    if not filepath:
        print("rerun-loader-xdf: missing filepath (positional or via --opened-application-id).", file=sys.stderr)
        return 1
    is_file = os.path.isfile(filepath)
    is_xdf = os.path.splitext(filepath)[1].lower() == ".xdf"
    if not is_file or not is_xdf:
        sys.exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    try:
        streams, header = pyxdf.load_xdf(filepath, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False, verbose=False)
    except Exception as e:
        print(f"rerun-loader-xdf: Failed to load XDF: {e}", file=sys.stderr)
        return 1

    application_id = args.opened_application_id or args.application_id or "rerun_loader_xdf"
    recording_id = args.opened_recording_id or args.recording_id
    if recording_id is None:
        file_dt = _file_datetime_from_header(header)
        if file_dt is not None:
            recording_id = file_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rr.init(application_id, recording_id=recording_id)
    rr.stdout()

    prefix = (args.entity_path_prefix.rstrip("/") + "/") if args.entity_path_prefix else ""
    t0 = _global_t0_from_streams(streams)
    _log_xdf_streams_imu_style(streams, prefix, t0)
    _log_xdf_text_streams_merged(streams, prefix, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
