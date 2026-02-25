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
from pathlib import Path

import numpy as np
import pyxdf
import rerun as rr


def _sanitize_entity_name(name: str) -> str:
    """Replace characters that are invalid in Rerun entity paths."""
    if not name:
        return "stream"
    return re.sub(r"[\/\\]", "_", name)


def _channel_labels_from_stream(stream: dict, n_channels: int) -> list[str]:
    """Derive channel labels from XDF stream info, or ch_0, ch_1, ..."""
    try:
        desc = stream.get("info", {}).get("desc", [None])
        if desc and desc[0] is not None:
            channels = desc[0].get("channels", [None])
            if channels and channels[0] is not None:
                ch_list = channels[0].get("channel", [])
                if isinstance(ch_list, list) and len(ch_list) >= n_channels:
                    return [str(c.get("label", c) if isinstance(c, dict) else c) for c in ch_list[:n_channels]]
    except Exception:
        pass
    return [f"ch_{i}" for i in range(n_channels)]


def _stream_name(stream: dict) -> str:
    """Get a short stream name for entity path."""
    try:
        name = stream.get("info", {}).get("name", [None])
        if name and name[0] is not None:
            return str(name[0]).strip() or "stream"
    except Exception:
        pass
    return "stream"


def _log_xdf_streams_imu_style(streams: list, header: dict | None, entity_path_prefix: str) -> None:
    """Log each numeric XDF stream as one entity with multi-channel scalars (IMU-style)."""
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
        t0 = float(time_stamps[0])
        time_sec = (time_stamps - t0).astype(np.float64)
        safe_name = _sanitize_entity_name(_stream_name(stream))
        path = f"{entity_path_prefix}xdf/{safe_name}".replace("//", "/")
        rr.send_columns(path, indexes=[rr.TimeColumn("time_sec", duration=time_sec)], columns=rr.Scalars.columns(scalars=time_series))


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

    application_id = args.opened_application_id or args.application_id or "rerun_loader_xdf"
    recording_id = args.opened_recording_id or args.recording_id
    rr.init(application_id, recording_id=recording_id)
    rr.stdout()

    try:
        streams, header = pyxdf.load_xdf(filepath, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False, verbose=False)
    except Exception as e:
        print(f"rerun-loader-xdf: Failed to load XDF: {e}", file=sys.stderr)
        return 1

    prefix = (args.entity_path_prefix.rstrip("/") + "/") if args.entity_path_prefix else ""
    _log_xdf_streams_imu_style(streams, header, prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
