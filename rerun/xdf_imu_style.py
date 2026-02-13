"""
Run XDF-to-Rerun conversion in IMU-style: one time-series panel per stream with multiple channels.

Mirrors https://rerun.io/examples/feature-showcase/imu_signals
Run from project root: uv run python rerun/xdf_imu_style.py path/to/file.xdf
"""

from pathlib import Path
import sys

# Ensure package is on path when run as script
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pyxdf
from phoofflineeeganalysis.converters.xdf_to_rerun import stream_xdf_to_rerun_imu_style


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stream XDF file data into a Rerun recording (IMU-style: one panel per stream, multi-channel).")
    parser.add_argument("xdf_path", type=Path, help="Path to the XDF file")
    parser.add_argument("--save", "-o", type=Path, default=None, help="Save recording to this .rrd path (default: same stem as XDF)")
    parser.add_argument("--no-spawn", action="store_true", help="Do not spawn the Rerun viewer")
    parser.add_argument("--step", type=int, default=1, help="Downsample to every Nth sample (default: 1)")
    args = parser.parse_args()
    path = args.xdf_path.resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    save_path = args.save if args.save is not None else path.with_suffix(".rrd")
    streams, header = pyxdf.load_xdf(str(path), synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False, verbose=False)
    stream_xdf_to_rerun_imu_style((streams, header), save_path=save_path, spawn=not args.no_spawn, step=args.step)


if __name__ == "__main__":
    main()
