#!/usr/bin/env python3
"""Export frames found by find_two_bi7_in_generator_h5.py to XYZ."""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np


REQUIRED_COLUMNS = {
    "trajectory_id",
    "h5_file",
    "frame_index",
    "md_step",
    "temperature_K",
    "cluster_a_indices",
    "cluster_b_indices",
    "minimum_intercluster_distance_A",
    "centroid_distance_A",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert matched Bi7 + Bi7 HDF5 frames to a Jmol XYZ trajectory."
    )
    parser.add_argument(
        "matches_csv",
        type=Path,
        help="CSV produced by find_two_bi7_in_generator_h5.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output XYZ path (default: <matches_csv stem>.xyz)",
    )
    parser.add_argument(
        "--h5-root",
        type=Path,
        help=(
            "Directory from which relative h5_file paths are resolved. By default, "
            "the current directory and the CSV directory are tried."
        ),
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Export every Nth selected CSV row (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Maximum number of XYZ models to export (default: all)",
    )
    parser.add_argument(
        "--one-per-episode",
        action="store_true",
        help="Export only the first matching frame from each trajectory episode",
    )
    return parser.parse_args()


def resolve_h5_path(raw_path: str, csv_path: Path, h5_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        candidates = [path]
    elif h5_root is not None:
        candidates = [h5_root / path]
    else:
        candidates = [Path.cwd() / path, csv_path.parent / path]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"HDF5 file {raw_path!r} was not found; tried: {tried}")


def cluster_atom_numbers(value: str) -> str:
    """Convert the CSV's zero-based indices to Jmol's one-based atom numbers."""
    return " ".join(str(int(index) + 1) for index in value.split())


def xyz_comment(row: dict[str, str], source: Path, frame_index: int) -> str:
    cluster_a = cluster_atom_numbers(row["cluster_a_indices"])
    cluster_b = cluster_atom_numbers(row["cluster_b_indices"])
    return (
        f"trajectory={row['trajectory_id']} source={source.name} frame={frame_index} "
        f"md_step={row['md_step']} temperature_K={float(row['temperature_K']):.3f} "
        f"min_intercluster_A={float(row['minimum_intercluster_distance_A']):.6f} "
        f"centroid_distance_A={float(row['centroid_distance_A']):.6f} "
        f"cluster_A_atoms={cluster_a} cluster_B_atoms={cluster_b}"
    )


def main() -> int:
    args = parse_args()
    csv_path = args.matches_csv.resolve()
    output_path = (args.output or args.matches_csv.with_suffix(".xyz")).resolve()
    h5_root = args.h5_root.resolve() if args.h5_root is not None else None

    if not csv_path.is_file():
        print(f"Error: match CSV does not exist: {csv_path}", file=sys.stderr)
        return 2
    if output_path == csv_path:
        print("Error: output path must differ from the input CSV", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    selected_rows = 0
    seen_episodes: set[str] = set()

    try:
        with ExitStack() as stack:
            input_handle = stack.enter_context(csv_path.open(newline=""))
            reader = csv.DictReader(input_handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"match CSV is missing columns: {sorted(missing)}")

            output_handle = stack.enter_context(output_path.open("w"))
            h5_files: dict[Path, h5py.File] = {}

            for row_number, row in enumerate(reader, start=2):
                episode = row["trajectory_id"]
                if args.one_per_episode and episode in seen_episodes:
                    continue
                if selected_rows % args.stride != 0:
                    selected_rows += 1
                    continue
                selected_rows += 1
                seen_episodes.add(episode)

                source = resolve_h5_path(row["h5_file"], csv_path, h5_root)
                if source not in h5_files:
                    h5_files[source] = stack.enter_context(h5py.File(source, "r"))
                h5_file = h5_files[source]
                if "coordinates" not in h5_file:
                    raise ValueError(f"{source} has no 'coordinates' dataset")

                frame_index = int(row["frame_index"])
                coordinates_dataset = h5_file["coordinates"]
                if frame_index < 0 or frame_index >= len(coordinates_dataset):
                    raise ValueError(
                        f"CSV row {row_number}: frame {frame_index} is outside "
                        f"{source} (frames: {len(coordinates_dataset)})"
                    )
                coordinates = np.asarray(coordinates_dataset[frame_index], dtype=float)
                if coordinates.shape != (14, 3) or not np.isfinite(coordinates).all():
                    raise ValueError(
                        f"CSV row {row_number}: expected finite coordinates with "
                        f"shape (14, 3), got {coordinates.shape}"
                    )

                output_handle.write("14\n")
                output_handle.write(xyz_comment(row, source, frame_index) + "\n")
                for x, y, z in coordinates:
                    output_handle.write(f"Bi {x:.10f} {y:.10f} {z:.10f}\n")
                exported += 1
                if args.limit is not None and exported >= args.limit:
                    break
    except (OSError, TypeError, ValueError) as exc:
        if output_path.exists():
            output_path.unlink()
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Exported {exported} matched frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
