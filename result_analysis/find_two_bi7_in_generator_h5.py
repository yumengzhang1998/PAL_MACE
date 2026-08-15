#!/usr/bin/env python3
"""Find separated Bi7 + Bi7 structures in saved AL generator trajectories.

The input is one or more HDF5 shards written by ``save_gene_traj=True``, or a
directory containing such shards.  Files are scanned in bounded chunks; the
coordinate dataset is never loaded fully into memory.

Run this analysis after the AL job has closed its HDF5 files.  The writer does
not use HDF5 SWMR mode, so reading a live shard is unsafe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from find_two_bi7 import (
    BI_ATOMIC_NUMBER,
    DEFAULT_REFERENCE,
    connected_components,
    degree_signature,
    detect_two_bi7,
    distance_matrix,
    load_reference,
    pair_distance_spectrum,
    positive_float,
)


REQUIRED_FRAME_DATASETS = (
    "coordinates",
    "slot",
    "episode",
    "md_step",
    "model_iteration",
    "pool_index",
    "temperature_K",
)

OUTPUT_COLUMNS = (
    "trajectory_id",
    "h5_file",
    "generator_rank",
    "frame_index",
    "slot",
    "episode",
    "md_step",
    "model_iteration",
    "pool_index",
    "temperature_K",
    "cluster_a_indices",
    "cluster_b_indices",
    "pair_distance_rmsd_a_A",
    "pair_distance_rmsd_b_A",
    "max_pair_distance_error_a_A",
    "max_pair_distance_error_b_A",
    "minimum_intercluster_distance_A",
    "centroid_distance_A",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def discover_h5_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            if input_path.suffix.lower() not in {".h5", ".hdf5"}:
                raise ValueError(f"Input is not an HDF5 file: {input_path}")
            files.append(input_path)
        elif input_path.is_dir():
            files.extend(input_path.rglob("generator_rank_*.h5"))
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    unique_files: dict[Path, Path] = {}
    for path in files:
        unique_files[path.resolve()] = path
    ordered = sorted(unique_files.values(), key=lambda path: str(path))
    if not ordered:
        raise ValueError("No generator_rank_*.h5 files were found")
    return ordered


def validate_h5_file(
    h5_file: h5py.File,
    path: Path,
    *,
    allow_unclean: bool,
) -> tuple[int, int]:
    schema = h5_file.attrs.get("schema", "")
    if isinstance(schema, bytes):
        schema = schema.decode(errors="replace")
    if schema != "pal_mace_generator_trajectory_v1":
        raise ValueError(
            f"{path} has unsupported or missing trajectory schema: {schema!r}"
        )

    closed_cleanly = bool(h5_file.attrs.get("closed_cleanly", False))
    if not closed_cleanly and not allow_unclean:
        raise ValueError(
            f"{path} is not marked closed_cleanly. The job may still be writing "
            "it; wait for completion or use --allow-unclean only after confirming "
            "that no writer is active."
        )

    required = set(REQUIRED_FRAME_DATASETS) | {"atomic_numbers"}
    missing = required.difference(h5_file.keys())
    if missing:
        raise ValueError(f"{path} is missing datasets: {sorted(missing)}")

    atomic_numbers = np.asarray(h5_file["atomic_numbers"][:]).reshape(-1)
    if atomic_numbers.shape != (14,) or not np.all(
        atomic_numbers == BI_ATOMIC_NUMBER
    ):
        raise ValueError(
            f"{path} is not a Bi14 trajectory shard; atomic numbers are "
            f"{atomic_numbers.tolist()}"
        )

    coordinates = h5_file["coordinates"]
    if coordinates.ndim != 3 or coordinates.shape[1:] != (14, 3):
        raise ValueError(
            f"{path}: coordinates must have shape (frames, 14, 3), got "
            f"{coordinates.shape}"
        )

    frame_count = int(coordinates.shape[0])
    for name in REQUIRED_FRAME_DATASETS[1:]:
        dataset = h5_file[name]
        if dataset.shape != (frame_count,):
            raise ValueError(
                f"{path}: dataset {name!r} has shape {dataset.shape}, expected "
                f"({frame_count},)"
            )

    committed_frames = int(h5_file.attrs.get("committed_frames", frame_count))
    if committed_frames != frame_count:
        raise ValueError(
            f"{path}: committed_frames={committed_frames}, but datasets contain "
            f"{frame_count} frames"
        )

    generator_rank = int(h5_file.attrs.get("generator_rank", -1))
    return generator_rank, frame_count


def rejection_counter_key(reason: object) -> str:
    if reason == "not_two_disconnected_bi7_components":
        return "component_rejections"
    if reason == "bi7_bond_topology_mismatch":
        return "topology_rejections"
    return "shape_rejections"


def scan_h5_file(
    path: Path,
    *,
    reference_spectrum: np.ndarray,
    reference_topology: tuple[int, ...],
    bond_cutoff: float,
    distance_rmsd_tolerance: float,
    max_distance_error: float,
    require_topology: bool,
    chunksize: int,
    allow_unclean: bool,
    emit_match: Callable[[dict[str, object]], None],
    stop_after_first: bool,
) -> tuple[Counter, bool]:
    counts: Counter = Counter()
    with h5py.File(path, "r") as h5_file:
        generator_rank, frame_count = validate_h5_file(
            h5_file, path, allow_unclean=allow_unclean
        )
        counts["files"] = 1

        for start in range(0, frame_count, chunksize):
            stop = min(start + chunksize, frame_count)
            coordinates = np.asarray(h5_file["coordinates"][start:stop])
            metadata = {
                name: np.asarray(h5_file[name][start:stop])
                for name in REQUIRED_FRAME_DATASETS[1:]
            }

            for offset, frame_coordinates in enumerate(coordinates):
                frame_index = start + offset
                counts["frames_scanned"] += 1
                if not np.isfinite(frame_coordinates).all():
                    counts["malformed_frames"] += 1
                    continue

                matched, details = detect_two_bi7(
                    frame_coordinates,
                    reference_spectrum=reference_spectrum,
                    reference_degree_signature=reference_topology,
                    bond_cutoff=bond_cutoff,
                    distance_rmsd_tolerance=distance_rmsd_tolerance,
                    max_distance_error=max_distance_error,
                    require_topology=require_topology,
                )
                if not matched:
                    counts[rejection_counter_key(details["reason"])] += 1
                    continue

                slot = int(metadata["slot"][offset])
                episode = int(metadata["episode"][offset])
                trajectory_id = (
                    f"{path.stem}:r{generator_rank}:s{slot}:e{episode}"
                )
                record = {
                    "trajectory_id": trajectory_id,
                    "h5_file": str(path),
                    "generator_rank": generator_rank,
                    "frame_index": frame_index,
                    "slot": slot,
                    "episode": episode,
                    "md_step": int(metadata["md_step"][offset]),
                    "model_iteration": int(
                        metadata["model_iteration"][offset]
                    ),
                    "pool_index": int(metadata["pool_index"][offset]),
                    "temperature_K": float(metadata["temperature_K"][offset]),
                    "cluster_a_indices": " ".join(
                        map(str, details["cluster_a"])
                    ),
                    "cluster_b_indices": " ".join(
                        map(str, details["cluster_b"])
                    ),
                    "pair_distance_rmsd_a_A": float(
                        details["distance_rmsd_a"]
                    ),
                    "pair_distance_rmsd_b_A": float(
                        details["distance_rmsd_b"]
                    ),
                    "max_pair_distance_error_a_A": float(
                        details["max_distance_error_a"]
                    ),
                    "max_pair_distance_error_b_A": float(
                        details["max_distance_error_b"]
                    ),
                    "minimum_intercluster_distance_A": float(
                        details["minimum_intercluster_distance"]
                    ),
                    "centroid_distance_A": float(
                        details["centroid_distance"]
                    ),
                }
                counts["matching_frames"] += 1
                emit_match(record)
                if stop_after_first:
                    return counts, True

    return counts, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find two separated optimized-Bi7-like clusters in compact AL "
            "generator HDF5 trajectories."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Generator HDF5 file(s), or directories recursively containing "
            "generator_rank_*.h5 files"
        ),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"Geometry reference CSV (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--reference-name",
        default="Bi7-3",
        help="Name of the optimized seven-atom reference (default: Bi7-3)",
    )
    parser.add_argument(
        "--bond-cutoff",
        type=positive_float,
        default=3.5,
        help="Maximum Bi--Bi bonded distance in angstrom (default: 3.5)",
    )
    parser.add_argument(
        "--distance-rmsd-tolerance",
        type=positive_float,
        default=0.35,
        help="Maximum Bi7 pair-distance RMS error in angstrom (default: 0.35)",
    )
    parser.add_argument(
        "--max-distance-error",
        type=positive_float,
        default=0.75,
        help="Maximum individual pair-distance error in angstrom (default: 0.75)",
    )
    parser.add_argument(
        "--allow-topology-mismatch",
        action="store_true",
        help="Do not require the Bi7 bond-degree signature to match the reference",
    )
    parser.add_argument(
        "--allow-unclean",
        action="store_true",
        help=(
            "Read a shard not marked closed_cleanly. Never use this while its "
            "AL generator is still running."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=positive_int,
        default=4096,
        help="Number of saved frames read per HDF5 chunk (default: 4096)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV receiving one row for every matching saved frame",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Maximum matching frames printed to the terminal (default: 20)",
    )
    parser.add_argument(
        "--stop-after-first",
        action="store_true",
        help="Stop as soon as the first matching frame is found",
    )
    args = parser.parse_args()
    if args.max_print < 0:
        parser.error("--max-print must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        h5_paths = discover_h5_files(args.inputs)
        reference_coordinates = load_reference(
            args.reference_csv, args.reference_name
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    reference_spectrum = pair_distance_spectrum(reference_coordinates)
    reference_topology = degree_signature(
        reference_coordinates, args.bond_cutoff
    )
    reference_components = connected_components(
        distance_matrix(reference_coordinates), args.bond_cutoff
    )
    if len(reference_components) != 1:
        print(
            "Error: optimized Bi7 is not connected at the requested bond "
            f"cutoff {args.bond_cutoff:.3f} A",
            file=sys.stderr,
        )
        return 2

    if args.output is not None:
        output_resolved = args.output.resolve()
        if any(output_resolved == path.resolve() for path in h5_paths):
            print("Error: output CSV must not overwrite an input HDF5 file", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)

    total_counts: Counter = Counter()
    matching_episodes: set[str] = set()
    printed = 0
    output_handle = None
    writer = None

    try:
        if args.output is not None:
            output_handle = args.output.open("w", newline="")
            writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

        def emit_match(record: dict[str, object]) -> None:
            nonlocal printed
            matching_episodes.add(str(record["trajectory_id"]))
            if writer is not None:
                writer.writerow(record)
            if printed < args.max_print:
                print(
                    f"{record['trajectory_id']} "
                    f"frame={record['frame_index']} "
                    f"step={record['md_step']} "
                    f"min_cross={record['minimum_intercluster_distance_A']:.4f} A "
                    f"pair_RMSD=({record['pair_distance_rmsd_a_A']:.4f}, "
                    f"{record['pair_distance_rmsd_b_A']:.4f}) A"
                )
                printed += 1

        stopped_early = False
        for path in h5_paths:
            file_counts, stopped_early = scan_h5_file(
                path,
                reference_spectrum=reference_spectrum,
                reference_topology=reference_topology,
                bond_cutoff=args.bond_cutoff,
                distance_rmsd_tolerance=args.distance_rmsd_tolerance,
                max_distance_error=args.max_distance_error,
                require_topology=not args.allow_topology_mismatch,
                chunksize=args.chunksize,
                allow_unclean=args.allow_unclean,
                emit_match=emit_match,
                stop_after_first=args.stop_after_first,
            )
            total_counts.update(file_counts)
            if stopped_early:
                break
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if output_handle is not None:
            output_handle.close()

    print(
        f"Scanned {total_counts['frames_scanned']} saved frames from "
        f"{total_counts['files']} HDF5 files."
    )
    print(
        f"Found {total_counts['matching_frames']} matching frames across "
        f"{len(matching_episodes)} trajectory episodes."
    )
    print(
        "Rejected: "
        f"components={total_counts['component_rejections']}, "
        f"topology={total_counts['topology_rejections']}, "
        f"shape={total_counts['shape_rejections']}, "
        f"malformed={total_counts['malformed_frames']}."
    )
    if args.output is not None:
        print(f"Wrote matching-frame summary to {args.output}")
    if args.stop_after_first and total_counts["matching_frames"]:
        print("Stopped after the first match as requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
