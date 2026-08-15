#!/usr/bin/env python3
"""Find separated Bi7 + Bi7 structures in an AL added-data CSV.

A match must satisfy all of the following:

1. The row contains exactly 14 Bi atoms.
2. A Bi--Bi neighbor graph splits into exactly two connected components of
   seven atoms each.  Consequently, every distance between the two components
   is larger than ``--bond-cutoff``.
3. Each component resembles the optimized Bi7 reference.  Similarity is
   measured using the sorted 21 pair distances, which is invariant to
   translation, rotation, reflection, and atom ordering.

By default, rows marked ``init == 1`` are skipped because ``*_added_data.csv``
contains both the initial training set and structures acquired during active
learning.  Pass ``--include-initial`` to scan both sets.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BI_ATOMIC_NUMBER = 83
DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / "optimized.csv"


@dataclass(frozen=True)
class Match:
    row_index: int
    cluster_a: tuple[int, ...]
    cluster_b: tuple[int, ...]
    distance_rmsd_a: float
    distance_rmsd_b: float
    max_distance_error_a: float
    max_distance_error_b: float
    minimum_intercluster_distance: float
    centroid_distance: float
    energy: object
    split: object
    data_type: object


def parse_literal(value: object, *, column: str, row_index: object) -> object:
    """Parse a list stored as text in one of the project CSV files."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"row {row_index}: could not parse column {column!r}: {exc}"
        ) from exc


def parse_coordinates(value: object, *, row_index: object) -> np.ndarray:
    coordinates = np.asarray(
        parse_literal(value, column="coordinates", row_index=row_index),
        dtype=np.float64,
    )
    if coordinates.shape != (14, 3):
        raise ValueError(
            f"row {row_index}: expected coordinates with shape (14, 3), "
            f"got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"row {row_index}: coordinates contain NaN or infinity")
    return coordinates


def is_bismuth_atom(value: object) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "bi":
            return True
        try:
            return int(stripped) == BI_ATOMIC_NUMBER
        except ValueError:
            return False
    try:
        return int(value) == BI_ATOMIC_NUMBER
    except (TypeError, ValueError, OverflowError):
        return False


def row_is_bi14(row: pd.Series, *, row_index: object) -> bool:
    """Validate atom identities when an atoms column is available."""
    if "atoms" not in row.index:
        return True
    atom_value = row["atoms"]
    if not isinstance(atom_value, (list, tuple, np.ndarray)) and pd.isna(atom_value):
        return True
    atoms = parse_literal(atom_value, column="atoms", row_index=row_index)
    return len(atoms) == 14 and all(is_bismuth_atom(atom) for atom in atoms)


def distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    displacement = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(displacement, axis=-1)


def pair_distance_spectrum(coordinates: np.ndarray) -> np.ndarray:
    distances = distance_matrix(coordinates)
    return np.sort(distances[np.triu_indices(len(coordinates), k=1)])


def connected_components(distances: np.ndarray, cutoff: float) -> list[np.ndarray]:
    adjacency = (distances <= cutoff) & (distances > 0.0)
    unseen = set(range(len(distances)))
    components: list[np.ndarray] = []

    while unseen:
        start = unseen.pop()
        stack = [start]
        component = [start]
        while stack:
            atom = stack.pop()
            neighbors = np.flatnonzero(adjacency[atom])
            for neighbor_value in neighbors:
                neighbor = int(neighbor_value)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))

    components.sort(key=lambda indices: (int(indices[0]), len(indices)))
    return components


def degree_signature(coordinates: np.ndarray, cutoff: float) -> tuple[int, ...]:
    distances = distance_matrix(coordinates)
    degrees = np.sum((distances <= cutoff) & (distances > 0.0), axis=1)
    return tuple(sorted(int(value) for value in degrees))


def load_reference(
    reference_csv: Path,
    reference_name: str,
) -> np.ndarray:
    frame = pd.read_csv(reference_csv)
    required = {"Name", "coord"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{reference_csv} is missing required columns: {sorted(missing)}"
        )

    selected = frame[
        frame["Name"].astype(str).str.casefold() == reference_name.casefold()
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected exactly one {reference_name!r} row in {reference_csv}, "
            f"found {len(selected)}"
        )

    coordinates = np.asarray(
        parse_literal(
            selected.iloc[0]["coord"],
            column="coord",
            row_index=selected.index[0],
        ),
        dtype=np.float64,
    )
    if coordinates.shape != (7, 3):
        raise ValueError(
            f"reference {reference_name!r} must have shape (7, 3), "
            f"got {coordinates.shape}"
        )
    return coordinates


def compare_to_reference(
    coordinates: np.ndarray,
    reference_spectrum: np.ndarray,
) -> tuple[float, float]:
    difference = pair_distance_spectrum(coordinates) - reference_spectrum
    rmsd = float(np.sqrt(np.mean(np.square(difference))))
    maximum_error = float(np.max(np.abs(difference)))
    return rmsd, maximum_error


def detect_two_bi7(
    coordinates: np.ndarray,
    *,
    reference_spectrum: np.ndarray,
    reference_degree_signature: tuple[int, ...],
    bond_cutoff: float,
    distance_rmsd_tolerance: float,
    max_distance_error: float,
    require_topology: bool,
) -> tuple[bool, dict[str, object]]:
    distances = distance_matrix(coordinates)
    components = connected_components(distances, bond_cutoff)

    if sorted(len(component) for component in components) != [7, 7]:
        return False, {"reason": "not_two_disconnected_bi7_components"}

    cluster_a, cluster_b = components
    coordinates_a = coordinates[cluster_a]
    coordinates_b = coordinates[cluster_b]

    topology_a = degree_signature(coordinates_a, bond_cutoff)
    topology_b = degree_signature(coordinates_b, bond_cutoff)
    if require_topology and (
        topology_a != reference_degree_signature
        or topology_b != reference_degree_signature
    ):
        return False, {"reason": "bi7_bond_topology_mismatch"}

    rmsd_a, maximum_a = compare_to_reference(
        coordinates_a, reference_spectrum
    )
    rmsd_b, maximum_b = compare_to_reference(
        coordinates_b, reference_spectrum
    )

    if max(rmsd_a, rmsd_b) > distance_rmsd_tolerance:
        return False, {"reason": "pair_distance_rmsd_too_large"}
    if max(maximum_a, maximum_b) > max_distance_error:
        return False, {"reason": "maximum_pair_distance_error_too_large"}

    cross_distances = distances[np.ix_(cluster_a, cluster_b)]
    centroid_distance = float(
        np.linalg.norm(
            np.mean(coordinates_a, axis=0) - np.mean(coordinates_b, axis=0)
        )
    )
    details: dict[str, object] = {
        "cluster_a": tuple(int(index) for index in cluster_a),
        "cluster_b": tuple(int(index) for index in cluster_b),
        "distance_rmsd_a": rmsd_a,
        "distance_rmsd_b": rmsd_b,
        "max_distance_error_a": maximum_a,
        "max_distance_error_b": maximum_b,
        "minimum_intercluster_distance": float(np.min(cross_distances)),
        "centroid_distance": centroid_distance,
    }
    return True, details


def is_added_row(row: pd.Series) -> bool:
    if "init" not in row.index:
        return True
    try:
        return float(row["init"]) == 0.0
    except (TypeError, ValueError):
        return False


def optional_value(row: pd.Series, *columns: str) -> object:
    for column in columns:
        if column in row.index:
            return row[column]
    return None


def scan_rows(
    chunks: Iterable[pd.DataFrame],
    *,
    reference_coordinates: np.ndarray,
    bond_cutoff: float,
    distance_rmsd_tolerance: float,
    max_distance_error: float,
    include_initial: bool,
    require_topology: bool,
) -> tuple[list[Match], dict[str, int]]:
    reference_spectrum = pair_distance_spectrum(reference_coordinates)
    reference_topology = degree_signature(reference_coordinates, bond_cutoff)
    reference_components = connected_components(
        distance_matrix(reference_coordinates), bond_cutoff
    )
    if len(reference_components) != 1:
        raise ValueError(
            "the optimized Bi7 reference is not connected at bond cutoff "
            f"{bond_cutoff:.3f} A; choose a larger --bond-cutoff"
        )

    counts = {
        "csv_rows": 0,
        "initial_rows_skipped": 0,
        "non_bi14_rows": 0,
        "malformed_rows": 0,
        "candidate_rows": 0,
        "component_rejections": 0,
        "topology_rejections": 0,
        "shape_rejections": 0,
        "matches": 0,
    }
    matches: list[Match] = []

    for chunk in chunks:
        coordinate_column = (
            "coordinates"
            if "coordinates" in chunk.columns
            else "node_feature"
            if "node_feature" in chunk.columns
            else None
        )
        if coordinate_column is None:
            raise ValueError(
                "input CSV must contain a 'coordinates' or 'node_feature' column"
            )

        for row_index, row in chunk.iterrows():
            counts["csv_rows"] += 1
            if not include_initial and not is_added_row(row):
                counts["initial_rows_skipped"] += 1
                continue

            try:
                if not row_is_bi14(row, row_index=row_index):
                    counts["non_bi14_rows"] += 1
                    continue
                coordinates = parse_coordinates(
                    row[coordinate_column], row_index=row_index
                )
            except (TypeError, ValueError):
                counts["malformed_rows"] += 1
                continue

            counts["candidate_rows"] += 1
            matched, details = detect_two_bi7(
                coordinates,
                reference_spectrum=reference_spectrum,
                reference_degree_signature=reference_topology,
                bond_cutoff=bond_cutoff,
                distance_rmsd_tolerance=distance_rmsd_tolerance,
                max_distance_error=max_distance_error,
                require_topology=require_topology,
            )
            if not matched:
                reason = details["reason"]
                if reason == "not_two_disconnected_bi7_components":
                    counts["component_rejections"] += 1
                elif reason == "bi7_bond_topology_mismatch":
                    counts["topology_rejections"] += 1
                else:
                    counts["shape_rejections"] += 1
                continue

            counts["matches"] += 1
            matches.append(
                Match(
                    row_index=int(row_index),
                    cluster_a=details["cluster_a"],
                    cluster_b=details["cluster_b"],
                    distance_rmsd_a=float(details["distance_rmsd_a"]),
                    distance_rmsd_b=float(details["distance_rmsd_b"]),
                    max_distance_error_a=float(details["max_distance_error_a"]),
                    max_distance_error_b=float(details["max_distance_error_b"]),
                    minimum_intercluster_distance=float(
                        details["minimum_intercluster_distance"]
                    ),
                    centroid_distance=float(details["centroid_distance"]),
                    energy=optional_value(row, "energy", "total_energy"),
                    split=optional_value(row, "init"),
                    data_type=optional_value(row, "type"),
                )
            )

    return matches, counts


def matches_to_frame(matches: list[Match]) -> pd.DataFrame:
    columns = [
        "row_index",
        "cluster_a_indices",
        "cluster_b_indices",
        "pair_distance_rmsd_a_A",
        "pair_distance_rmsd_b_A",
        "max_pair_distance_error_a_A",
        "max_pair_distance_error_b_A",
        "minimum_intercluster_distance_A",
        "centroid_distance_A",
        "energy",
        "init",
        "type",
    ]
    return pd.DataFrame(
        [
            {
                "row_index": match.row_index,
                "cluster_a_indices": " ".join(map(str, match.cluster_a)),
                "cluster_b_indices": " ".join(map(str, match.cluster_b)),
                "pair_distance_rmsd_a_A": match.distance_rmsd_a,
                "pair_distance_rmsd_b_A": match.distance_rmsd_b,
                "max_pair_distance_error_a_A": match.max_distance_error_a,
                "max_pair_distance_error_b_A": match.max_distance_error_b,
                "minimum_intercluster_distance_A": (
                    match.minimum_intercluster_distance
                ),
                "centroid_distance_A": match.centroid_distance,
                "energy": match.energy,
                "init": match.split,
                "type": match.data_type,
            }
            for match in matches
        ],
        columns=columns,
    )


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find Bi14 structures composed of two separated, optimized-Bi7-like "
            "clusters."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to results/<prefix>/<model_number>_added_data.csv",
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
        help="Name of the seven-atom reference row (default: Bi7-3)",
    )
    parser.add_argument(
        "--bond-cutoff",
        type=positive_float,
        default=3.5,
        help=(
            "Maximum Bi--Bi distance considered a bond, in angstrom "
            "(default: 3.5)"
        ),
    )
    parser.add_argument(
        "--distance-rmsd-tolerance",
        type=positive_float,
        default=0.35,
        help=(
            "Maximum RMS difference between sorted Bi7 pair distances, in "
            "angstrom (default: 0.35)"
        ),
    )
    parser.add_argument(
        "--max-distance-error",
        type=positive_float,
        default=0.75,
        help=(
            "Maximum error in any sorted Bi7 pair distance, in angstrom "
            "(default: 0.75)"
        ),
    )
    parser.add_argument(
        "--include-initial",
        action="store_true",
        help="Also scan rows marked init == 1",
    )
    parser.add_argument(
        "--allow-topology-mismatch",
        action="store_true",
        help=(
            "Do not require each Bi7 component to have the same sorted bond "
            "degree sequence as the reference"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="CSV rows read at once (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV receiving one summary row per match",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Maximum number of matching rows printed to the terminal",
    )
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if args.max_print < 0:
        parser.error("--max-print must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if not args.input_csv.is_file():
        print(f"Input CSV not found: {args.input_csv}", file=sys.stderr)
        return 2
    if not args.reference_csv.is_file():
        print(f"Reference CSV not found: {args.reference_csv}", file=sys.stderr)
        return 2
    if (
        args.output is not None
        and args.output.resolve() == args.input_csv.resolve()
    ):
        print("Output CSV must not overwrite the input CSV", file=sys.stderr)
        return 2

    try:
        reference_coordinates = load_reference(
            args.reference_csv, args.reference_name
        )
        matches, counts = scan_rows(
            pd.read_csv(args.input_csv, chunksize=args.chunksize),
            reference_coordinates=reference_coordinates,
            bond_cutoff=args.bond_cutoff,
            distance_rmsd_tolerance=args.distance_rmsd_tolerance,
            max_distance_error=args.max_distance_error,
            include_initial=args.include_initial,
            require_topology=not args.allow_topology_mismatch,
        )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Scanned {counts['candidate_rows']} eligible Bi14 rows from "
        f"{counts['csv_rows']} CSV rows; found {counts['matches']} matches."
    )
    print(
        "Rejected: "
        f"components={counts['component_rejections']}, "
        f"topology={counts['topology_rejections']}, "
        f"shape={counts['shape_rejections']}, "
        f"non-Bi14={counts['non_bi14_rows']}, "
        f"malformed={counts['malformed_rows']}, "
        f"initial-skipped={counts['initial_rows_skipped']}."
    )

    for match in matches[: args.max_print]:
        worst_rmsd = max(match.distance_rmsd_a, match.distance_rmsd_b)
        print(
            f"row={match.row_index} "
            f"clusters={match.cluster_a}|{match.cluster_b} "
            f"worst_pair_RMSD={worst_rmsd:.4f} A "
            f"min_cross={match.minimum_intercluster_distance:.4f} A "
            f"centroid_distance={match.centroid_distance:.4f} A"
        )
    if len(matches) > args.max_print:
        print(f"... {len(matches) - args.max_print} additional matches not printed")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        matches_to_frame(matches).to_csv(args.output, index=False)
        print(f"Wrote match summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
