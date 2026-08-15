#!/usr/bin/env python3
"""Plot energy and force parity for every Bi14 2Bi7 model sample."""

import argparse
import ast
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS_DIR = (
    SCRIPT_DIR / "results" / "charge_embedding" / "bi14-6_samples_logs"
)
PREDICTION_FILENAME = "Bi14_2Bi7_samples_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create combined energy/force parity plots from all available "
            "sample_*/Bi14_2Bi7_samples_predictions.csv files."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=f"Directory containing sample_* folders (default: {DEFAULT_LOGS_DIR}).",
    )
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional comma-separated sample indices; default uses every sample_* file.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help=(
            "Output path without an extension. Each sample is saved as "
            "<stem>_sample_<N>.png/pdf."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300).")
    return parser.parse_args()


def sample_number(path: Path) -> int:
    try:
        return int(path.parent.name.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not determine sample number from {path}.") from exc


def find_prediction_files(logs_dir: Path, samples: str = None) -> List[Path]:
    paths = sorted(
        logs_dir.glob(f"sample_*/{PREDICTION_FILENAME}"), key=sample_number
    )
    if samples is not None:
        selected = {
            int(value.strip()) for value in samples.split(",") if value.strip()
        }
        paths = [path for path in paths if sample_number(path) in selected]
        missing = selected.difference(sample_number(path) for path in paths)
        if missing:
            raise FileNotFoundError(
                f"Missing prediction CSVs for samples: {sorted(missing)}."
            )
    if not paths:
        raise FileNotFoundError(
            f"No sample_*/{PREDICTION_FILENAME} files found under {logs_dir}."
        )
    return paths


def parse_scalar(value) -> float:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one scalar value, got {values.size}: {value!r}")
    return float(values[0])


def parse_array(value) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) array, got shape {array.shape}.")
    return array


def load_predictions(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(path)
    required = {
        "configuration_index",
        "total_energy",
        "reference_force",
        "predicted_energy",
        "predicted_force",
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}.")

    indices = dataframe["configuration_index"].to_numpy(dtype=np.int64)
    reference_energy = np.asarray(
        [parse_scalar(value) for value in dataframe["total_energy"]],
        dtype=np.float64,
    )
    predicted_energy = dataframe["predicted_energy"].to_numpy(dtype=np.float64)
    reference_force = np.concatenate(
        [parse_array(value).reshape(-1) for value in dataframe["reference_force"]]
    )
    predicted_force = np.concatenate(
        [parse_array(value).reshape(-1) for value in dataframe["predicted_force"]]
    )
    arrays = (reference_energy, predicted_energy, reference_force, predicted_force)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"Non-finite prediction or reference values found in {path}.")
    return indices, reference_energy, predicted_energy, reference_force, predicted_force


def validate_common_reference(
    expected_indices: np.ndarray,
    expected_energy: np.ndarray,
    expected_force: np.ndarray,
    indices: np.ndarray,
    energy: np.ndarray,
    force: np.ndarray,
    path: Path,
) -> None:
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"Configuration indices differ in {path}.")
    if not np.allclose(energy, expected_energy, rtol=0.0, atol=1e-10):
        raise ValueError(f"Reference energies differ in {path}.")
    if not np.allclose(force, expected_force, rtol=0.0, atol=1e-10):
        raise ValueError(f"Reference forces differ in {path}.")


def square_limits(reference: np.ndarray, predictions: Sequence[np.ndarray]):
    low = min(float(reference.min()), *(float(values.min()) for values in predictions))
    high = max(float(reference.max()), *(float(values.max()) for values in predictions))
    span = high - low
    padding = 0.05 * span if span > 0 else 1.0
    return low - padding, high + padding


def create_plot(
    prediction_files: Sequence[Path], output_stem: Path, dpi: int
) -> List[Path]:
    sample_results = []
    expected_indices = expected_energy = expected_force = None

    for path in prediction_files:
        indices, reference_energy, predicted_energy, reference_force, predicted_force = (
            load_predictions(path)
        )
        if expected_indices is None:
            expected_indices = indices
            expected_energy = reference_energy
            expected_force = reference_force
        else:
            validate_common_reference(
                expected_indices,
                expected_energy,
                expected_force,
                indices,
                reference_energy,
                reference_force,
                path,
            )
        sample_results.append(
            {
                "sample": sample_number(path),
                "energy": predicted_energy,
                "force": predicted_force,
                "energy_rmse": float(
                    np.sqrt(np.mean((predicted_energy - reference_energy) ** 2))
                ),
                "energy_mae": float(
                    np.mean(np.abs(predicted_energy - reference_energy))
                ),
                "force_rmse": float(
                    np.sqrt(np.mean((predicted_force - reference_force) ** 2))
                ),
                "force_mae": float(
                    np.mean(np.abs(predicted_force - reference_force))
                ),
            }
        )

    energy_center = float(expected_energy.mean())
    centered_reference_energy = expected_energy - energy_center
    centered_predicted_energies = [
        result["energy"] - energy_center for result in sample_results
    ]
    predicted_forces = [result["force"] for result in sample_results]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    # Keep identical limits across samples so the five figures are comparable.
    energy_limits = square_limits(
        centered_reference_energy, centered_predicted_energies
    )
    force_limits = square_limits(expected_force, predicted_forces)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_paths = []

    for result, centered_energy in zip(sample_results, centered_predicted_energies):
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
        axes[0].scatter(
            centered_reference_energy,
            centered_energy,
            s=7,
            alpha=0.48,
            color="#2166ac",
            edgecolors="none",
            rasterized=True,
        )
        axes[1].scatter(
            expected_force,
            result["force"],
            s=1.4,
            alpha=0.12,
            color="#2166ac",
            edgecolors="none",
            rasterized=True,
        )

        for axis, limits in zip(axes, (energy_limits, force_limits)):
            axis.plot(limits, limits, color="black", linewidth=0.8)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(which="both", length=3.5)

        axes[0].set_xlabel(r"DFT $E - \langle E_{\mathrm{DFT}}\rangle$ (eV)")
        axes[0].set_ylabel(r"MACE $E - \langle E_{\mathrm{DFT}}\rangle$ (eV)")
        axes[1].set_xlabel(r"DFT force component (eV $\mathrm{\AA}^{-1}$)")
        axes[1].set_ylabel(r"MACE force component (eV $\mathrm{\AA}^{-1}$)")

        figure.tight_layout(w_pad=2.2)
        sample_stem = output_stem.with_name(
            f"{output_stem.name}_sample_{result['sample']}"
        )
        png_path = sample_stem.with_suffix(".png")
        pdf_path = sample_stem.with_suffix(".pdf")
        figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
        figure.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        png_paths.append(png_path)

    metrics_path = output_stem.with_name(f"{output_stem.name}_metrics.csv")
    pd.DataFrame(
        [
            {
                "sample": result["sample"],
                "number_of_configurations": len(expected_indices),
                "energy_rmse": result["energy_rmse"],
                "energy_mae": result["energy_mae"],
                "force_component_rmse": result["force_rmse"],
                "force_component_mae": result["force_mae"],
            }
            for result in sample_results
        ]
    ).to_csv(metrics_path, index=False)

    print(f"Samples: {[result['sample'] for result in sample_results]}")
    print(f"Configurations per sample: {len(expected_indices)}")
    for path in png_paths:
        print(f"Wrote parity plot: {path}")
    print(f"Wrote metrics: {metrics_path}")
    return png_paths


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")
    logs_dir = args.logs_dir.expanduser().resolve()
    prediction_files = find_prediction_files(logs_dir, args.samples)
    output_stem = (
        args.output_stem.expanduser().resolve()
        if args.output_stem is not None
        else logs_dir / "Bi14_2Bi7_parity"
    )
    create_plot(prediction_files, output_stem, args.dpi)


if __name__ == "__main__":
    main()
