#!/usr/bin/env python3
"""Evaluate one pretrained Bi14 MACE model on the 2Bi7 dataset.

The input CSV's ``forces`` column contains gradients. This script negates those
values in memory before calculating force errors and never modifies the input
CSV. Predictions and metrics are written to new files.
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from ase.data import atomic_numbers
from torch_geometric.data import Data

from evaluation import evaluate


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT.parent / "dataset_Bi" / "dataset" / "Bi14_2Bi7_samples.csv"
)
DEFAULT_LOGS_ROOT = SCRIPT_DIR / "results" / "charge_embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a model from <logs-root>/<prefix>_logs/sample_<model-number>/"
            "<prefix>.model and evaluate it on Bi14_2Bi7_samples.csv."
        )
    )
    parser.add_argument(
        "--model-number",
        "--model_number",
        type=int,
        required=True,
        help="Sample-folder index, for example 0 loads sample_0/<prefix>.model.",
    )
    parser.add_argument(
        "--prefix",
        default="bi14-6_samples",
        help="Model prefix (default: bi14-6_samples).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Input CSV (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=DEFAULT_LOGS_ROOT,
        help=f"Directory containing <prefix>_logs (default: {DEFAULT_LOGS_ROOT}).",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=-6,
        help="Total molecular charge supplied to the model (default: -6).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (default: auto).",
    )
    parser.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating-point dtype used for evaluation (default: float64).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Prediction CSV path. By default it is written beside the model as "
            "Bi14_2Bi7_samples_predictions.csv."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N rows (useful for smoke tests).",
    )
    return parser.parse_args()


def parse_literal(value: Any, field_name: str, row_number: int) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Could not parse {field_name!r} in CSV row {row_number}."
        ) from exc


def parse_energy(value: Any, row_number: int) -> float:
    value = parse_literal(value, "total_energy", row_number)
    if isinstance(value, (list, tuple, np.ndarray)):
        flattened = np.asarray(value, dtype=np.float64).reshape(-1)
        if flattened.size != 1:
            raise ValueError(
                f"Expected one total energy in CSV row {row_number}, got {flattened.size}."
            )
        value = flattened[0]
    energy = float(value)
    if not np.isfinite(energy):
        raise ValueError(f"Non-finite total energy in CSV row {row_number}.")
    return energy


def parse_atoms(value: Any, row_number: int) -> List[int]:
    atoms = parse_literal(value, "atoms", row_number)
    if not isinstance(atoms, (list, tuple)) or not atoms:
        raise ValueError(f"Expected a non-empty atom list in CSV row {row_number}.")

    atomic_numbers_list = []
    for atom in atoms:
        if isinstance(atom, str):
            symbol = atom.strip().capitalize()
            if symbol not in atomic_numbers:
                raise ValueError(
                    f"Unknown element symbol {atom!r} in CSV row {row_number}."
                )
            atomic_numbers_list.append(int(atomic_numbers[symbol]))
        else:
            atomic_numbers_list.append(int(atom))
    return atomic_numbers_list


def parse_matrix(
    value: Any, field_name: str, row_number: int, atom_count: int
) -> np.ndarray:
    value = parse_literal(value, field_name, row_number)
    matrix = np.asarray(value, dtype=np.float64)
    expected_shape = (atom_count, 3)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected {field_name} shape {expected_shape} in CSV row {row_number}, "
            f"got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite values in {field_name} in CSV row {row_number}.")
    return matrix


def load_dataset(
    csv_path: Path, charge: int, torch_dtype: torch.dtype, limit: int = None
) -> Tuple[pd.DataFrame, List[Data], np.ndarray, List[np.ndarray], List[np.ndarray]]:
    dataframe = pd.read_csv(csv_path)
    required_columns = {"atoms", "coordinates", "total_energy", "forces"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}.")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        dataframe = dataframe.iloc[:limit].copy()
    if dataframe.empty:
        raise ValueError("Input CSV contains no rows to evaluate.")

    eval_dataset = []
    reference_energies = []
    reference_gradients = []
    reference_forces = []

    for output_index, (_, row) in enumerate(dataframe.iterrows()):
        row_number = output_index + 2  # Account for the CSV header and zero indexing.
        z = parse_atoms(row["atoms"], row_number)
        coordinates = parse_matrix(
            row["coordinates"], "coordinates", row_number, len(z)
        )
        gradient = parse_matrix(row["forces"], "forces", row_number, len(z))
        force = -gradient
        energy = parse_energy(row["total_energy"], row_number)

        eval_dataset.append(
            Data(
                pos=torch.as_tensor(coordinates, dtype=torch_dtype),
                z=torch.as_tensor(z, dtype=torch.long),
                y=torch.as_tensor([energy], dtype=torch_dtype),
                forces=torch.as_tensor(force, dtype=torch_dtype),
                charge=torch.tensor(charge, dtype=torch_dtype),
            )
        )
        reference_energies.append(energy)
        reference_gradients.append(gradient)
        reference_forces.append(force)

    return (
        dataframe,
        eval_dataset,
        np.asarray(reference_energies, dtype=np.float64),
        reference_gradients,
        reference_forces,
    )


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested_device


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    try:
        try:
            return torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Compatibility with PyTorch versions that predate weights_only.
            return torch.load(model_path, map_location=device)
    except RuntimeError as exc:
        message = str(exc)
        if "NVIDIA driver" in message or "libcuda" in message:
            raise RuntimeError(
                "This saved model contains CUDA/cuequivariance modules and cannot be "
                "loaded on the current CPU-only node. Run the script on a GPU node."
            ) from exc
        raise


def arrays_to_json(arrays: Sequence[np.ndarray]) -> List[str]:
    return [json.dumps(np.asarray(array, dtype=np.float64).tolist()) for array in arrays]


def write_outputs(
    dataframe: pd.DataFrame,
    output_path: Path,
    model_path: Path,
    dataset_path: Path,
    model_number: int,
    charge: int,
    reference_energies: np.ndarray,
    reference_gradients: Sequence[np.ndarray],
    reference_forces: Sequence[np.ndarray],
    predicted_energies: np.ndarray,
    predicted_forces: Sequence[np.ndarray],
) -> Path:
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Refusing to overwrite the input dataset CSV.")

    predicted_energies = np.asarray(predicted_energies, dtype=np.float64).reshape(-1)
    if predicted_energies.shape != reference_energies.shape:
        raise ValueError(
            "Predicted and reference energy arrays have different shapes: "
            f"{predicted_energies.shape} vs {reference_energies.shape}."
        )
    if len(predicted_forces) != len(reference_forces):
        raise ValueError("Predicted and reference force collections have different lengths.")

    force_errors = []
    force_rmse_per_structure = []
    force_mae_per_structure = []
    for index, (prediction, reference) in enumerate(
        zip(predicted_forces, reference_forces)
    ):
        prediction = np.asarray(prediction, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if prediction.shape != reference.shape:
            raise ValueError(
                f"Force shape mismatch for configuration {index}: "
                f"{prediction.shape} vs {reference.shape}."
            )
        error = prediction - reference
        force_errors.append(error)
        force_rmse_per_structure.append(float(np.sqrt(np.mean(error**2))))
        force_mae_per_structure.append(float(np.mean(np.abs(error))))

    energy_errors = predicted_energies - reference_energies
    all_force_errors = np.concatenate(
        [np.asarray(error).reshape(-1) for error in force_errors]
    )

    output_dataframe = dataframe.copy()
    output_dataframe.rename(columns={"forces": "reference_gradient"}, inplace=True)
    output_dataframe.insert(0, "configuration_index", np.arange(len(dataframe)))
    output_dataframe["reference_gradient"] = arrays_to_json(reference_gradients)
    output_dataframe["reference_force"] = arrays_to_json(reference_forces)
    output_dataframe["predicted_energy"] = predicted_energies
    output_dataframe["energy_error"] = energy_errors
    output_dataframe["predicted_force"] = arrays_to_json(predicted_forces)
    output_dataframe["force_error"] = arrays_to_json(force_errors)
    output_dataframe["force_rmse"] = force_rmse_per_structure
    output_dataframe["force_mae"] = force_mae_per_structure

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, index=False)

    atom_counts = [np.asarray(force).shape[0] for force in reference_forces]
    metrics = {
        "model_number": model_number,
        "model_path": str(model_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "prediction_path": str(output_path.resolve()),
        "number_of_configurations": len(dataframe),
        "atom_counts": sorted(set(atom_counts)),
        "charge": charge,
        "input_forces_column_contains": "gradient",
        "gradient_to_force_conversion": "force = -gradient (performed in memory)",
        "energy_rmse": float(np.sqrt(np.mean(energy_errors**2))),
        "energy_mae": float(np.mean(np.abs(energy_errors))),
        "force_component_rmse": float(np.sqrt(np.mean(all_force_errors**2))),
        "force_component_mae": float(np.mean(np.abs(all_force_errors))),
        "force_component_max_abs_error": float(np.max(np.abs(all_force_errors))),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics_path


def main() -> None:
    args = parse_args()
    if args.model_number < 0:
        raise ValueError("--model-number must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    dataset_path = args.dataset.expanduser().resolve()
    logs_root = args.logs_root.expanduser().resolve()
    model_dir = logs_root / f"{args.prefix}_logs" / f"sample_{args.model_number}"
    model_path = model_dir / f"{args.prefix}.model"
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_dir / "Bi14_2Bi7_samples_predictions.csv"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if output_path.resolve() == dataset_path:
        raise ValueError("Refusing to overwrite the input dataset CSV.")

    torch_dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    device = resolve_device(args.device)
    print(f"Loading dataset: {dataset_path}")
    dataframe, eval_dataset, true_energy, gradients, true_forces = load_dataset(
        csv_path=dataset_path,
        charge=args.charge,
        torch_dtype=torch_dtype,
        limit=args.limit,
    )
    print(
        f"Converted gradients to forces in memory for {len(eval_dataset)} configurations; "
        "the input CSV was not modified."
    )
    print(f"Loading model: {model_path}")
    print(f"Evaluation device: {device}")
    model = load_model(model_path, device)
    predicted_energy, predicted_forces, _, _ = evaluate(
        model=model,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        default_dtype=args.default_dtype,
        device=device,
        compute_stress=False,
        return_contributions=False,
    )

    metrics_path = write_outputs(
        dataframe=dataframe,
        output_path=output_path,
        model_path=model_path,
        dataset_path=dataset_path,
        model_number=args.model_number,
        charge=args.charge,
        reference_energies=true_energy,
        reference_gradients=gradients,
        reference_forces=true_forces,
        predicted_energies=predicted_energy,
        predicted_forces=predicted_forces,
    )
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
