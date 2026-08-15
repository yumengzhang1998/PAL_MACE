import os
import argparse
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import json
# ---------- NEW: dynamic std extractor ----------
# Map friendly property names to summary filenames produced by your VAL script
_PROP_TO_SUMMARY = {
    "energy_std":               "energy_std_summary.json",
    "force_std_rms":            "force_std_rms_summary.json",
    "force_std_max_atomnorm":   "force_std_max_atomnorm_summary.json",
    # (available too if you ever want them)
    "force_std_max_coord":      "force_std_max_coord_summary.json",
    "force_std_p95_atomnorm":   "force_std_p95_atomnorm_summary.json",
}

def add_time_stamp(s):
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{s}_{timestamp}"

def _normalize_quantile(q):
    """Accept 'q75', '75', 75 -> 'q75'."""
    if isinstance(q, (int, float)):
        return f"q{int(q)}"
    s = str(q).strip().lower()
    return s if s.startswith("q") else f"q{int(float(s))}"

def extract_std(std_path: str, property_list, quantile: str | int):
    """
    Read quantiles from summary JSONs produced by make_val_std_distribution.py.
    std_path: directory like 'usr/pretrain/results/charge_embedding/{prefix}/{prefix}_VAL_uncertainty/'
    property_list: e.g. ['energy_std', 'force_std_max_atomnorm', 'force_std_rms']
    quantile: e.g. 'q75' or 75
    Returns: dict {property_name: float}
    Raises ValueError with a helpful message if anything is missing.
    """
    std_dir = Path(std_path)
    if not std_dir.exists():
        raise ValueError(f"std_path does not exist: {std_dir}")

    qkey = _normalize_quantile(quantile)
    out = {}

    for prop in property_list:
        if prop not in _PROP_TO_SUMMARY:
            raise ValueError(
                f"Unknown property '{prop}'. Supported: {list(_PROP_TO_SUMMARY.keys())}"
            )
        fname = _PROP_TO_SUMMARY[prop]
        fpath = std_dir / fname
        if not fpath.exists():
            # Helpful hint if user accidentally passed wrong folder
            candidates = list(std_dir.glob("*summary.json"))
            hint = f"Found: {[p.name for p in candidates]}" if candidates else "No summary JSONs found."
            raise ValueError(f"Missing summary file for '{prop}': {fpath}\n{hint}")

        with open(fpath, "r") as f:
            js = json.load(f)

        if qkey not in js or js[qkey] is None:
            raise ValueError(
                f"Quantile '{qkey}' not found in {fpath}. "
                f"Available keys: {sorted([k for k in js.keys() if k.startswith('q')])}"
            )
        out[prop] = float(js[qkey])

    return out
# Load extra data from CSV
def load_coord_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    coord_dict = {}
    for _, row in df.iterrows():
        name = row["Name"].lower()
        coord = ast.literal_eval(row["coord"])
        coord_dict[name] = coord
    return coord_dict


def _parse_csv_array(value, column, row_number, train_path):
    """Parse one array-like CSV field and attach row/path context to failures."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Could not parse '{column}' at data row {row_number} in {train_path}: {exc}"
            ) from exc
    return value


def load_pretrain_reference(prefix, num_models):
    """Derive global AL references from the first ``num_models`` bootstrap members.

    For example, ``num_models=2`` combines ``sample_0/train.csv`` and
    ``sample_1/train.csv``. The same value controls the AL predictor/trainer
    ensemble size in the generated configuration.
    """
    if num_models < 1:
        raise ValueError(f"num_models must be positive, got {num_models}.")

    selected_model_indices = list(range(num_models))
    train_paths = [
        Path("usr/pretrain/samples")
        / prefix
        / f"sample_{model_index}"
        / "train.csv"
        for model_index in selected_model_indices
    ]
    num_atom = None
    max_distances = []
    energies = []

    for train_path in train_paths:
        if not train_path.is_file():
            raise ValueError(f"Pretraining reference CSV does not exist: {train_path}")

        df = pd.read_csv(train_path)
        required_columns = {"coordinates", "total_energy"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Pretraining reference CSV {train_path} is missing required columns: "
                f"{missing_columns}"
            )
        if df.empty:
            raise ValueError(f"Pretraining reference CSV is empty: {train_path}")

        for row_number, row in df.iterrows():
            coordinates = _parse_csv_array(
                row["coordinates"], "coordinates", row_number, train_path
            )
            try:
                coordinates = np.asarray(coordinates, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Coordinates at data row {row_number} in {train_path} are not numeric."
                ) from exc

            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(
                    f"Coordinates at data row {row_number} in {train_path} must have "
                    f"shape (N, 3), got {coordinates.shape}."
                )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    f"Coordinates at data row {row_number} in {train_path} contain "
                    "non-finite values."
                )

            row_num_atom = int(coordinates.shape[0])
            if num_atom is None:
                num_atom = row_num_atom
            elif row_num_atom != num_atom:
                raise ValueError(
                    f"Inconsistent atom counts in {train_path}: expected {num_atom}, "
                    f"got {row_num_atom} at data row {row_number}."
                )

            if "atoms" in df.columns:
                atoms = _parse_csv_array(
                    row["atoms"], "atoms", row_number, train_path
                )
                if len(atoms) != row_num_atom:
                    raise ValueError(
                        f"Atom/coordinate count mismatch at data row {row_number} in "
                        f"{train_path}: {len(atoms)} atoms and "
                        f"{row_num_atom} coordinates."
                    )

            if row_num_atom < 2:
                max_distances.append(0.0)
            else:
                displacements = (
                    coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
                )
                max_distances.append(
                    float(
                        np.sqrt(
                            np.sum(displacements * displacements, axis=-1)
                        ).max()
                    )
                )

            energy = _parse_csv_array(
                row["total_energy"], "total_energy", row_number, train_path
            )
            try:
                energy_values = np.asarray(energy, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Energy at data row {row_number} in {train_path} is not numeric."
                ) from exc
            if energy_values.size != 1 or not np.isfinite(energy_values[0]):
                raise ValueError(
                    f"Energy at data row {row_number} in {train_path} must contain "
                    f"one finite value, got {energy!r}."
                )
            energies.append(float(energy_values[0]))

    energy_threshold = float(np.median(energies))
    q25, q75 = (float(value) for value in np.percentile(energies, [25, 75]))
    iqr = q75 - q25

    reference = {
        "num_atom": int(num_atom),
        "max_dist": float(np.max(max_distances)),
        "energy_threshold": energy_threshold,
        "hard_bound": 3.0 * iqr,
        "soft_bound": 1.5 * iqr,
        "reference_sources": [str(path) for path in train_paths],
        "reference_model_indices": selected_model_indices,
    }
    print(
        f"Using pretraining models {selected_model_indices} from {train_paths}: "
        f"num_atom={reference['num_atom']}, "
        f"max_dist={reference['max_dist']}, "
        f"energy_threshold={reference['energy_threshold']}, "
        f"q25={q25}, q75={q75}"
    )
    return reference

# Prefix base settings (excluding coord)
# prefix_settings = {
#     "bi4-2": {
#         "energy_threshold": -23365.0,
#         "energy_std_threshold":  0.0098,
#         "force_atom_max_std": 0.0073,
#         "force_rms_std": 0.0034,
#         "bound": 10,
#         "num_atom": 4,
#         "max_dist": 4.161544
#     },
#     "bi4-6": {
#         "energy_threshold":  -23374.0,
#         "energy_std_threshold": 0.026,
#         "force_atom_max_std": 0.0090,
#         "force_rms_std": 0.004,
#         "bound": 10,
#         "num_atom": 4,
#         "max_dist": 7.074161
#     },
#     "bi7-3": {
#         "energy_threshold":  -40889.9,
#         "energy_std_threshold": 0.123,
#         "force_atom_max_std": 0.0203,
#         "force_rms_std": 0.0083,
#         "bound": 10,
#         "num_atom": 7,
#         "max_dist": 4.748247
#     },
#     "bi11-3": {
#         "energy_threshold": -64250.5,
#         "energy_std_threshold":  0.119,
#         "force_atom_max_std": 0.0293,
#         "force_rms_std": 0.011,
#         "bound": 10,
#         "num_atom": 11,
#         "max_dist": 6.909196
#     },
#     "bi11-3_samples": {
#         "energy_threshold": -64250.5,
#         "energy_std_threshold": 0.2844,
#         "force_atom_max_std":  0.3380,
#         "force_rms_std": 0.1181,
#         "bound": 10,
#         "num_atom": 11,
#         "max_dist": 9.055371
#     }
# }
def load_prefix_settings(prefix, use_pretrain_reference=False, num_models=2):
    base_dir = Path("usr/pretrain/results/charge_embedding")
    prefixes = ["bi4-2", "bi4-6", "bi7-3", "bi11-3", "bi11-3_samples","bi2-2", "bi11-3_samples_bi2", "bi14-6_samples"]
    if not use_pretrain_reference and prefix not in prefixes:
        raise ValueError(f"Prefix '{prefix}' is not recognized. Available: {prefixes}")
    settings = {}

    std_path = base_dir / f"{prefix}_logs" / f"{prefix}_VAL_uncertainty"
    try:
        stds = extract_std(
            std_path=str(std_path),
            property_list=["energy_std", "force_std_max_atomnorm", "force_std_rms"],
            quantile="q75"
        )
    except ValueError as e:
        raise ValueError(f"Error extracting stds for prefix '{prefix}': {e}")

    if use_pretrain_reference:
        reference = load_pretrain_reference(prefix, num_models)
    else:
        reference_file = "optimized.csv"
        if not os.path.exists(reference_file):
            raise ValueError(f"Reference file does not exist: {reference_file}")

        df = pd.read_csv(reference_file)
        required_columns = {
            "Name", "MaxDistance", "NumAtoms", "energy_threshold", "q25", "q75"
        }
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Reference file {reference_file} is missing required columns: "
                f"{missing_columns}"
            )

        matching_rows = df[df["Name"].astype(str).str.lower() == prefix]
        if matching_rows.empty:
            raise ValueError(
                f"Could not find prefix '{prefix}' in reference file {reference_file}."
            )
        row = matching_rows.iloc[0]
        try:
            energy_threshold = float(
                np.asarray(ast.literal_eval(row["energy_threshold"])).reshape(-1)[0]
            )
            q25 = float(np.asarray(ast.literal_eval(row["q25"])).reshape(-1)[0])
            q75 = float(np.asarray(ast.literal_eval(row["q75"])).reshape(-1)[0])
            num_atom = int(row["NumAtoms"])
            max_dist = float(row["MaxDistance"])
        except (TypeError, ValueError, SyntaxError, IndexError) as exc:
            raise ValueError(
                f"Invalid reference values for prefix '{prefix}' in {reference_file}: {exc}"
            ) from exc

        iqr = q75 - q25
        reference = {
            "energy_threshold": energy_threshold,
            "hard_bound": 3.0 * iqr,
            "soft_bound": 1.5 * iqr,
            "num_atom": num_atom,
            "max_dist": max_dist,
            "reference_sources": [reference_file],
            "reference_model_indices": None,
        }
        print(
            f"Using optimized reference {reference_file} for prefix '{prefix}': "
            f"num_atom={num_atom}, max_dist={max_dist}, "
            f"energy_threshold={energy_threshold}, q25={q25}, q75={q75}"
        )

    settings[prefix] = {
        "energy_threshold": reference["energy_threshold"],
        "energy_std_threshold": stds["energy_std"],
        "force_atom_max_std": stds["force_std_max_atomnorm"],
        "force_rms_std": stds["force_std_rms"],
        "bound": 10,
        "hard_bound": reference["hard_bound"],
        "soft_bound": reference["soft_bound"],
        "num_atom": reference["num_atom"],
        "max_dist": reference["max_dist"],
        "reference_sources": reference["reference_sources"],
        "reference_model_indices": reference["reference_model_indices"],
    }

    return settings

def generate_config_yaml(
    prefix,
    full_dataset,
    coord_dict,
    num_traj_per_gene,
    load_model,
    load_dataset,
    starting_pool_update,
    use_pretrain_reference=False,
    num_models=2,
    save_gene_traj=False,
    gene_temperature_low=None,
    gene_temperature_high=None,
    max_steps_per_traj=100000,
    num_gen_process=2,
):
    if not full_dataset in [True, False]:
        raise ValueError(f"full_dataset must be True or False, got '{full_dataset}'")
    if num_models < 1:
        raise ValueError(f"num_models must be positive, got {num_models}.")
    if max_steps_per_traj < 1:
        raise ValueError(
            f"max_steps_per_traj must be positive, got {max_steps_per_traj}."
        )
    if num_gen_process < 1:
        raise ValueError(
            f"num_gen_process must be positive, got {num_gen_process}."
        )
    if (gene_temperature_low is None) != (gene_temperature_high is None):
        raise ValueError(
            "gene_temperature_low and gene_temperature_high must either both "
            "be set or both be omitted."
        )
    if gene_temperature_low is not None:
        gene_temperature_low = float(gene_temperature_low)
        gene_temperature_high = float(gene_temperature_high)
        if not (
            np.isfinite(gene_temperature_low)
            and np.isfinite(gene_temperature_high)
        ):
            raise ValueError("Generator temperature bounds must be finite.")
        if gene_temperature_low <= 0 or gene_temperature_high <= 0:
            raise ValueError("Generator temperature bounds must be positive.")
        if gene_temperature_low > gene_temperature_high:
            raise ValueError(
                "gene_temperature_low must not exceed gene_temperature_high."
            )
    if not use_pretrain_reference and (coord_dict is None or prefix not in coord_dict):
        raise ValueError(f"Coordinates not found in CSV for prefix '{prefix}'.")
    if not full_dataset:
        prefix_settings = load_prefix_settings(
            prefix,
            use_pretrain_reference=use_pretrain_reference,
            num_models=num_models,
        )
        settings = prefix_settings[prefix]
    if full_dataset:
        raise ValueError("full_dataset=True is not supported in this script. Please provide specific settings for the prefix.")
    coord = None if use_pretrain_reference else coord_dict[prefix]
    coord_yaml = "null" if coord is None else repr(coord)
    reference_model_indices_yaml = (
        "null"
        if settings["reference_model_indices"] is None
        else json.dumps(settings["reference_model_indices"])
    )
    reference_sources_yaml = json.dumps(settings["reference_sources"])
    gene_temperature_low_yaml = (
        "null" if gene_temperature_low is None else repr(gene_temperature_low)
    )
    gene_temperature_high_yaml = (
        "null" if gene_temperature_high is None else repr(gene_temperature_high)
    )
    print(f"Using settings for prefix '{prefix}': {settings}")
    if coord is not None:
        print(f"Using coordinates for prefix '{prefix}': {coord}")

    content = f'''# MACE
args_dict: {{
    "name": "MACE_on_{prefix}",
    "num_workers": 16,
    "train_file": "train.xyz",
    "valid_file": "test.xyz",
    "test_file": "test.xyz",
    "results_dir": "results",
    "E0s": "average",
    "statistics_file": None,
    "model": "MACE_with_charge",
    "num_interactions": 2,
    "num_channels": 128,
    "max_L": 1,
    "r_max": 9.0,
    "patience": 20,
    "correlation": 3,
    "batch_size": 32,
    "valid_batch_size": 32,
    "max_num_epochs": 200,
    "swa": False,
    "ema": True,
    "ema_decay": 0.99,
    "amsgrad": True,
    "error_table": "TotalMAE",
    "device": "cpu",
    "seed": 123
}}

# time
time_stamp: {add_time_stamp(prefix)}

# MD settings
num_traj_per_gene: {num_traj_per_gene}
max_steps_per_traj: {max_steps_per_traj}
save_gene_traj: {str(save_gene_traj)}
gene_temperature_low: {gene_temperature_low_yaml}
gene_temperature_high: {gene_temperature_high_yaml}

# Retraining settings
load_model: {str(load_model)}
load_dataset: {str(load_dataset)}
starting_pool_update: {str(starting_pool_update)}
pool_csv: "starting_point_pool.csv"

# active learning
patience_threshold: 10

num_pred_process: {num_models}
num_orcl_process: 50
num_gen_process: {num_gen_process}
retrain_size: 50

full_dataset: {full_dataset}
use_pretrain_reference: {str(use_pretrain_reference)}
reference_sources: {reference_sources_yaml}
reference_model_indices: {reference_model_indices_yaml}

prefix: {prefix}
energy_threshold: {settings['energy_threshold']}
energy_std_threshold: {settings['energy_std_threshold']}
force_atom_max_std: {settings['force_atom_max_std']}
force_rms_std: {settings['force_rms_std']}
bound: {settings['bound']}
hard_bound: {settings['hard_bound']}
soft_bound: {settings['soft_bound']}
num_atom: {settings['num_atom']}
coord: {coord_yaml}
max_dist: {settings['max_dist']}
source: {{"real": 0, "synthesis_bi4": 1, "synthesis_bi2": 2}}
metadata:
  - {{ name: coords,          type: array,  shape: [{settings['num_atom']}, 3], dtype: float64 }}
  - {{ name: atomic_numbers,  type: tensor,   shape: [{settings['num_atom']}],    dtype: torch.int64}}  # or tensor with int dtype
  - {{ name: energy,          type: scalar_nullable,        dtype: float  }}  # was None OK too
  - {{ name: forces,          type: array,  shape: [{settings['num_atom']}, 3], dtype: float64 }}
  - {{ name: charge,          type: charge,                 dtype: torch.long}}                                 # 1 scalar int
  - {{ name: pred_forces,     type: array,  shape: [{settings['num_atom']}, 3], dtype: float64 }} # ← this one commonly mis-set
  - {{ name: pred_energy,     type: scalar_nullable,        dtype: float  }}
  - {{ name: patience,        type: list,   shape: [2]    , dtype: int }}  # must be BEFORE velocities
  - {{ name: velocities,      type: array,  shape: [{settings['num_atom']}, 3], dtype: float64 }}
'''

    with open("config.yaml", "w") as f:
        f.write(content)
    print(f"✅ config.yaml generated for prefix '{prefix}'")
def str2bool(v):
    return v.lower() in ("true", "1", "yes")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Prefix to use (e.g., bi4-2)")
    parser.add_argument("--full_dataset", required=True, help="Use full dataset (True/False)")
    parser.add_argument("--num_traj_per_gene", type=int, default=1, help="Number of trajectories per generation (default: 1)")
    parser.add_argument(
        "--max_steps_per_traj",
        type=int,
        default=100000,
        help=(
            "Maximum MD steps in one generator trajectory episode before "
            "restart (default: 100000)"
        ),
    )
    parser.add_argument(
        "--num_gen_process",
        type=int,
        default=2,
        help="Number of generator MPI processes (default: 2)",
    )
    parser.add_argument("--load_model", type=str2bool, default=False, help="Whether to load existing model (default: False)")
    parser.add_argument("--load_dataset", type=str2bool, default=False, help="Whether to load existing dataset (default: False)")
    parser.add_argument("--starting_pool_update", type=str2bool, default=False, help="Whether to update starting point pool in Generator process (default: False)")
    parser.add_argument(
        "--save_gene_traj",
        type=str2bool,
        default=False,
        help=(
            "Save compact generator coordinates every 100 successful MD steps "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--gene_temperature_low",
        type=float,
        default=None,
        help=(
            "Lower bound for uniformly sampled generator temperatures in K. "
            "Set together with --gene_temperature_high."
        ),
    )
    parser.add_argument(
        "--gene_temperature_high",
        type=float,
        default=None,
        help=(
            "Upper bound for uniformly sampled generator temperatures in K. "
            "Set together with --gene_temperature_low."
        ),
    )
    parser.add_argument(
        "--use_pretrain_reference",
        type=str2bool,
        default=False,
        help=(
            "Derive num_atom, max_dist, and energy_threshold from a pretraining "
            "sample train.csv instead of optimized.csv (default: False)"
        ),
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=2,
        help=(
            "Number of AL ensemble models. With --use_pretrain_reference True, "
            "N selects and combines sample_0 through sample_(N-1) (default: 2)"
        ),
    )

    args = parser.parse_args()
    full_dataset_bool = args.full_dataset == "True"
    print(
        args.load_model,
        args.load_dataset,
        args.starting_pool_update,
        args.save_gene_traj,
        args.gene_temperature_low,
        args.gene_temperature_high,
        args.use_pretrain_reference,
        args.num_models,
    )

    coord_data = (
        None
        if args.use_pretrain_reference
        else load_coord_from_csv("optimized.csv")
    )
    generate_config_yaml(
        args.prefix,
        full_dataset_bool,
        coord_data,
        args.num_traj_per_gene,
        args.load_model,
        args.load_dataset,
        args.starting_pool_update,
        save_gene_traj=args.save_gene_traj,
        gene_temperature_low=args.gene_temperature_low,
        gene_temperature_high=args.gene_temperature_high,
        max_steps_per_traj=args.max_steps_per_traj,
        num_gen_process=args.num_gen_process,
        use_pretrain_reference=args.use_pretrain_reference,
        num_models=args.num_models,
    )
