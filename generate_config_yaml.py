import os
import argparse
import ast
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
    std_path: directory like 'usr/initial_pyg/results/charge_embedding/{prefix}/{prefix}_VAL_uncertainty/'
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
def load_prefix_settings(prefix):
    base_dir = Path("usr/initial_pyg/results/charge_embedding")
    prefixes = ["bi4-2", "bi4-6", "bi7-3", "bi11-3", "bi11-3_samples","bi2-2", "bi11-3_samples_bi2"]
    if prefix not in prefixes:
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

    # Load max_dist from a precomputed JSON file
    max_dist_file ='optimized.csv'
    if not os.path.exists(max_dist_file):
        raise ValueError(f"max_dist_file does not exist: {max_dist_file}")
    df = pd.read_csv(max_dist_file)
    max_dist_data = df[df["Name"].str.lower() == prefix]['MaxDistance']
    max_dist_data = max_dist_data.iloc[0] if not max_dist_data.empty else None
    max_dist = float(max_dist_data) if max_dist_data is not None else None
    if max_dist is None:    
        raise ValueError(f"Could not find max_dist for prefix '{prefix}' in {max_dist_file}")

    num_atom = df[df["Name"].str.lower() == prefix]['NumAtoms']
    num_atom = int(num_atom.iloc[0]) if not num_atom.empty else None
    if max_dist_data is None or num_atom is None:
        raise ValueError(f"Could not find max_dist or num_atom for prefix '{prefix}' in {max_dist_file}")
    energy_threshold = df[df["Name"].str.lower() == prefix]['energy_threshold']
    q25 = df[df["Name"].str.lower() == prefix]['q25']
    q75 = df[df["Name"].str.lower() == prefix]['q75']
    # convert string to list of floats
    energy_threshold = energy_threshold.iloc[0] if not energy_threshold.empty else None
    q25 = q25.iloc[0] if not q25.empty else None
    q75 = q75.iloc[0] if not q75.empty else None
    energy_threshold = ast.literal_eval(energy_threshold)[0]
    q25 = ast.literal_eval(q25)[0]
    q75 = ast.literal_eval(q75)[0]

    if energy_threshold is None:
        raise ValueError(f"Could not find energy_threshold for prefix '{prefix}' in {max_dist_file}")
    print(f"For prefix '{prefix}': energy_threshold={energy_threshold}, q25={q25}, q75={q75}")
    IQR = q75 - q25
    energy_bound_soft = 1.5 * IQR
    energy_bound_hard = 3.0 * IQR
    print(f"For prefix '{prefix}': energy_bound_soft={energy_bound_soft}, energy_bound_hard={energy_bound_hard}")


    # Set energy_threshold based on known values
    # energy_thresholds = {
    #     "bi4-2": -23365.0,
    #     "bi4-6": -23374.0,
    #     "bi7-3": -40889.9,
    #     "bi11-3": -64250.5,
    #     "bi11-3_samples": -64250.5
    # }
    # energy_threshold = energy_thresholds.get(prefix)
    if energy_threshold is None:
        raise ValueError(f"No predefined energy_threshold for prefix '{prefix}'")

    settings[prefix] = {
        "energy_threshold": energy_threshold,
        "energy_std_threshold": stds["energy_std"],
        "force_atom_max_std": stds["force_std_max_atomnorm"],
        "force_rms_std": stds["force_std_rms"],
        "bound": 10,
        "hard_bound": energy_bound_hard,
        "soft_bound": energy_bound_soft,
        "num_atom": num_atom,
        "max_dist": max_dist
    }

    return settings

def generate_config_yaml(prefix, full_dataset, coord_dict, num_traj_per_gene, load_model, load_dataset, starting_pool_update):
    if prefix not in coord_dict:
        raise ValueError(f"Coordinates not found in CSV for prefix '{prefix}'.")
    if not full_dataset in [True, False]:
        raise ValueError(f"full_dataset must be True or False, got '{full_dataset}'")
    if not full_dataset:
        prefix_settings = load_prefix_settings(prefix)
        settings = prefix_settings[prefix]
    if full_dataset:
        raise ValueError("full_dataset=True is not supported in this script. Please provide specific settings for the prefix.")
    coord = coord_dict[prefix]
    print(f"Using settings for prefix '{prefix}': {settings}")
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

# Retraining settings
load_model: {str(load_model)}
load_dataset: {str(load_dataset)}
starting_pool_update: {str(starting_pool_update)}
pool_csv: "starting_point_pool.csv"

# active learning
patience_threshold: 10

num_pred_process: 2
num_orcl_process: 50
num_gen_process: 2
retrain_size: 50

full_dataset: {full_dataset}

prefix: {prefix}
energy_threshold: {settings['energy_threshold']}
energy_std_threshold: {settings['energy_std_threshold']}
force_atom_max_std: {settings['force_atom_max_std']}
force_rms_std: {settings['force_rms_std']}
bound: {settings['bound']}
hard_bound: {settings['hard_bound']}
soft_bound: {settings['soft_bound']}
num_atom: {settings['num_atom']}
coord: {coord}
max_dist: {settings['max_dist']}

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
    parser.add_argument("--load_model", type=str2bool, default=False, help="Whether to load existing model (default: False)")
    parser.add_argument("--load_dataset", type=str2bool, default=False, help="Whether to load existing dataset (default: False)")
    parser.add_argument("--starting_pool_update", type=str2bool, default=False, help="Whether to update starting point pool in Generator process (default: False)")

    args = parser.parse_args()
    full_dataset_bool = args.full_dataset == "True"
    print(args.load_model, args.load_dataset, args.starting_pool_update)


    coord_data = load_coord_from_csv("optimized.csv")
    generate_config_yaml(args.prefix, full_dataset_bool, coord_data, args.num_traj_per_gene, args.load_model, args.load_dataset, args.starting_pool_update)
