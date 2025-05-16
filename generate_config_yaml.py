
from ast import arg
import yaml
import os
import argparse

prefix_settings = {
    "bi4-2": {
        "energy_threshold": -23365.0,
        "std_threshold": 0.007,
        "bound": 10,
        "num_atom": 4
    },
    "bi4-6": {
        "energy_threshold":  -23374.0,
        "std_threshold": 0.020,
        "bound": 10,
        "num_atom": 4
    },
    "bi7-3": {
        "energy_threshold":  -40889.9,
        "std_threshold": 0.001,
        "bound": 10,
        "num_atom": 7
    },
    "bi11-3": {
        "energy_threshold": -64250.5,
        "std_threshold": 0.03,
        "bound": 10,
        "num_atom": 11
    },
    "bi11-3_samples": {
        "energy_threshold": -64250.5,
        "std_threshold": 0.06,
        "bound": 10,
        "num_atom": 11
    }
}

def generate_config_yaml(prefix, full_dataset):
    if prefix not in prefix_settings:
        raise ValueError(f"Prefix '{prefix}' is not recognized. Available: {list(prefix_settings.keys())}")

    settings = prefix_settings[prefix]

    content = f'''# MACE
args_dict: {{
    "name": "MACE_on_{prefix}",
    "num_workers": 16,
    "train_file": "train.xyz",
    "valid_file": "test.xyz",
    "test_file": "test.xyz",
    "results_dir": "results",
    "E0s": "average",
    # "E0s": {{83: -5802.83}},
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
    "swa": False,  # Flags set to True if present
    # "start_swa": 60,
    "ema": True,
    "ema_decay": 0.99,
    "amsgrad": True,
    "error_table": "TotalMAE",
    "device": "cpu",
    "seed": 123
}}


# active learning
patience_threshold: 10

num_pred_process: 2
num_orcl_process: 50
num_gen_process: 4
retrain_size: 50

full_dataset: {full_dataset}

prefix: {prefix}
energy_threshold: {settings['energy_threshold']}
std_threshold: {settings['std_threshold']}
bound: {settings['bound']}
num_atom: {settings['num_atom']}

# data metadata
metadata:
  - type: array # coordinates
    shape: [{settings['num_atom']}, 3]
    dtype: float
  - type: tensor # atom_number
    shape: [{settings['num_atom']}]
    dtype: torch.int64
  - type: scalar_nullable # true_energy
    dtype: int
  - type: array # true_forces
    shape: [{settings['num_atom']}, 3]
    dtype: float
  - type: charge # charge
    dtype: torch.long
  - type: array # pred_forces
    shape: [{settings['num_atom']}, 3]
    dtype: float
  - type: scalar_nullable # pred_energy
    dtype: int
  - type: scalar # patience
    dtype: int
  - type: array #velocities
    shape: [{settings['num_atom']}, 3]
    dtype: float
'''

    with open("config.yaml", "w") as f:
        f.write(content)
    print(f"✅ config.yaml generated for prefix '{prefix}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Prefix to use (e.g., bi4-2)")
    parser.add_argument("--full_dataset", required=True, help="Use full dataset")
    args = parser.parse_args()
    if args.full_dataset == "True":
        args.full_dataset = True
    else: 
        args.full_dataset = False
    generate_config_yaml(args.prefix, args.full_dataset)