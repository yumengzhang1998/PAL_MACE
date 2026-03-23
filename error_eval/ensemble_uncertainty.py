import os
import torch
import numpy as np
import pandas as pd
import sys

sys.path.append("../usr/pretrain")

from evaluation import evaluate
from data import big_list


device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# Load models
# ==============================
def load_models(model_dir, n_models, prefix):

    models = []

    for i in range(n_models):

        path = f"{model_dir}/sample_{i}/{prefix}.model"

        print(f"Loading model: {path}")

        model = torch.load(path, map_location=device)

        model.eval()
        model.to(device)

        models.append(model)

    return models


# ==============================
# Load dataset
# ==============================
def load_dataset(csv_path, num_atom, charge):

    dataset = big_list(
        raw_data_path=csv_path,
        num_atom=num_atom,
        charge=charge,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    )

    return dataset.data_list


# ==============================
# Ensemble prediction
# ==============================
def ensemble_predict(models, dataset):

    energies = []
    forces = []

    for i, model in enumerate(models):

        print(f"Running prediction with model {i}")

        e, f, _, _ = evaluate(
            model=model,
            eval_dataset=dataset,
            batch_size=128,
            default_dtype="float64",
            device=device,
            compute_stress=False,
        )

        energies.append(np.array(e))
        forces.append(np.array(f))

    energies = np.stack(energies)      # (n_models, n_structures)
    forces = np.stack(forces)          # (n_models, n_structures, n_atoms, 3)

    return energies, forces


# ==============================
# Save predictions
# ==============================
def save_predictions(path, energies, forces):

    np.savez_compressed(
        path,
        energies=energies,
        forces=forces
    )

    print(f"Predictions saved → {path}")


# ==============================
# Load predictions
# ==============================
def load_predictions(path):

    data = np.load(path)

    energies = data["energies"]
    forces = data["forces"]

    print(f"Loaded cached predictions ← {path}")

    return energies, forces


# ==============================
# Get predictions (cached or run)
# ==============================
def get_predictions(pred_file, models, dataset):

    if os.path.exists(pred_file):

        energies, forces = load_predictions(pred_file)

    else:

        print("Running ensemble predictions...")

        energies, forces = ensemble_predict(models, dataset)

        save_predictions(pred_file, energies, forces)

    return energies, forces


# ==============================
# Compute uncertainty
# ==============================
def compute_uncertainty(energies, forces):

    # energies: (n_models, n_structures)
    # forces:   (n_models, n_structures, n_atoms, 3)

    energy_std = np.std(energies, axis=0)

    force_std = np.std(forces, axis=0)

    per_atom_std = np.linalg.norm(force_std, axis=2)

    force_max_std = np.max(per_atom_std, axis=1)

    force_rms_std = np.sqrt(np.mean(per_atom_std ** 2, axis=1))

    return energy_std, force_max_std, force_rms_std


# ==============================
# Compute prediction errors
# ==============================
def compute_error(energies, forces, dataset):

    # true values
    true_energy = np.array([d.y.item() for d in dataset])

    true_forces = torch.stack([d.forces for d in dataset]).cpu().numpy()

    # ensemble mean prediction
    mean_energy = np.mean(energies, axis=0)

    mean_forces = np.mean(forces, axis=0)

    # energy error
    energy_error = np.abs(mean_energy - true_energy)

    # force error
    diff = mean_forces - true_forces

    diff_norm = np.linalg.norm(diff, axis=2)

    force_error = np.sqrt(np.mean(diff_norm ** 2, axis=1))

    return energy_error, force_error

# ==============================
# Main
# ==============================
def main():

    # --------------------------
    # Config
    # --------------------------
    n_models = 2
    num_atom = 11
    charge = -3

    prefix = f"bi{num_atom}{charge}_samples"

    model_dir = f"../usr/pretrain/results/charge_embedding/{prefix}_logs"

    csv_path = f"../results/{prefix}_org/56_added_data.csv"

    pred_file = f"{prefix}_ensemble_predictions.npz"


    # --------------------------
    # Load models
    # --------------------------
    models = load_models(model_dir, n_models, prefix)


    # --------------------------
    # Load dataset
    # --------------------------
    dataset = load_dataset(csv_path, num_atom, charge)


    # --------------------------
    # Predictions
    # --------------------------
    energies, forces = get_predictions(pred_file, models, dataset)

    print("Prediction shapes")
    print("Energies:", energies.shape)
    print("Forces:", forces.shape)


    # --------------------------
    # Uncertainty
    # --------------------------
    energy_std, force_max_std, force_rms_std = compute_uncertainty(
        energies,
        forces
    )


    # --------------------------
    # Error
    # --------------------------
    energy_error, force_error = compute_error(
        energies,
        forces,
        dataset
    )


    # --------------------------
    # Save results
    # --------------------------
    df = pd.DataFrame({
        "energy_std": energy_std,
        "force_max_std": force_max_std,
        "force_rms_std": force_rms_std,
        "energy_error": energy_error,
        "force_error": force_error
    })

    df.to_csv("std_vs_error.csv", index=False)

    print("Saved → std_vs_error.csv")


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    main()