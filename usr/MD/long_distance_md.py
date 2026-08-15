import numpy as np
import argparse
import os
from pathlib import Path
import periodictable
import traceback

# your existing class
from batch_traj_full_h5 import stratified_sample, Generate_TrajsBatch   # <-- CHANGE THIS IMPORT


# ----------------------------
# NPZ Loader
# ----------------------------

def load_npz_dataset(npz_path):
    data = np.load(npz_path)

    coords = data["coords"]       # (N, n_atoms, 3)
    labels_raw = data["labels"]   # (N,)

    print("Loaded NPZ:")
    print("coords:", coords.shape)
    print("labels:", labels_raw.shape)

    # ----------------------------
    # FIX LABELS HERE
    # ----------------------------
    label_map = {
        0: "synthesis_bi4",
        1: "synthesis_bi2"
    }

    labels = np.array([
        label_map.get(int(l), str(l)) for l in labels_raw
    ])

    print("Mapped labels example:", labels[:5])

    return coords, labels

def build_data_batch(coords, labels, charge):
    """
    Convert NPZ → your internal data_batch format
    """
    n_samples, n_atoms, _ = coords.shape

    atom_numbers = np.array([83] * n_atoms)  # Bi = 83

    data_batch = []

    for i in range(n_samples):
        data_batch.append([
            coords[i],          # positions
            atom_numbers,       # atomic numbers
            None,               # true_energy
            None,               # true_forces
            charge,
            None,               # pred_forces
            None,               # pred_energy
            labels[i],          # label
            None
        ])

    return data_batch


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--model_number", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--charge", type=int, default=-3)
    parser.add_argument("--T", type=float, default=700.0)

    args = parser.parse_args()

    # -------- load dataset --------
    coords, labels = load_npz_dataset(args.npz)

    # -------- optional subsample --------
    batch_size = 100

    if labels is not None:
        import pandas as pd

        df_tmp = pd.DataFrame({
            "idx": np.arange(len(coords)),
            "source": labels
        })

        df_tmp["source"] = df_tmp["source"].astype(str).str.strip().str.lower()

        df_syn = df_tmp[df_tmp["source"] != "real"].reset_index(drop=True)
        df_real = df_tmp[df_tmp["source"] == "real"].reset_index(drop=True)

        n_total = min(batch_size, len(df_syn))
        df_sampled = stratified_sample(df_syn, "source", n_total)

        if len(df_real) > 0:
            df_real_sampled = df_real.sample(len(df_sampled)//2, random_state=42)
            df_sampled = pd.concat([df_sampled, df_real_sampled])

        df_sampled = df_sampled.sample(frac=1, random_state=42)

        idx = df_sampled["idx"].values
        labels = df_sampled["source"].values

    else:
        idx = np.random.choice(len(coords), min(batch_size, len(coords)), replace=False)
        labels = labels[idx]

    coords = coords[idx]
    

    print(f"Using {len(coords)} trajectories")

    # -------- build batch --------
    data_batch = build_data_batch(coords, labels, args.charge)

    # -------- paths --------
    prefix = Path(args.npz).stem
    job_tmp = Path(os.environ.get("PAL_MACE_JOB_TMP", "./tmp"))

    result_path = job_tmp / "results" / prefix
    result_path.mkdir(parents=True, exist_ok=True)

    out_file = result_path / f"{args.model_number}_{args.steps}steps_traj.h5"

    # -------- run MD --------
    traj_gen = Generate_TrajsBatch(
        data_batch=data_batch,
        result_path=result_path,
        model_number=args.model_number,
        prefix=None,
        temperature=args.T,
    )

    try:
        traj_gen.run_mixed_skip_bad(
            steps=args.steps,
            h5_path=str(out_file),
            label_batch=labels.tolist(),
            final_path=str(out_file),
        )
    except Exception:
        traceback.print_exc()
        raise