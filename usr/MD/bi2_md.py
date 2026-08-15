#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import pandas as pd

from batch_traj_full_h5 import stratified_sample, Generate_TrajsBatch


# ----------------------------
# Geometry helpers
# ----------------------------

def center(coords):
    return coords - np.mean(coords, axis=0)


def random_rotation(coords):
    return R.random().apply(coords)


def translate(coords, vec):
    return coords + vec


def min_pair_distance(A, B):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    return dist.min()


def no_clash(*mols, min_atomic_distance=3.0):
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            if min_pair_distance(mols[i], mols[j]) < min_atomic_distance:
                return False
    return True


# ----------------------------
# Placement
# ----------------------------

def place_fragment(core, frag,
                   target_dist=5.0,
                   tol=0.2,
                   min_atomic_distance=3.0,
                   max_expand_steps=200,
                   max_refine_steps=50,
                   initial_step=0.3):
    """
    Place frag around core so that min interatomic distance is ~ target_dist.
    Returns placed fragment coords or None.
    """
    frag_rot = random_rotation(frag)

    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    t = 0.0

    for _ in range(max_expand_steps):
        trial = translate(frag_rot, direction * t)
        d = min_pair_distance(core, trial)

        if d >= target_dist:
            break
        t += initial_step
    else:
        return None

    for _ in range(max_refine_steps):
        trial = translate(frag_rot, direction * t)
        d = min_pair_distance(core, trial)

        if abs(d - target_dist) < tol and no_clash(core, trial, min_atomic_distance=min_atomic_distance):
            return trial

        t += (target_dist - d) * 0.5

    return None


# ----------------------------
# Generator: Bi2 + Bi7 + Bi2 only
# ----------------------------

def generate_bi2_bi7_bi2(bi2, bi7, n,
                         target_dist=4.5,
                         bi2_bi2_min_distance=None,
                         core_frag_min_atomic_distance=3.0,
                         max_attempts_per_sample=1000):
    """
    Generate n structures of type Bi2 + Bi7 + Bi2.

    bi2_bi2_min_distance:
        minimum allowed interatomic distance between the two placed Bi2 fragments.
        If None, defaults to target_dist.
    """
    bi2 = center(np.asarray(bi2))
    bi7 = center(np.asarray(bi7))

    if bi2_bi2_min_distance is None:
        bi2_bi2_min_distance = target_dist

    samples = []

    for _ in tqdm(range(n), desc="Generating Bi2+Bi7+Bi2"):
        success = False

        for _attempt in range(max_attempts_per_sample):
            a = place_fragment(
                bi7, bi2,
                target_dist=target_dist,
                min_atomic_distance=core_frag_min_atomic_distance
            )
            if a is None:
                continue

            b = place_fragment(
                bi7, bi2,
                target_dist=target_dist,
                min_atomic_distance=core_frag_min_atomic_distance
            )
            if b is None:
                continue

            if not no_clash(a, b, min_atomic_distance=bi2_bi2_min_distance):
                continue

            merged = np.vstack([a, bi7, b])
            samples.append(merged)
            success = True
            break

        if not success:
            raise RuntimeError(
                f"Failed to generate sample after {max_attempts_per_sample} attempts. "
                f"Try relaxing target_dist or clash thresholds."
            )

    return samples


# ----------------------------
# IO
# ----------------------------

def read_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()

    mols = []
    i = 0
    while i < len(lines):
        n = int(lines[i].strip())
        i += 2  # skip atom count + comment line
        coords = []
        for _ in range(n):
            parts = lines[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        mols.append(np.array(coords, dtype=float))

    return mols


def save_npz(filename, coords, labels):
    np.savez_compressed(filename, coords=coords, labels=labels)
    print(f"Saved dataset: {filename}")
    print("coords shape:", coords.shape)
    print("labels shape:", labels.shape)


# ----------------------------
# MD batch builder
# ----------------------------

def build_data_batch(coords, labels, charge):
    """
    Convert generated coords into your internal data_batch format.
    """
    n_samples, n_atoms, _ = coords.shape
    atom_numbers = np.array([83] * n_atoms, dtype=int)  # Bi = 83

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
# Optional sampling helper
# ----------------------------

def subsample_batch(coords, labels, batch_size):
    if batch_size is None or batch_size >= len(coords):
        return coords, labels

    df_tmp = pd.DataFrame({
        "idx": np.arange(len(coords)),
        "source": labels
    })

    df_tmp["source"] = df_tmp["source"].astype(str).str.strip().str.lower()

    # Same logic style as your existing script.
    df_syn = df_tmp[df_tmp["source"] != "real"].reset_index(drop=True)
    df_real = df_tmp[df_tmp["source"] == "real"].reset_index(drop=True)

    n_total = min(batch_size, len(df_syn))
    df_sampled = stratified_sample(df_syn, "source", n_total)

    if len(df_real) > 0:
        df_real_sampled = df_real.sample(len(df_sampled) // 2, random_state=42)
        df_sampled = pd.concat([df_sampled, df_real_sampled], ignore_index=True)

    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    idx = df_sampled["idx"].values
    return coords[idx], labels[idx]


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Bi2-based starting structures and run MD in one script."
    )

    # Required by your request
    parser.add_argument("--target_dist", type=float, required=True,
                        help="Target minimum inter-fragment distance used during structure generation.")
    parser.add_argument("--number", type=int, required=True,
                        help="Number of Bi2+Bi7+Bi2 structures to generate.")

    # MD args
    parser.add_argument("--model_number", type=int, required=True,
                        help="Model number for Generate_TrajsBatch.")
    parser.add_argument("--steps", type=int, required=True,
                        help="Number of MD steps.")

    # Optional settings
    parser.add_argument("--charge", type=int, default=-3)
    parser.add_argument("--T", type=float, default=700.0)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional subsample size before MD. Default: use all generated structures.")

    parser.add_argument("--prefix_bi2", type=str, default="Bi2-2")
    parser.add_argument("--prefix_bi7", type=str, default="Bi7-3")
    parser.add_argument("--building_block_dir", type=str, default="building_blocks")
    parser.add_argument("--out_dir", type=str, default="bi2")

    parser.add_argument("--core_frag_min_atomic_distance", type=float, default=3.0,
                        help="Min allowed atom-atom distance between Bi7 and one Bi2 fragment.")
    parser.add_argument("--bi2_bi2_min_distance", type=float, default=None,
                        help="Min allowed atom-atom distance between the two Bi2 fragments. "
                             "Default: same as target_dist.")
    parser.add_argument("--tol", type=float, default=0.2,
                        help="Placement tolerance around target_dist.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- load building blocks --------
    bi2_path = Path(args.building_block_dir) / f"{args.prefix_bi2}_optimized.xyz"
    bi7_path = Path(args.building_block_dir) / f"{args.prefix_bi7}_optimized.xyz"

    bi2 = read_xyz(str(bi2_path))[0]
    bi7 = read_xyz(str(bi7_path))[0]

    bi2 = np.atleast_2d(bi2)
    bi7 = np.atleast_2d(bi7)

    print("Loaded building blocks:")
    print("bi2 shape:", bi2.shape)
    print("bi7 shape:", bi7.shape)

    # -------- generate structures --------
    samples_bi2 = generate_bi2_bi7_bi2(
        bi2=bi2,
        bi7=bi7,
        n=args.number,
        target_dist=args.target_dist,
        bi2_bi2_min_distance=args.bi2_bi2_min_distance,
        core_frag_min_atomic_distance=args.core_frag_min_atomic_distance
    )

    coords = np.array(samples_bi2, dtype=float)
    labels = np.array(["synthesis_bi2"] * len(coords))

    print("Generated dataset:", coords.shape)

    # Save generated structures for later reuse/debugging
    dataset_prefix = f"bi2_td{args.target_dist:g}_n{args.number}"
    npz_path = out_dir / f"{dataset_prefix}.npz"
    save_npz(npz_path, coords, labels)

    # -------- optional subsample --------
    coords_md, labels_md = subsample_batch(coords, labels, args.batch_size)

    print(f"Using {len(coords_md)} trajectories for MD")

    # -------- build batch --------
    data_batch = build_data_batch(coords_md, labels_md, args.charge)

    # -------- result paths --------
    job_tmp = Path(os.environ.get("PAL_MACE_JOB_TMP", "./tmp"))
    result_path = job_tmp / "results" / dataset_prefix
    result_path.mkdir(parents=True, exist_ok=True)

    out_file = result_path / f"{args.model_number}_{args.steps}steps_traj.h5"

    # -------- run MD --------
    traj_gen = Generate_TrajsBatch(
        data_batch=data_batch,
        result_path=result_path,
        model_number=args.model_number,
        prefix="bi11-3_samples",
        temperature=args.T,
    )

    try:
        traj_gen.run_mixed_skip_bad(
            steps=args.steps,
            h5_path=str(out_file),
            label_batch=labels_md.tolist(),
            final_path=str(out_file),
        )
    except Exception:
        traceback.print_exc()
        raise

    print("MD finished.")
    print("Trajectory file:", out_file)


if __name__ == "__main__":
    main()
