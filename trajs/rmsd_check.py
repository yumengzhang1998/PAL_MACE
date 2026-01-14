import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from ase import Atoms
from ase.io import read


def read_traj(traj_file):
    """Read a trajectory from a file."""
    with open(traj_file, "rb") as f:
        traj = pickle.load(f)
    return traj

def read_reference_xyz(ref_file):
    """Read the D3 symmetric reference structure from .xyz."""
    return read(ref_file)

def kabsch_rmsd(P, Q):
    """Align P to Q and compute RMSD (P and Q are Nx3 arrays)"""
    # Subtract centroid
    P -= P.mean(axis=0)
    Q -= Q.mean(axis=0)

    # Kabsch alignment
    C = np.dot(P.T, Q)
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, W)))
    U = np.dot(V, np.dot(np.diag([1, 1, d]), W))
    P_aligned = np.dot(P, U)

    return np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))

def compute_rmsd_traj(traj, reference: Atoms):
    """Compute RMSD of each frame in traj vs reference"""
    ref_pos = reference.get_positions()
    rmsd_values = []
    for frame in traj:
        coords = np.array(frame[0])  # atomic positions
        rmsd_val = kabsch_rmsd(coords.copy(), ref_pos.copy())
        rmsd_values.append(rmsd_val)
    return rmsd_values

def plot_rmsd_trends(rmsd_trajs, prefix, traj_name):
    time_steps = list(range(len(rmsd_trajs[0])))
    plt.figure(figsize=(8, 5))
    for rmsd_list in rmsd_trajs:
        plt.plot(time_steps, rmsd_list, alpha=0.7, label="_nolegend_")

    plt.xlabel("Time Step")
    plt.ylabel("RMSD to D₃ Symmetry (Å)")
    plt.title("RMSD Trend to D₃ Symmetric Reference Over Time")
    plt.grid(True)
    plt.savefig(f"{prefix}/{traj_name}/rmsd_traj.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSD to D₃ symmetric structure over time.")
    parser.add_argument("--element", type=str, required=True)
    parser.add_argument("--charge", type=int, required=True)
    parser.add_argument("--num_atom", type=int, required=True)
    parser.add_argument("--model_number", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--synthesis", type=str, required=True)
    parser.add_argument("--reference_xyz", type=str, required=True, help="Path to D3 symmetric .xyz file")

    args = parser.parse_args()

    if args.synthesis == "True":
        prefix = f"{args.element}{args.num_atom}{args.charge}_samples"
    else:
        prefix = f"{args.element}{args.num_atom}{args.charge}"

    traj_name = f"{args.model_number}_{args.steps}steps"
    output_dir = f"{prefix}/{traj_name}"
    os.makedirs(output_dir, exist_ok=True)

    all_trajs = read_traj(f"{prefix}/{traj_name}_traj.pkl")
    reference = read_reference_xyz(args.reference_xyz)

    # sample 10 trajectories
    random.seed(42)
    sample_trajs = random.sample(all_trajs, 150)
    rmsd_trajs = [compute_rmsd_traj(traj, reference) for traj in sample_trajs]

    plot_rmsd_trends(rmsd_trajs, prefix, traj_name)
    print(f"✅ RMSD trend plot saved in: {output_dir}/rmsd_traj.png")
