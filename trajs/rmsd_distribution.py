import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from ase import Atoms
from ase.io import read
import numpy as np
from scipy.optimize import linear_sum_assignment

def rmsd_hungarian(P, Q):
    # 去中心
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    # 距离矩阵
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)

    # Hungarian
    row_ind, col_ind = linear_sum_assignment(D)

    Q_perm = Q[col_ind]

    # Kabsch
    C = P.T @ Q_perm
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ W))
    U = V @ np.diag([1, 1, d]) @ W

    P_aligned = P @ U
    return np.sqrt(np.mean(np.sum((P_aligned - Q_perm) ** 2, axis=1)))

def read_traj(traj_file):
    with open(traj_file, "rb") as f:
        return pickle.load(f)

def read_reference_xyz(ref_file):
    return read(ref_file)

def kabsch_rmsd(P, Q):
    P -= P.mean(axis=0)
    Q -= Q.mean(axis=0)
    C = np.dot(P.T, Q)
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, W)))
    U = np.dot(V, np.dot(np.diag([1, 1, d]), W))
    P_aligned = np.dot(P, U)
    return np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))

def compute_all_rmsds(trajs, reference):
    ref_pos = reference.get_positions()
    all_rmsd_values = []
    traj_indices_with_match = set()
    labels = []

    for idx, traj in enumerate(trajs):
        label = traj[0][7]
        labels.append(label)
        for frame in traj:
            if label!= 'real':
                coords = np.array(frame[0])
                rmsd = rmsd_hungarian(coords.copy(), ref_pos.copy())
                all_rmsd_values.append(rmsd)
                if rmsd < 1.5:
                    traj_indices_with_match.add(idx)
    return all_rmsd_values, sorted(list(traj_indices_with_match)), labels

def plot_rmsd_distribution(rmsd_values, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(rmsd_values, bins=50, alpha=0.75, color='skyblue')
    plt.xlabel("RMSD to Optimal Structure (\u00c5)")
    plt.ylabel("Count")
    plt.title("RMSD Distribution Across All Trajectories")
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RMSD distribution and identify similar structures.")
    parser.add_argument("--traj_file", type=str, required=True, help="Pickle file containing all trajectories")
    parser.add_argument("--reference_xyz", type=str, required=True, help="Reference structure (.xyz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    trajs = read_traj(args.traj_file)
    reference = read_reference_xyz(args.reference_xyz)

    rmsd_values, matching_traj_indices, labels = compute_all_rmsds(trajs, reference)
    print(labels)

    print(f"\n✅ RMSD < 1.5 \u00c5 found in {len(matching_traj_indices)} trajectories.")
    print("Matching trajectory indices:", matching_traj_indices)

    plot_path = os.path.join(args.output_dir, "rmsd_distribution.png")
    plot_rmsd_distribution(rmsd_values, plot_path)
    print(f"\n✅ Histogram saved to: {plot_path}")
