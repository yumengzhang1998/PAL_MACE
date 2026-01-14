import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import argparse
import os


def compute_rdf_from_positions(positions, r_edges):
    rdf_hist = np.zeros(len(r_edges) - 1)
    for pos in positions:
        dists = pdist(pos)
        hist, _ = np.histogram(dists, bins=r_edges)
        rdf_hist += hist

    rdf_hist /= len(positions)
    shell_volumes = 4 / 3 * np.pi * (np.power(r_edges[1:], 3) - np.power(r_edges[:-1], 3))
    g_r = rdf_hist / shell_volumes
    return g_r


def load_rdf_chunks(h5_path, interval, r_max=10.0, bins=50):
    with h5py.File(h5_path, "r") as h5f:
        traj_grp = h5f["trajectories"]
        traj_keys = sorted([k for k in traj_grp.keys() if k.startswith("traj_")])

        # Load first trajectory to get length
        traj_0 = traj_grp[traj_keys[0]]
        traj_length = traj_0.shape[0]

        r_edges = np.linspace(0, r_max, bins)
        r_mid = (r_edges[:-1] + r_edges[1:]) / 2

        rdf_list = []
        for t in range(0, traj_length, interval):
            frames = []
            for key in traj_keys:
                coords = traj_grp[key]
                if t < coords.shape[0]:
                    frames.append(coords[t])
            if frames:
                g_r = compute_rdf_from_positions(frames, r_edges)
                rdf_list.append(g_r)

        return r_mid, r_edges, np.array(rdf_list)



def plot_rdf_heatmap(r_mid, r_edges, rdf_array, output_path, interval_steps=1000, timestep_fs=2):
    num_chunks = rdf_array.shape[0]
    time_per_chunk_ps = interval_steps * timestep_fs / 1000
    times_ps = np.arange(num_chunks) * time_per_chunk_ps

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        rdf_array,
        aspect="auto",
        origin="lower",
        extent=[r_edges[0], r_edges[-1], times_ps[0], times_ps[-1]],
        cmap="inferno",
        interpolation="nearest"
    )

    # Grid lines
    for t in times_ps:
        ax.axhline(y=t, color='gray', linewidth=0.3, linestyle='--', alpha=0.3)
    for r in range(int(r_mid[-1]) + 1):
        ax.axvline(r, color='white', linewidth=0.3, linestyle='-', alpha=0.3)

    ax.set_xlabel("Distance r (Å)")
    ax.set_ylabel(f"Time (ps, Δt={time_per_chunk_ps:.2f} ps)")
    ax.set_title("Time-Resolved RDF Heatmap")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("g(r) (raw)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Unnormalized RDF heatmap saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unnormalized RDF heatmap from HDF5 trajectory.")
    parser.add_argument("--input", type=str, required=True, help="Path to .h5 trajectory file")
    parser.add_argument("--output", type=str, default="rdf_heatmap_raw.png", help="Output plot filename")
    parser.add_argument("--interval", type=int, default=1000, help="Steps per RDF chunk")
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument("--rmax", type=float, default=10.0)

    args = parser.parse_args()

    r_mid, r_edges, rdf_array = load_rdf_chunks(args.input, args.interval, args.rmax, args.bins)
    plot_rdf_heatmap(r_mid, r_edges, rdf_array, args.output, interval_steps=args.interval, timestep_fs=2)
