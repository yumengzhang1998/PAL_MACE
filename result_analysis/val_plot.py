#!/usr/bin/env python3
# plot_val_mae_and_force.py
"""
Make two subplots over iteration index:
  1) init_val_mae vs new_val_mae (validation set)
  2) init_val_force_mae vs new_val_force_mae (validation set)

Usage:
  python plot_val_mae_and_force.py --json retrain_history_57.json
  python plot_val_mae_and_force.py --json retrain_history_57.json --out fig.png --logy
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def arr(d, key, friendly):
    if key not in d:
        raise KeyError(f"Missing key '{key}' for {friendly}. "
                       f"Available keys: {list(d.keys())[:10]} ...")
    return np.asarray(d[key]).ravel()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=str, default="bi4-2",
                   help="cluster prefix (not used directly, for user reference).")
    p.add_argument("--model_number", type=int, default=56,
                   help="model number (not used directly, for user reference).")

    args = p.parse_args()
    file_path = f"../results/{args.prefix}/retrain_history_{args.model_number}.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    # --- Extract & flatten ---
    init_val_mae = arr(data, "init_val_mae", "initial validation MAE")
    new_val_mae  = arr(data, "new_val_mae",  "new validation MAE")
    init_val_f   = arr(data, "init_val_force_mae", "initial validation force MAE")
    new_val_f    = arr(data, "new_val_force_mae",  "new validation force MAE")

    # --- Align lengths (just in case) ---
    n_iter = min(len(init_val_mae), len(new_val_mae), len(init_val_f), len(new_val_f))
    if n_iter == 0:
        raise ValueError("No iterations found. Check arrays in the JSON.")
    init_val_mae = init_val_mae[:n_iter]
    new_val_mae  = new_val_mae[:n_iter]
    init_val_f   = init_val_f[:n_iter]
    new_val_f    = new_val_f[:n_iter]

    x = np.arange(1, n_iter + 1)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle(f"Validation MAE and Force MAE over Iterations for {args.prefix} Model {args.model_number} ")
    ax_mae, ax_force = axes

    # Left: energy MAE on validation set
    ax_mae.plot(x, init_val_mae, marker="o", linewidth=2, label="Init val MAE")
    ax_mae.plot(x, new_val_mae,  marker="o", linewidth=2, label="New val MAE")
    ax_mae.set_title(f"{args.prefix} Validation set: MAE (Energy)")
    ax_mae.set_xlabel("Iteration")
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, linestyle="--", alpha=0.6)
    ax_mae.legend()

    # Right: force MAE on validation set
    ax_force.plot(x, init_val_f, marker="o", linewidth=2, label="Init val Force MAE")
    ax_force.plot(x, new_val_f,  marker="o", linewidth=2, label="New val Force MAE")
    ax_force.set_title(f"{args.prefix} Validation set: MAE (Forces)")
    ax_force.set_xlabel("Iteration")
    ax_force.grid(True, linestyle="--", alpha=0.6)
    ax_force.legend()

    fig.tight_layout()

    # --- Save ---

    plt.savefig(f"../results/{args.prefix}/{args.model_number}_val_plot.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {args.prefix}/{args.model_number}_val_plot.png")

if __name__ == "__main__":
    main()
