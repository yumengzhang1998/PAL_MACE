# plot_iter_metrics.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt

prefix = 'bi11-3_samples'
model_number = 56
path = f"../results/{prefix}/"
json_path = os.path.join(path, f'retrain_history_{model_number}.json')

def get_series(d, candidates, name_for_error):
    for k in candidates:
        if k in d:
            return np.asarray(d[k]).ravel(), k
    raise KeyError(f"Missing data for {name_for_error}. Tried keys: {candidates}")

# === Load ===
with open(json_path, "r") as f:
    data = json.load(f)

# === Extract; robust to naming (MSE/MAE) ===
mse_train, k_train = get_series(data, ["MSE_train", "train_MSE", "mse_train"], "MSE_train")
mse_val,   k_val   = get_series(data, ["MSE_val", "val_MSE", "mse_val"], "MSE_val")

start_train, k_start_train = get_series(
    data, ["start_MSE_train", "start_MAE_train", "train_start_MSE", "train_start_MAE"],
    "start_*_train"
)
start_val, k_start_val = get_series(
    data, ["start_MSE_val", "start_MAE_val", "val_start_MSE", "val_start_MAE"],
    "start_*_val"
)

# === Align lengths ===
n_iter = min(len(mse_train), len(mse_val), len(start_train), len(start_val))
if n_iter == 0:
    raise ValueError("No iterations to plot after alignment.")
mse_train = mse_train[:n_iter]
mse_val   = mse_val[:n_iter]
start_train = start_train[:n_iter]
start_val   = start_val[:n_iter]

x = np.arange(1, n_iter + 1)  # iteration index: 1..n_iter

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax_tr, ax_val = axes

# Left: Train
ax_tr.plot(x, start_train, marker='o', linewidth=2, label='Start (Train)')
ax_tr.plot(x, mse_train,   marker='o', linewidth=2, label='After Retrain (Train)')
ax_tr.set_title(f"Train: Start vs After Retrain")
ax_tr.set_xlabel("Iteration")
ax_tr.set_ylabel("Metric value")
ax_tr.grid(True, linestyle='--', alpha=0.6)
ax_tr.legend()

# Right: Validation
ax_val.plot(x, start_val, marker='o', linewidth=2, label='Start (Val)')
ax_val.plot(x, mse_val,   marker='o', linewidth=2, label='After Retrain (Val)')
ax_val.set_title(f"Start vs After Retrain")
ax_val.set_xlabel("Iteration")
ax_val.grid(True, linestyle='--', alpha=0.6)
ax_val.legend()

fig.suptitle(f"Ranked {prefix} Per-Iteration Metrics (x = iteration index)")
fig.tight_layout()

out_path = os.path.join(path, f"{model_number}_iter_metrics_train_val.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.show()

print(f"Saved figure to: {out_path}")
