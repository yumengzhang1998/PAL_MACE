#!/usr/bin/env python3
# compare_pred_vs_dft.py
#
# End-to-end script to:
# - load predicted trajectories (energies & forces) from a single pickle
# - load DFT energies/forces from pickle(s)
# - align by downsampling every --dft_interval steps
# - compute per-trajectory metrics (bias, std, RMSE, linear drift)
# - plot time-series overlays, residual histograms
# - plot Predicted vs DFT ENERGY scatter, colored by traj label (if provided)
# - plot Predicted vs DFT FORCE scatter/hist (also label-aware)
# - label-aware distributions & separate per-label plots

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from sympy import N


# -------------------- IO helpers --------------------
def read_pickle_stream(path: Path):
    """Read one or many pickles written sequentially. If it's a single list, return that list."""
    items = []
    with open(path, "rb") as f:
        try:
            while True:
                items.append(pickle.load(f))
        except EOFError:
            pass
    if len(items) == 1 and isinstance(items[0], list):
        return items[0]
    return items


def read_traj(traj_file: Path):
    with open(traj_file, "rb") as f:
        return pickle.load(f)  # list of trajectories; each trajectory is list of frames


# -------------------- data extraction --------------------
def get_energy_list(traj):
    """Energy stored as frame[-3]."""
    return [float(frame[-3]) for frame in traj]


def _coerce_force_frame(f):
    """Coerce a 'force frame' into (n_atoms, 3) float array. Return None if impossible."""
    # torch tensor -> numpy
    if hasattr(f, "detach") and hasattr(f, "cpu"):
        try:
            f = f.detach().cpu().numpy()
        except Exception:
            pass

    # dicts
    if isinstance(f, dict):
        for key in ("forces", "F", "force", "f"):
            if key in f:
                return _coerce_force_frame(f[key])
        comps = []
        for key in ("fx", "fy", "fz", "Fx", "Fy", "Fz", "FX", "FY", "FZ"):
            if key in f:
                comps.append(np.asarray(f[key], dtype=float).ravel())
        if len(comps) == 3:
            try:
                arr = np.stack(comps, axis=-1)
                return np.asarray(arr, dtype=float)
            except Exception:
                return None

    # ndarray/list/tuple
    try:
        arr = np.asarray(f)
    except Exception:
        return None

    if arr.dtype == object:
        rows = []
        for item in f:
            a = np.asarray(item, dtype=float).ravel()
            if a.size == 3:
                rows.append(a)
            elif a.size % 3 == 0:
                rows.extend(a.reshape(-1, 3))
            else:
                return None
        return np.asarray(rows, dtype=float) if rows else None

    try:
        arr = np.asarray(arr, dtype=float)
    except Exception:
        return None

    if arr.ndim == 2:
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
        flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
    elif arr.ndim == 1 and arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    else:
        flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
    return None


def get_force_list(traj):
    """Forces stored at frame[-4]. Return list of (n_atoms, 3) arrays or None."""
    out = []
    for frame in traj:
        f = frame[-4]
        out.append(_coerce_force_frame(f))
    return out


def get_label_for_traj(traj):
    """Label stored as traj[0][-2]; constant per trajectory."""
    return traj[0][-2]


def to_float_array_or_nan(seq):
    """Convert a list like [ [E], None, [E], ... ] into a float array with NaNs where needed."""
    out = []
    for e in seq:
        if e is None:
            out.append(np.nan)
            continue
        try:
            v = e[0] if (isinstance(e, (list, tuple, np.ndarray)) and len(e) > 0) else e
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


# -------------------- utilities --------------------
def downsample(series, step):
    """Return values at indices 0, step, 2*step, ... within bounds."""
    idx = np.arange(0, len(series), step, dtype=int)
    return np.asarray([series[i] for i in idx], dtype=float), idx


def parse_idx_list(text):
    if text is None or str(text).strip() == "":
        return None
    return [int(x) for x in str(text).replace(" ", "").split(",") if x != ""]


def choose_label_colors(unique_labels):
    """Return a dict label->color. If exactly 2 labels, use highly distinct, colorblind-safe colors."""
    if len(unique_labels) == 2:
        # Okabe–Ito palette: Blue & Vermillion (very distinguishable)
        special = ["#0072B2", "#D55E00"]
        return {lbl: special[i] for i, lbl in enumerate(unique_labels)}
    cmap = plt.get_cmap("tab20")
    return {lbl: cmap(i % cmap.N) for i, lbl in enumerate(unique_labels)}


def safe_label_str(lbl):
    """Make a label safe for filenames."""
    return str(lbl).replace("/", "_").replace(" ", "_")


# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser(description="Compare downsampled predicted vs DFT energies/forces; label-colored plots.")
    # Dataset identification
    p.add_argument("--element", required=True)
    p.add_argument("--charge", type=int, required=True)
    p.add_argument("--num_atom", type=int, required=True)
    p.add_argument("--model_number", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--synthesis", type=str, required=True, help="True/False; affects prefix naming")
    p.add_argument("--base", type=str, default="bi4")

    # Files & paths
    p.add_argument("--traj_pkl", type=str, default=None, help="Path to *traj.pkl if not in standard location")
    p.add_argument("--dft_pkl", type=str, default=None, help="Path to dft_energy_list.pkl (or pickle stream)")
    p.add_argument("--force_pkl", type=str, default=None, help="Path to dft_force_list.pkl (or pickle stream)")
    p.add_argument("--optimized_csv", type=str, default="optimized.csv", help="CSV with optimized energy (columns: Name,Energy)")
    p.add_argument("--output_dir", type=str, default=None)

    # Selection & sampling
    p.add_argument("--dft_interval", type=int, default=500)
    p.add_argument("--traj_indices", type=str, default=None, help="Comma-separated trajectory indices to analyze, e.g. '20,3,0,23,8'")
    p.add_argument("--select_indices", type=str, default=None, help="Subset indices into --traj_indices (e.g. '0,3,4')")

    args = p.parse_args()

    synthesis_flag = args.synthesis == "True"
    prefix = f"{args.element}{args.num_atom}{args.charge}_samples" if synthesis_flag else f"{args.element}{args.num_atom}{args.charge}"
    if args.base != "bi4":
        prefix = f"{prefix}_{args.base}"
    traj_name = f"{args.model_number}_{args.steps}steps"

    # Resolve paths
    traj_pkl = Path(args.traj_pkl) if args.traj_pkl else Path(f"{prefix}/{traj_name}_traj.pkl")
    if args.dft_pkl:
        dft_pkl = Path(args.dft_pkl)
    else:
        dft_pkl = Path(f"{prefix}/{traj_name}/dft_energy_list.pkl")
    if args.force_pkl:
        force_pkl = Path(args.force_pkl)
    else:
        force_pkl = Path(f"{prefix}/{traj_name}/dft_force_list.pkl")

    out_dir = Path(args.output_dir) if args.output_dir else Path(prefix) / traj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load predicted trajectories --------
    trajs = read_traj(traj_pkl)  # list of trajectories; each trajectory is list of frames
    print(f"Loaded {len(trajs)} trajectories from {traj_pkl}")

    energy_trajs = [get_energy_list(tr) for tr in trajs]
    force_trajs  = [get_force_list(tr)  for tr in trajs]
    label_trajs  = [get_label_for_traj(tr) for tr in trajs]
    has_label = all(lbl is not None for lbl in label_trajs)

    # -------- Load optimized energy from CSV --------
    df_opt = pd.read_csv(args.optimized_csv)
    df_opt["Name"] = df_opt["Name"].str.lower()
    name_key = prefix.lower()
    if name_key == "bi11-3_samples":
        name_key = "bi11-3"
    row = df_opt[df_opt["Name"] == name_key]
    if row.empty:
        raise RuntimeError(f"Could not find optimized energy for '{prefix}' in {args.optimized_csv}")
    optimized_energy = float(eval(row.iloc[0]["Energy"])[0])  # CSV stores stringified list
    if prefix.lower() == "bi11-3_samples":
        print("11111111111111111111111111111111111111111111111111111111111111111111111111111111")
        row_bi4 = df_opt[df_opt["Name"] == "bi11-3_samples"]
        optimized_bi4 = float(eval(row_bi4.iloc[0]["Energy"])[0]) if not row_bi4.empty else None
        row_bi2 = df_opt[df_opt["Name"] == "bi11-3_samples_bi2"]
        optimized_bi2 = float(eval(row_bi2.iloc[0]["Energy"])[0]) if not row_bi2.empty else None
        
    # Shift predicted energies by optimized
    pred_shifted = [np.asarray(tr, dtype=float) - optimized_energy for tr in energy_trajs]

    # -------- Load DFT energies/forces --------
    if not dft_pkl.exists():
        raise FileNotFoundError(f"DFT energy pickle not found: {dft_pkl}")
    dft_energy_lists_raw = read_pickle_stream(dft_pkl)  # usually list of lists (per traj)

    if not force_pkl.exists():
        print(f"WARNING: DFT force pickle not found: {force_pkl}")
        dft_force_lists_raw = []
    else:
        dft_force_lists_raw = read_pickle_stream(force_pkl)

    # Convert DFT energies to floats and shift by optimized
    dft_energy_lists = [to_float_array_or_nan(lst) - optimized_energy for lst in dft_energy_lists_raw]

    # -------- Select / reorder trajectories if requested --------
    n_traj = len(trajs)
    base_indices = list(range(n_traj))
    if args.traj_indices:
        idx_list = parse_idx_list(args.traj_indices)
        idx_list = [i for i in idx_list if 0 <= i < n_traj]
    else:
        idx_list = base_indices

    if args.select_indices:
        sel_pos = parse_idx_list(args.select_indices)  # positions inside idx_list
        sel_pos = [i for i in sel_pos if 0 <= i < len(idx_list)]
        final_indices = [idx_list[i] for i in sel_pos]
    else:
        final_indices = idx_list

    def take(seq, idxs):
        return [seq[i] for i in idxs]

    pred_shifted   = take(pred_shifted, final_indices)
    force_trajs    = take(force_trajs, final_indices)
    label_trajs    = take(label_trajs, final_indices)

    # DFT files should be aligned with original traj order. If lengths mismatch, cut to min length.
    if len(dft_energy_lists) != len(label_trajs):
        print(f"WARNING: DFT energy list count ({len(dft_energy_lists)}) != #trajs ({n_traj}). "
              f"Will slice to the first {len(final_indices)} entries.")
        dft_energy_lists = dft_energy_lists[:len(final_indices)]
        if len(dft_force_lists_raw) >= len(final_indices):
            dft_force_lists = dft_force_lists_raw[:len(final_indices)]
        else:
            dft_force_lists = dft_force_lists_raw
    else:
        dft_indices = list(range(len(final_indices)))
        dft_energy_lists = take(dft_energy_lists, dft_indices)
        dft_force_lists  = take(dft_force_lists_raw, dft_indices) if dft_force_lists_raw else []

    # -------- Downsample predictions at DFT interval --------
    interval = args.dft_interval
    pred_ds_all, time_ds_all = [], []
    for tr in pred_shifted:
        vals, idx = downsample(tr, interval)
        pred_ds_all.append(vals)
        time_ds_all.append(idx)

    # -------- Compare energies --------
    metrics_rows = []
    residual_series = []
    residuals_by_label = defaultdict(list)  # for label-wise distribution plots
    all_dft_scatter_by_label = defaultdict(list)
    all_pred_scatter_by_label = defaultdict(list)
    energy_mae_by_traj = []
    energy_mae_time_series = []  # list of (time_idx, mae_series, label)


    for i, (pred_ds, dft_seq) in enumerate(zip(pred_ds_all, dft_energy_lists), start=1):
        dft_ds = np.asarray(dft_seq, dtype=float)
        n = min(len(pred_ds), len(dft_ds))
        if n == 0:
            continue

        pred_use = pred_ds[:n].astype(float)
        dft_use  = dft_ds[:n].astype(float)

        mask = np.isfinite(pred_use) & np.isfinite(dft_use)
        pred_use = pred_use[mask]
        dft_use  = dft_use[mask]
        n = pred_use.size
        if n == 0:
            continue

        res = pred_use - dft_use
        energy_mae_series = np.abs(pred_use - dft_use)
        time_idx = time_ds_all[i-1][:n]
        energy_mae_time_series.append(
            (time_idx, energy_mae_series, label_trajs[i-1] if has_label else "all")
        )

        energy_mae_by_traj.append(float(np.mean(energy_mae_series)))
        residual_series.append(res)
        lbl = label_trajs[i-1] if has_label else "all"
        residuals_by_label[lbl].append(res)

        bias = float(np.mean(res)) if n else np.nan
        std  = float(np.std(res, ddof=1)) if n > 1 else np.nan
        rmse = float(np.sqrt(np.mean(res**2))) if n else np.nan

        x = np.arange(n, dtype=float)
        dft_slope  = float(np.polyfit(x, dft_use, 1)[0]) if n >= 2 else np.nan
        pred_slope = float(np.polyfit(x, pred_use, 1)[0]) if n >= 2 else np.nan
        mae = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res**2)))

        ss_res = np.sum((pred_use - dft_use)**2)
        ss_tot = np.sum((dft_use - np.mean(dft_use))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


        metrics_rows.append({
            "traj": i,
            "label": label_trajs[i-1] if has_label else None,
            "n_points": n,
            "bias_eV": bias,
            "std_eV": std,
            "rmse_eV": rmse,
            "dft_drift_per_frame_eV": dft_slope,
            "pred_drift_per_frame_eV": pred_slope,
            "mae_eV": mae,
            "r2": r2,
        })

        # group by label for scatter coloring
        all_dft_scatter_by_label[lbl].append(dft_use)
        all_pred_scatter_by_label[lbl].append(pred_use)

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "pred_vs_dft_metrics_energy.csv"
    metrics.to_csv(metrics_path, index=False)

    # -------- Plot aligned curves (energy vs time) --------
    # Combined overlay (all trajs; label aware in legend)
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (pred_ds, dft_seq, idx) in enumerate(zip(pred_ds_all, dft_energy_lists, time_ds_all), start=1):
        n = min(len(pred_ds), len(dft_seq))
        if n == 0:
            continue
        col = colors[(i-1) % len(colors)] if colors else None
        if has_label:
            lbl = label_trajs[i-1]
            ax.plot(idx[:n], dft_seq[:n], label=f"DFT traj {i} ({lbl})", linewidth=2, color=col)
            ax.plot(idx[:n], pred_ds[:n], "--", label=f"Pred traj {i} ({lbl})", alpha=0.9, color=col)
        else:
            ax.plot(idx[:n], dft_seq[:n], label=f"DFT traj {i}", linewidth=2, color=col)
            ax.plot(idx[:n], pred_ds[:n], "--", label=f"Pred traj {i}", alpha=0.9, color=col)

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1, label="Optimized Energy")
    if prefix.lower() == "bi11-3_samples":
        ax.axhline(optimized_bi4 - optimized_energy, color="green", linestyle="--", linewidth=1, label="Optimized Bi4 Energy")
        ax.axhline(optimized_bi2 - optimized_energy, color="orange", linestyle="--", linewidth=1, label="Optimized Bi2 Energy")
    ax.set_xlabel(f"Time step (every {interval})")
    ax.set_ylabel("Energy - Optimized (eV)")
    ax.set_title("Downsampled Predicted vs DFT Energies")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "downsampled_pred_vs_dft.png", dpi=180)
    plt.close(fig)

    # -------- NEW: separate per-label energy overlays --------
    # -------- NEW: separate per-label energy overlays (FIXED COLORS) --------
    if has_label:
        unique_labels = sorted(set(label_trajs))

        for lbl in unique_labels:
            fig, ax = plt.subplots(figsize=(8, 5))

            # collect trajectory indices for this label
            traj_indices = [
                i for i, L in enumerate(label_trajs)
                if L == lbl
            ]

            # generate one color per trajectory
            cmap = plt.get_cmap("tab10")
            colors = {i: cmap(k % cmap.N) for k, i in enumerate(traj_indices)}

            for i in traj_indices:
                pred_ds = pred_ds_all[i]
                dft_seq = dft_energy_lists[i]
                idx     = time_ds_all[i]

                n = min(len(pred_ds), len(dft_seq))
                if n == 0:
                    continue

                color = colors[i]

                # DFT: solid
                ax.plot(
                    idx[:n],
                    dft_seq[:n],
                    color=color,
                    linewidth=2,
                    label=f"DFT traj {i+1}",
                )

                # Predicted: dashed, SAME COLOR
                ax.plot(
                    idx[:n],
                    pred_ds[:n],
                    "--",
                    color=color,
                    alpha=0.9,
                    label=f"Pred traj {i+1}",
                )

            ax.axhline(0.0, color="red", linestyle="--", linewidth=1)

            if prefix.lower() == "bi11-3_samples":
                ax.axhline(
                    optimized_bi4 - optimized_energy,
                    color="green",
                    linestyle="--",
                    linewidth=1,
                    label="Optimized Bi4 Energy",
                )
                ax.axhline(
                    optimized_bi2 - optimized_energy,
                    color="orange",
                    linestyle="--",
                    linewidth=1,
                    label="Optimized Bi2 Energy",
                )

            ax.set_xlabel(f"Time step (every {interval})")
            ax.set_ylabel("Energy - Optimized (eV)")
            ax.set_title(f"Downsampled Predicted vs DFT Energies — Label {lbl}")
            ax.grid(True)
            ax.legend(ncol=2, fontsize=8)

            fig.tight_layout()
            fig.savefig(
                out_dir / f"downsampled_pred_vs_dft_label_{safe_label_str(lbl)}.png",
                dpi=180,
            )
            plt.close(fig)

    # -------- Residual histogram (energies): overall --------
    if residual_series:
        all_res = np.concatenate(residual_series)
        plt.figure(figsize=(6, 4))
        plt.hist(all_res, bins=30, alpha=0.85)
        plt.xlabel("Predicted − DFT (eV)")
        plt.ylabel("Count")
        plt.title("Residuals over downsampled frames (overall)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "residual_hist.png", dpi=180)
        plt.close()

    # -------- Residual histogram (energies): by label, overlay --------
    if residuals_by_label:
        labels_list = list(residuals_by_label.keys())
        conc = {lbl: np.concatenate(residuals_by_label[lbl])
                for lbl in labels_list if residuals_by_label[lbl]}
        if conc:
            all_vals = np.concatenate(list(conc.values()))
            bins = np.histogram_bin_edges(all_vals, bins=40)
            colors_map = choose_label_colors(labels_list)

            plt.figure(figsize=(7, 4.5))
            for lbl in labels_list:
                if lbl not in conc:
                    continue
                plt.hist(conc[lbl], bins=bins, histtype="step", linewidth=2.0,
                         color=colors_map[lbl], label=str(lbl), density=False)
            plt.xlabel("Predicted − DFT (eV)")
            plt.ylabel("Count")
            plt.title("Residuals by trajectory label")
            plt.grid(True, alpha=0.3)
            plt.legend(title="Traj type", frameon=False, fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / "residual_hist_by_label.png", dpi=180)
            plt.close()

        # -------- NEW: separate per-label energy residual histograms --------
        if has_label:
            for lbl in labels_list:
                if not residuals_by_label[lbl]:
                    continue
                vals = np.concatenate(residuals_by_label[lbl])
                plt.figure(figsize=(6, 4))
                plt.hist(vals, bins=35, alpha=0.85)
                plt.xlabel("Predicted − DFT (eV)")
                plt.ylabel("Count")
                plt.title(f"Residual histogram — Label {lbl}")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / f"residual_hist_label_{safe_label_str(lbl)}.png", dpi=180)
                plt.close()

    # -------- Energy scatter colored by LABEL (overlay) --------
    fig, ax = plt.subplots(figsize=(6, 6))
    if all_dft_scatter_by_label:
        all_dft_concat  = np.concatenate([np.concatenate(v) for v in all_dft_scatter_by_label.values() if v])
        all_pred_concat = np.concatenate([np.concatenate(v) for v in all_pred_scatter_by_label.values() if v])
        if all_dft_concat.size and all_pred_concat.size:
            min_val = float(min(all_dft_concat.min(), all_pred_concat.min()))
            max_val = float(max(all_dft_concat.max(), all_pred_concat.max()))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

        unique_labels_scatter = list(all_dft_scatter_by_label.keys())
        color_map = choose_label_colors(unique_labels_scatter)

        for lbl in unique_labels_scatter:
            dft_chunks  = all_dft_scatter_by_label[lbl]
            pred_chunks = all_pred_scatter_by_label[lbl]
            if not dft_chunks or not pred_chunks:
                continue
            dft_arr  = np.concatenate(dft_chunks)
            pred_arr = np.concatenate(pred_chunks)
            ax.scatter(dft_arr, pred_arr, s=12, alpha=0.6, color=color_map[lbl], label=str(lbl))

        ax.set_xlabel("DFT Energy - Optimized (eV)")
        ax.set_ylabel("Predicted Energy - Optimized (eV)")
        ax.set_title("Predicted vs DFT Energies (colored by traj label)")
        ax.grid(True)
        ax.legend(title="Traj type", frameon=False, fontsize=8, ncol=1)
        plt.tight_layout()
        plt.savefig(out_dir / "pred_vs_dft_scatter.png", dpi=180)
        plt.close()
    else:
        print("WARNING: No aligned energy data available for scatter plot.")

    # -------- NEW: Separate per-label energy scatter --------
    if has_label and all_dft_scatter_by_label:
        for lbl in all_dft_scatter_by_label.keys():
            d_chunks = all_dft_scatter_by_label[lbl]
            p_chunks = all_pred_scatter_by_label[lbl]
            if not d_chunks or not p_chunks:
                continue
            d = np.concatenate(d_chunks)
            p = np.concatenate(p_chunks)
            fig, ax = plt.subplots(figsize=(6, 6))
            min_val = float(min(d.min(), p.min()))
            max_val = float(max(d.max(), p.max()))
            ax.plot([min_val, max_val], [min_val, max_val], "r--")
            ax.scatter(d, p, s=12, alpha=0.6, color="C0")
            ax.set_xlabel("DFT Energy - Optimized (eV)")
            ax.set_ylabel("Predicted Energy - Optimized (eV)")
            ax.set_title(f"Predicted vs DFT Energies — Label {lbl}")
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"pred_vs_dft_scatter_label_{safe_label_str(lbl)}.png", dpi=180)
            plt.close()
    # -------- Energy Error along trajectory --------
    # -------- Color map for labels --------
    if has_label:
        unique_labels = sorted(set(label_trajs))
        label_color_map = choose_label_colors(unique_labels)
    else:
        label_color_map = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    for steps, mae, lbl in energy_mae_time_series:
        color = label_color_map.get(lbl, None)
        ax.plot(steps, mae, alpha=0.8, color=color, label=str(lbl))
    ax.set_xlabel(f"Time step (every {interval})")
    ax.set_ylabel("Energy Error |Pred − DFT| (eV)")
    ax.set_title("Energy Error along trajectory")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(title="Traj type", frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "energy_mae_along_traj.png", dpi=180)
    plt.close()

    # -------- Forces: scatter + residuals (label-aware) --------
    force_mae_time_series = []  # list of (time_idx, mae_series, label)
    metrics_force_rows = []

    if dft_force_lists:
        by_label_pred_forces = defaultdict(list)
        by_label_dft_forces  = defaultdict(list)

        print(f"[forces] #trajs pred={len(force_trajs)} dft={len(dft_force_lists)} idx={len(time_ds_all)}", flush=True)

        for i, (pred_force_tr, dft_force_tr) in enumerate(zip(force_trajs, dft_force_lists), start=1):
            idx = time_ds_all[i-1] if (i-1) < len(time_ds_all) else np.array([], dtype=int)
            if idx.size == 0:
                continue
            lbl = label_trajs[i-1] if has_label else "all"

            pred_ds_forces = [pred_force_tr[j] if j < len(pred_force_tr) else None for j in idx]
            n = min(len(pred_ds_forces), len(dft_force_tr))
            force_mae_series = []
            force_mae_steps  = []

            if n == 0:
                continue

            pred_frames, dft_frames = [], []
            for pf, df in zip(pred_ds_forces[:n], dft_force_tr[:n]):
                if df is not None and not isinstance(df, np.ndarray):
                    df = _coerce_force_frame(df)
                if pf is None or df is None:
                    continue
                pf_arr = pf if isinstance(pf, np.ndarray) else _coerce_force_frame(pf)
                df_arr = df if isinstance(df, np.ndarray) else _coerce_force_frame(df)
                if pf_arr is None or df_arr is None:
                    continue
                m = min(pf_arr.shape[0], df_arr.shape[0])
                if m == 0:
                    continue
                # predicted ≈ -DFT
                diff = pf_arr[:m] + df_arr[:m]

                # averaged MAE among xyz (vector norm)
                mae_f = np.mean(np.linalg.norm(diff, axis=1))
                pred_frames.append(pf_arr[:m, :])
                dft_frames.append(df_arr[:m, :])
                force_mae_series.append(mae_f)
                force_mae_steps.append(idx[len(force_mae_steps)])

            if force_mae_series:
                force_mae_time_series.append(
                    (np.array(force_mae_steps),
                    np.array(force_mae_series),
                    lbl)
                )
            if not pred_frames:
                continue

            pred_f = np.concatenate(pred_frames, axis=0)
            dft_f  = np.concatenate(dft_frames,  axis=0)
            n_force = pred_f.shape[0]
            if n_force == 0:
                continue

            # sign convention: predicted ≈ -DFT
            res_f = pred_f + dft_f          # shape (N_atoms_total, 3)

            # vector-norm metrics (PHYSICALLY IMPORTANT)
            res_norm = np.linalg.norm(res_f, axis=1)
            force_mae  = float(np.mean(res_norm))
            force_rmse = float(np.sqrt(np.mean(res_norm**2)))
            force_std  = float(np.std(res_norm, ddof=1)) if n_force > 1 else np.nan

            # component-wise diagnostics (optional but useful)
            bias_xyz = np.mean(res_f, axis=0)
            std_xyz  = np.std(res_f, axis=0, ddof=1) if n_force > 1 else np.full(3, np.nan)
            rmse_xyz = np.sqrt(np.mean(res_f**2, axis=0))
            metrics_force_rows.append({
                "traj": i,
                "label": lbl,
                "n_points": n_force,

                # vector metrics
                "force_mae_eV_per_A": force_mae,
                "force_rmse_eV_per_A": force_rmse,
                "force_std_eV_per_A": force_std,

                # component-wise (debugging)
                "bias_x_eV_per_A": bias_xyz[0],
                "bias_y_eV_per_A": bias_xyz[1],
                "bias_z_eV_per_A": bias_xyz[2],
                "std_x_eV_per_A": std_xyz[0],
                "std_y_eV_per_A": std_xyz[1],
                "std_z_eV_per_A": std_xyz[2],
                "rmse_x_eV_per_A": rmse_xyz[0],
                "rmse_y_eV_per_A": rmse_xyz[1],
                "rmse_z_eV_per_A": rmse_xyz[2],
            })



            mask = np.isfinite(pred_f).all(axis=1) & np.isfinite(dft_f).all(axis=1)
            if not np.any(mask):
                continue
            pred_f = pred_f[mask]
            dft_f  = dft_f[mask]

            # sign convention: compare predicted vs (-DFT)
            by_label_pred_forces[lbl].append(pred_f)
            by_label_dft_forces[lbl].append(-1.0 * dft_f)

        # Scatter by label (overlay, colored)
        # -------- Force Error along trajectory --------
        fig, ax = plt.subplots(figsize=(8, 5))

        for steps, mae, lbl in force_mae_time_series:
            color = label_color_map.get(lbl, None)
            ax.plot(steps, mae, alpha=0.8, color=color, label=str(lbl))

        ax.set_xlabel(f"Time step (every {interval})")
        ax.set_ylabel("Force Error (eV/Å)")
        ax.set_title("Force Error along trajectory")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(title="Traj type", frameon=False, fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / "force_mae_along_traj.png", dpi=180)
        plt.close()

        if by_label_pred_forces:
            fig, ax = plt.subplots(figsize=(6, 6))
            unique_labels_forces = list(by_label_pred_forces.keys())
            color_map = choose_label_colors(unique_labels_forces)

            all_d = []
            all_p = []
            for lbl in unique_labels_forces:
                if by_label_dft_forces[lbl] and by_label_pred_forces[lbl]:
                    all_d.append(np.vstack(by_label_dft_forces[lbl]).flatten())
                    all_p.append(np.vstack(by_label_pred_forces[lbl]).flatten())
            if all_d and all_p:
                all_d = np.concatenate(all_d); all_p = np.concatenate(all_p)
                min_val = float(min(all_d.min(), all_p.min()))
                max_val = float(max(all_d.max(), all_p.max()))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

            for lbl in unique_labels_forces:
                if not by_label_dft_forces[lbl] or not by_label_pred_forces[lbl]:
                    continue
                d = np.vstack(by_label_dft_forces[lbl]).flatten()
                p = np.vstack(by_label_pred_forces[lbl]).flatten()
                ax.scatter(d, p, s=6, alpha=0.35, color=color_map[lbl], label=str(lbl))

            ax.set_xlabel("DFT Force (eV/Å)")
            ax.set_ylabel("Predicted Force (eV/Å)")
            ax.set_title(f"Predicted vs DFT Forces (every {interval} steps)")
            ax.grid(True)
            ax.legend(title="Traj type", frameon=False, fontsize=8, ncol=1)
            plt.tight_layout()
            plt.savefig(out_dir / "pred_vs_dft_force_scatter.png", dpi=180)
            plt.close()

            # Residual histograms (components, norms) — overlay by label
            comps_by_label = {}
            norms_by_label = {}
            for lbl in unique_labels_forces:
                if by_label_pred_forces[lbl] and by_label_dft_forces[lbl]:
                    P = np.vstack(by_label_pred_forces[lbl])
                    D = np.vstack(by_label_dft_forces[lbl])
                    comps_by_label[lbl] = (P - D).flatten()
                    norms_by_label[lbl] = np.linalg.norm(P - D, axis=1)

            if comps_by_label:
                all_comps = np.concatenate(list(comps_by_label.values()))
                bins = np.histogram_bin_edges(all_comps, bins=60)
                colors_map = choose_label_colors(list(comps_by_label.keys()))

                plt.figure(figsize=(7, 4.5))
                for lbl, arr in comps_by_label.items():
                    plt.hist(arr, bins=bins, histtype="step", linewidth=2.0,
                             color=colors_map[lbl], label=str(lbl))
                plt.xlabel("Predicted − DFT Force (eV/Å) [components]")
                plt.ylabel("Count")
                plt.title(f"Force Residuals (components) — every {interval} steps")
                plt.grid(True, alpha=0.3)
                plt.legend(title="Traj type", frameon=False, fontsize=8)
                plt.tight_layout()
                plt.savefig(out_dir / "force_residual_components_by_label.png", dpi=180)
                plt.close()

            if norms_by_label:
                all_norms = np.concatenate(list(norms_by_label.values()))
                bins = np.histogram_bin_edges(all_norms, bins=60)
                colors_map = choose_label_colors(list(norms_by_label.keys()))

                plt.figure(figsize=(7, 4.5))
                for lbl, arr in norms_by_label.items():
                    plt.hist(arr, bins=bins, histtype="step", linewidth=2.0,
                             color=colors_map[lbl], label=str(lbl))
                plt.xlabel("‖Predicted − DFT‖ (eV/Å) [vector norm]")
                plt.ylabel("Count")
                plt.title(f"Force Residual Norms — every {interval} steps")
                plt.grid(True, alpha=0.3)
                plt.legend(title="Traj type", frameon=False, fontsize=8)
                plt.tight_layout()
                plt.savefig(out_dir / "force_residual_norms_by_label.png", dpi=180)
                plt.close()

            # -------- NEW: separate per-label force scatter & histograms --------
            if has_label:
                for lbl in unique_labels_forces:
                    # per-label scatter
                    if not by_label_dft_forces[lbl] or not by_label_pred_forces[lbl]:
                        continue
                    D = np.vstack(by_label_dft_forces[lbl]).flatten()
                    P = np.vstack(by_label_pred_forces[lbl]).flatten()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    mn = float(min(D.min(), P.min()))
                    mx = float(max(D.max(), P.max()))
                    ax.plot([mn, mx], [mn, mx], "r--")
                    ax.scatter(D, P, s=6, alpha=0.35, color="C0")
                    ax.set_xlabel("DFT Force (eV/Å)")
                    ax.set_ylabel("Predicted Force (eV/Å)")
                    ax.set_title(f"Predicted vs DFT Forces — Label {lbl}")
                    ax.grid(True)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"pred_vs_dft_force_scatter_label_{safe_label_str(lbl)}.png", dpi=180)
                    plt.close()

                # per-label residual components
                for lbl, arr in comps_by_label.items():
                    plt.figure(figsize=(6, 4))
                    plt.hist(arr, bins=50, alpha=0.85, color="C0")
                    plt.xlabel("Predicted − DFT Force (eV/Å) [components]")
                    plt.ylabel("Count")
                    plt.title(f"Force Residual Components — Label {lbl}")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"force_residual_components_label_{safe_label_str(lbl)}.png", dpi=180)
                    plt.close()

                # per-label residual norms
                for lbl, arr in norms_by_label.items():
                    plt.figure(figsize=(6, 4))
                    plt.hist(arr, bins=50, alpha=0.85, color="C0")
                    plt.xlabel("‖Predicted − DFT‖ (eV/Å)")
                    plt.ylabel("Count")
                    plt.title(f"Force Residual Norms — Label {lbl}")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"force_residual_norms_label_{safe_label_str(lbl)}.png", dpi=180)
                    plt.close()

            metrics_force = pd.DataFrame(metrics_force_rows)
            metrics_force_path = out_dir / "pred_vs_dft_metrics_force.csv"
            metrics_force.to_csv(metrics_force_path, index=False)
        else:
            print("WARNING: No aligned force data available to plot (check lengths/indices).")

    # -------- Final outputs --------
    print("Saved:")
    print(f" - metrics: {metrics_path}")
    print(f" - energy overlay: {out_dir / 'downsampled_pred_vs_dft.png'}")
    if (out_dir / "downsampled_pred_vs_dft_label_all.png").exists():
        print(f" - per-label energy overlays")
    if (out_dir / "residual_hist.png").exists():
        print(f" - energy residual hist (overall): {out_dir / 'residual_hist.png'}")
    if (out_dir / "residual_hist_by_label.png").exists():
        print(f" - energy residual hist (by label overlay): {out_dir / 'residual_hist_by_label.png'}")
    if (out_dir / "pred_vs_dft_scatter.png").exists():
        print(f" - energy scatter overlay: {out_dir / 'pred_vs_dft_scatter.png'}")
    if (out_dir / 'pred_vs_dft_force_scatter.png').exists():
        print(f" - force scatter overlay: {out_dir / 'pred_vs_dft_force_scatter.png'}")
    if (out_dir / "force_residual_components_by_label.png").exists():
        print(f" - force residual comps overlay (by label): {out_dir / 'force_residual_components_by_label.png'}")
    if (out_dir / "force_residual_norms_by_label.png").exists():
        print(f" - force residual norms overlay (by label): {out_dir / 'force_residual_norms_by_label.png'}")
    # per-label files will be many; not printing them all


if __name__ == "__main__":
    main()
