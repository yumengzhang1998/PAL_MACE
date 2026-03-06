#!/usr/bin/env python3
# make_oob_std_distribution.py
import argparse, os, sys, json, glob, ast, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ---- Robust imports (project root) ----
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for p in [THIS_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import evaluate
from data import full_data_list

# Optional plotting fallback
try:
    from plot import plot_distribution as _plot_distribution
    HAVE_PROJECT_PLOT = True
except Exception:
    HAVE_PROJECT_PLOT = False
    import matplotlib.pyplot as plt

def _plot_hist(values, out_dir, title, fname_prefix):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=float)
    if HAVE_PROJECT_PLOT:
        _plot_distribution(values.tolist(), str(out_dir), title)
    else:
        plt.figure()
        plt.hist(values, bins=50)
        plt.title(title); plt.xlabel("Value"); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(out_dir / f"{fname_prefix}_hist.png", dpi=180)
        plt.close()
    counts, bins = np.histogram(values, bins=50)
    pd.DataFrame({"bin_left": bins[:-1], "bin_right": bins[1:], "count": counts})\
      .to_csv(out_dir / f"{fname_prefix}_hist.csv", index=False)

# -------- Helpers to parse your CSV format --------
def _literal_or_str(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x

_CHARGE_RE = re.compile(r"(-?\d+)")
def parse_charge(x):
    """Accepts numbers, lists, 'tensor(-2, dtype=...)', or None -> returns scalar or None."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        m = _CHARGE_RE.search(x)
        return float(m.group(1)) if m else None
    if isinstance(x, (list, tuple)) and len(x) == 1:
        try: return float(x[0])
        except Exception: return None
    try:
        return float(x)
    except Exception:
        return None

def config_id_from_coords(coords, charge=None, ndp=6):
    """Stable ID from rounded coords (+ optional charge). Ignores atoms to avoid symbol/number mismatches."""
    coords = _literal_or_str(coords)
    coords_r = [[round(float(c), ndp) for c in row] for row in coords]
    payload = {"coords": coords_r}
    if charge is not None:
        payload["charge"] = float(charge)
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ids_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        coords = _literal_or_str(r.get("coordinates"))
        chg = parse_charge(_literal_or_str(r.get("charge")))
        out.append(config_id_from_coords(coords, chg))
    return set(out)
def _normalize_dtype(s: str) -> str:
    s = (s or "").strip().lower()
    aliases = {
        "float32": "float32", "fp32": "float32", "float": "float32", "single": "float32",
        "float64": "float64", "fp64": "float64", "double": "float64"
    }
    if s not in aliases:
        raise ValueError(f"Unsupported default_dtype '{s}'. Use float32 or float64.")
    return aliases[s]
def id_source_map_from_csv(csv_path):
    """Build {config_id -> source} using the same ID logic as the script."""
    df = pd.read_csv(csv_path)
    id2src = {}
    has_source = "source" in df.columns
    for _, r in df.iterrows():
        coords = _literal_or_str(r.get("coordinates"))
        chg = parse_charge(_literal_or_str(r.get("charge")))
        cid = config_id_from_coords(coords, chg)
        src = r.get("source") if has_source else None
        id2src[cid] = src
    return id2src

# ===== Overlay plots (ALL points) and (filtered by min_oob) =====
import matplotlib.pyplot as plt

def overlay_hist(by_src, title, out_png, xlabel, out_dir):
    # consistent bins across clusters
    arrays = [v for v in by_src.values() if v.size]
    if not arrays:
        print(f"[overlay_hist] No data for {title}"); return
    all_vals = np.concatenate(arrays, axis=0)
    bins = np.linspace(all_vals.min(), all_vals.max(), 60)

    plt.figure()
    for src, vals in by_src.items():
        label = f"{src} (n={vals.size})"
        if vals.size == 0:  # still show legend entry with empty data
            continue
        plt.hist(vals, bins=bins, histtype="step", density=True, linewidth=1.8, label=label)
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir + out_png, dpi=200); plt.close()

def overlay_ecdf(by_src, title, out_png, xlabel, out_dir):
    plt.figure()
    any_data = False
    for src, vals in by_src.items():
        label = f"{src} (n={vals.size})"
        if vals.size == 0:
            continue
        any_data = True
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=label)
    if not any_data:
        print(f"[overlay_ecdf] No data for {title}"); plt.close(); return
    plt.xlabel(xlabel); plt.ylabel("ECDF"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir + out_png, dpi=200); plt.close()
def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
# ---- Save per-source JSON summaries (filtered & ALL) ----
from pathlib import Path

# ---- Helpers to aggregate by source and save JSON summaries ----
def _summ_stats(vals: np.ndarray):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return {
            "n": 0, "mean": None, "median": None, "min": None, "max": None,
            "q50": None, "q75": None, "q90": None, "q95": None, "q97_5": None, "q99": None
        }
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q50": float(np.percentile(vals, 50)),
        "q75": float(np.percentile(vals, 75)),
        "q90": float(np.percentile(vals, 90)),
        "q95": float(np.percentile(vals, 95)),
        "q97_5": float(np.percentile(vals, 97.5)),
        "q99": float(np.percentile(vals, 99)),
    }

def by_src_from_combined(combined_df: pd.DataFrame, value_col: str, min_oob: int, oob_col: str) -> dict:
    """Return {source -> np.array(values)} applying min_oob filter on oob_col."""
    df = combined_df.copy()
    df["source"] = df["source"].fillna("unknown").astype(str).str.strip()
    mask = (df[oob_col] >= int(min_oob))
    vals = df.loc[mask, ["source", value_col]].dropna()
    out = {}
    for src, sub in vals.groupby("source"):
        out[src] = sub[value_col].to_numpy()
    return out

def save_per_source_json(by_src: dict, out_dir: str, metric_name: str, min_oob_used: int, suffix: str = ""):
    """
    by_src: dict {source -> 1D array}
    metric_name: column/metric label, e.g. 'energy_std_oob' or 'force_std_max_atomnorm_oob'
    min_oob_used: the min_oob applied (0 means ALL points)
    suffix: extra tag for filenames, e.g. '_ALL'
    """
    base_dir = Path(out_dir) / "by_source"
    base_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src, arr in by_src.items():
        stats = _summ_stats(arr)
        stats["min_oob_required"] = int(min_oob_used)
        sdir = base_dir / _sanitize(src)
        sdir.mkdir(parents=True, exist_ok=True)
        with open(sdir / f"{metric_name}{suffix}_summary.json", "w") as f:
            json.dump(stats, f, indent=2)
        rows.append({"source": src, **stats})

    pd.DataFrame(rows).to_csv(base_dir / f"{metric_name}{suffix}_cluster_summaries.csv", index=False)

# -------------- Main --------------
def main():
    ap = argparse.ArgumentParser(description="Compute OOB std distribution (energy + force RMS) for AL threshold.")
    ap.add_argument("--logs_dir", default="./logs", help="Directory containing sample_*/ with models + train.csv")
    ap.add_argument("--default_dtype",default="float64",help="One of: float32, float64 (also accepts: fp32, fp64, float, double).",)
    ap.add_argument("--prefix", required=True, help="Model name, e.g. bi0 (expects sample_*/bi0.model)")
    ap.add_argument("--dataset", help="CSV to evaluate (default: ../raw/<prefix>_parsed.csv)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--min_oob", type=int, default=2, help="Min OOB models required to include a point in summary plots")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    dd = _normalize_dtype(args.default_dtype)


    dataset_csv = args.dataset or str(PROJECT_ROOT / "raw" / f"{args.prefix}_parsed.csv")
    out_dir = args.out_dir or os.path.join(args.logs_dir, f"{args.prefix}_oob_uncertainty")
    os.makedirs(out_dir, exist_ok=True)

    # Ensemble members
    model_paths = sorted(glob.glob(os.path.join(args.logs_dir, "sample_*", f"{args.prefix}.model")))
    if len(model_paths) < 2:
        raise RuntimeError(f"Need >=2 models. Found {len(model_paths)} with {args.logs_dir}/sample_*/{args.prefix}.model")
    print("Models:")
    for p in model_paths: print("  -", p)

    # For each model, read its train IDs (from saved CSV)
    train_id_sets = []
    for mp in model_paths:
        sdir = Path(mp).parent
        tcsv = sdir / "train.csv"
        if not tcsv.exists():
            raise FileNotFoundError(f"Missing {tcsv}")
        train_id_sets.append(ids_from_csv(tcsv))

    # Evaluation set = a fixed parsed dataset
    ds = full_data_list(raw_data_path=dataset_csv, gradeint_to_force=False)
    eval_set = ds.data_list
    n_cfg = len(eval_set)
    print(f"Eval set: {n_cfg} configs from {dataset_csv}")

    # Build IDs for eval_set using coords (+ charge) to match train.csv IDs
    eval_ids = []
    for d in eval_set:
        coords = d.pos.detach().cpu().numpy().tolist()
        chg = getattr(d, "charge", None)
        if torch.is_tensor(chg):
            chg_np = chg.detach().cpu().numpy().tolist()
            # normalize: scalar or length-1 -> scalar
            if isinstance(chg_np, list) and len(chg_np) == 1:
                chg = chg_np[0]
            else:
                chg = chg_np
        eval_ids.append(config_id_from_coords(coords, parse_charge(chg)))
    
    id2src = id_source_map_from_csv(dataset_csv)
    sources = [id2src.get(cid, None) for cid in eval_ids]
    src_arr = np.array([s if (s is not None and str(s) != "nan") else "unknown" for s in sources], dtype=object)

    # Evaluate all models
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    energy_preds = []            # [n_models, n_cfg]
    forces_preds_per_model = []  # list over models; each is list over configs of ndarray [n_atoms_i, 3]
    for mp in model_paths:
        print(f"Evaluating {mp}")
        model = torch.load(mp, map_location=device)
        energy, forces, _, _ = evaluate(
            model=model,
            eval_dataset=eval_set,
            batch_size=args.batch_size,
            default_dtype=dd,
            device=device,
            compute_stress=False,
            return_contributions=False,
        )
        energy_preds.append(np.asarray(energy, dtype=np.float64).reshape(-1))
        forces_preds_per_model.append(forces)

    E = np.stack(energy_preds, axis=0)  # [n_models, n_cfg]
    n_models = E.shape[0]

    # OOB mask: True if model m did NOT train on config i
    mask_oob = np.zeros((n_models, n_cfg), dtype=bool)
    for m in range(n_models):
        train_ids_m = train_id_sets[m]
        mask_oob[m, :] = np.array([eid not in train_ids_m for eid in eval_ids], dtype=bool)

    # Energy OOB std
    energy_std_oob = np.full(n_cfg, np.nan, dtype=np.float64)
    oob_counts_E = np.zeros(n_cfg, dtype=int)
    for i in range(n_cfg):
        msk = mask_oob[:, i]
        k = int(msk.sum())
        oob_counts_E[i] = k
        if k == 0: 
            continue
        energy_std_oob[i] = float(E[msk, i].std(ddof=1)) if k > 1 else 0.0

    # Force OOB std RMS
    force_std_rms_oob = np.full(n_cfg, np.nan, dtype=np.float64)
    force_std_max_coord_oob = np.full(n_cfg, np.nan, dtype=np.float64)     # NEW: max over all coords
    force_std_max_atomnorm_oob = np.full(n_cfg, np.nan, dtype=np.float64)  # NEW: max over atoms of vector-norm
    force_std_p95_atomnorm_oob = np.full(n_cfg, np.nan, dtype=np.float64)  # NEW: 95th pct over atoms of vector-norm
    oob_counts_F = np.zeros(n_cfg, dtype=int)
    for i in range(n_cfg):
        idx = np.where(mask_oob[:, i])[0]
        k = len(idx)
        oob_counts_F[i] = k
        if k == 0: 
            continue
        Fi = np.stack([forces_preds_per_model[m][i] for m in idx], axis=0)

        # std across models per atom/component -> [n_atoms_i, 3]
        std_i = Fi.std(axis=0, ddof=1) if k > 1 else np.zeros_like(Fi[0])

        # 1) RMS over atoms/components (what you had)
        force_std_rms_oob[i] = float(np.sqrt(np.mean(std_i**2)))

        # 2) MAX over all coordinates (component-wise)
        force_std_max_coord_oob[i] = float(np.max(std_i))

        # 3) MAX over atoms of the vector-norm of std (rotation-invariant)
        atom_norm = np.linalg.norm(std_i, axis=-1)  # [n_atoms_i]
        force_std_max_atomnorm_oob[i] = float(atom_norm.max())

        # 4) 95th percentile over atoms of the vector-norm (robust)
        force_std_p95_atomnorm_oob[i] = float(np.percentile(atom_norm, 95))

    # Save per-config outputs
    # (true energy optional)
    try:
        true_E = torch.cat([d.y for d in eval_set], dim=0).cpu().numpy().astype(np.float64).reshape(-1)
    except Exception:
        true_E = np.full(n_cfg, np.nan)

    pd.DataFrame({
        "idx": np.arange(n_cfg, dtype=int),
        "config_id": eval_ids,
        "true_energy": true_E,
        "oob_model_count": oob_counts_E,
        "energy_std_oob": energy_std_oob,
        "source": src_arr,                   # <-- add this line
    }).to_csv(Path(out_dir) / "oob_energy_uncertainty.csv", index=False)

    pd.DataFrame({
        "idx": np.arange(n_cfg, dtype=int),
        "config_id": eval_ids,
        "oob_model_count": oob_counts_F,
        "force_std_rms_oob": force_std_rms_oob,
        "force_std_max_coord_oob": force_std_max_coord_oob,               # NEW
        "force_std_max_atomnorm_oob": force_std_max_atomnorm_oob,         # NEW
        "force_std_p95_atomnorm_oob": force_std_p95_atomnorm_oob,         # NEW
        "source": src_arr,                   # <-- add this line
    }).to_csv(Path(out_dir) / "oob_force_uncertainty.csv", index=False)

    # Summaries / histograms on points with at least min_oob contributors
    def summarize_plot(vals, counts, name):
        vals = np.asarray(vals, dtype=float)
        mask = np.isfinite(vals) & (counts >= args.min_oob)
        used = vals[mask]
        quantiles = [50, 75, 90, 95, 97.5, 99]
        summary = {
            "n_models": int(n_models),
            "n_points_used": int(mask.sum()),
            "min_oob_required": int(args.min_oob),
            "mean": float(np.mean(used)) if used.size else float("nan"),
            "median": float(np.median(used)) if used.size else float("nan"),
            "max": float(np.max(used)) if used.size else float("nan"),
        }
        for q in quantiles:
            if used.size:
                summary[f"q{q}"] = float(np.percentile(used, q))
        with open(Path(out_dir) / f"{name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if used.size:
            _plot_hist(used, out_dir, f"OOB {name} (min_oob={args.min_oob})", f"oob_{name}")
        print(f"{name} summary:", summary)

    summarize_plot(energy_std_oob, oob_counts_E, "energy_std")
    summarize_plot(force_std_rms_oob, oob_counts_F, "force_std_rms")
    summarize_plot(force_std_max_coord_oob, oob_counts_F, "force_std_max_coord")               # NEW
    summarize_plot(force_std_max_atomnorm_oob, oob_counts_F, "force_std_max_atomnorm")         # NEW
    summarize_plot(force_std_p95_atomnorm_oob, oob_counts_F, "force_std_p95_atomnorm")         # NEW


    print(f"Outputs -> {out_dir}")
    print("Use quantiles in *_summary.json (e.g., q85–q95 of force_std_rms) as your certainty threshold.")
    # ---- Build a combined DataFrame used by per-cluster analysis ----
    combined = pd.DataFrame({
        "idx": np.arange(n_cfg, dtype=int),
        "config_id": eval_ids,
        "true_energy": true_E,
        "source": src_arr,
        "oob_models_energy": oob_counts_E,
        "oob_models_force": oob_counts_F,
        "energy_std_oob": energy_std_oob,
        "force_std_rms_oob": force_std_rms_oob,
        "force_std_max_coord_oob": force_std_max_coord_oob,
        "force_std_max_atomnorm_oob": force_std_max_atomnorm_oob,
        "force_std_p95_atomnorm_oob": force_std_p95_atomnorm_oob,
    })
    # ---- Build per-source dicts (filtered by min_oob) ----
    energy_by_src         = by_src_from_combined(combined, "energy_std_oob",                args.min_oob, "oob_models_energy")
    force_rms_by_src      = by_src_from_combined(combined, "force_std_rms_oob",            args.min_oob, "oob_models_force")
    force_maxcoord_by_src = by_src_from_combined(combined, "force_std_max_coord_oob",      args.min_oob, "oob_models_force")
    force_maxatom_by_src  = by_src_from_combined(combined, "force_std_max_atomnorm_oob",   args.min_oob, "oob_models_force")
    force_p95atom_by_src  = by_src_from_combined(combined, "force_std_p95_atomnorm_oob",   args.min_oob, "oob_models_force")

    # ---- Save per-source JSONs (filtered) ----
    save_per_source_json(energy_by_src,         out_dir, "energy_std_oob",              args.min_oob)
    save_per_source_json(force_rms_by_src,      out_dir, "force_std_rms_oob",           args.min_oob)
    save_per_source_json(force_maxcoord_by_src, out_dir, "force_std_max_coord_oob",     args.min_oob)
    save_per_source_json(force_maxatom_by_src,  out_dir, "force_std_max_atomnorm_oob",  args.min_oob)
    save_per_source_json(force_p95atom_by_src,  out_dir, "force_std_p95_atomnorm_oob",  args.min_oob)

    # ---- Also save ALL-points versions (ignores min_oob) ----
    energy_by_src_ALL         = by_src_from_combined(combined, "energy_std_oob",                0, "oob_models_energy")
    force_rms_by_src_ALL      = by_src_from_combined(combined, "force_std_rms_oob",            0, "oob_models_force")
    force_maxcoord_by_src_ALL = by_src_from_combined(combined, "force_std_max_coord_oob",      0, "oob_models_force")
    force_maxatom_by_src_ALL  = by_src_from_combined(combined, "force_std_max_atomnorm_oob",   0, "oob_models_force")
    force_p95atom_by_src_ALL  = by_src_from_combined(combined, "force_std_p95_atomnorm_oob",   0, "oob_models_force")

    save_per_source_json(energy_by_src_ALL,         out_dir, "energy_std_oob",              0, suffix="_ALL")
    save_per_source_json(force_rms_by_src_ALL,      out_dir, "force_std_rms_oob",           0, suffix="_ALL")
    save_per_source_json(force_maxcoord_by_src_ALL, out_dir, "force_std_max_coord_oob",     0, suffix="_ALL")
    save_per_source_json(force_maxatom_by_src_ALL,  out_dir, "force_std_max_atomnorm_oob",  0, suffix="_ALL")
    save_per_source_json(force_p95atom_by_src_ALL,  out_dir, "force_std_p95_atomnorm_oob",  0, suffix="_ALL")
    # --- Build overlay arrays straight from the combined table ---
    def overlay_arrays_from_combined(combined_df, min_oob, which: str):
        """
        which: 'energy' or 'force'
        returns: dict {source -> np.ndarray of values}
        """
        by_src = {}
        # normalize/strip source labels
        sources = combined_df["source"].fillna("unknown").astype(str).str.strip()
        df = combined_df.copy()
        df["source"] = sources

        if which == "energy":
            mask = (df["oob_models_energy"] >= min_oob)
            vals = df.loc[mask, ["source", "energy_std_oob"]].dropna()
            for src, sub in vals.groupby("source"):
                by_src[src] = sub["energy_std_oob"].to_numpy()
        elif which == "force":
            mask = (df["oob_models_force"] >= min_oob)
            vals = df.loc[mask, ["source", "force_std_rms_oob"]].dropna()
            for src, sub in vals.groupby("source"):
                by_src[src] = sub["force_std_rms_oob"].to_numpy()
        else:
            raise ValueError("which must be 'energy' or 'force'")
        return by_src

    # overlay dicts that will drive the plots (respecting min_oob)
    energy_by_src = overlay_arrays_from_combined(combined, args.min_oob, "energy")
    force_by_src  = overlay_arrays_from_combined(combined, args.min_oob, "force")

    # also "ALL points" (ignoring min_oob) for sanity check
    energy_by_src_all = overlay_arrays_from_combined(combined, 0, "energy")
    force_by_src_all  = overlay_arrays_from_combined(combined, 0, "force")
    energy_by_src_all = overlay_arrays_from_combined(combined, 0, "energy")
    force_by_src_all  = overlay_arrays_from_combined(combined, 0, "force")
    # --- ALL points overlays (ignores min_oob) ---
    overlay_hist(energy_by_src_all,
                title="OOB Energy std overlay — ALL points",
                out_png="overlay_energy_std_hist_ALL.png",
                xlabel="Energy std (OOB)",
                out_dir=out_dir)
    overlay_ecdf(energy_by_src_all,
                title="OOB Energy std ECDF — ALL points",
                out_png="overlay_energy_std_ecdf_ALL.png",
                xlabel="Energy std (OOB)",
                out_dir=out_dir)
    overlay_hist(force_by_src_all,
                title="OOB Force std RMS overlay — ALL points",
                out_png="overlay_force_std_rms_hist_ALL.png",
                xlabel="Force std RMS (OOB)",
                out_dir=out_dir)
    overlay_ecdf(force_by_src_all,
                title="OOB Force std RMS ECDF — ALL points",
                out_png="overlay_force_std_rms_ecdf_ALL.png",
                xlabel="Force std RMS (OOB)",
                out_dir=out_dir)

    # --- Filtered overlays (respect min_oob) ---
    overlay_hist(energy_by_src,
                title=f"OOB Energy std overlay (min_oob={args.min_oob})",
                out_png=f"overlay_energy_std_hist_minOOB{args.min_oob}.png",
                xlabel="Energy std (OOB)",
                out_dir=out_dir)
    overlay_ecdf(energy_by_src,
                title=f"OOB Energy std ECDF (min_oob={args.min_oob})",
                out_png=f"overlay_energy_std_ecdf_minOOB{args.min_oob}.png",
                xlabel="Energy std (OOB)",
                out_dir=out_dir)
    overlay_hist(force_by_src,
                title=f"OOB Force std RMS overlay (min_oob={args.min_oob})",
                out_png=f"overlay_force_std_rms_hist_minOOB{args.min_oob}.png",
                xlabel="Force std RMS (OOB)",
                out_dir=out_dir)
    overlay_ecdf(force_by_src,
                title=f"OOB Force std RMS ECDF (min_oob={args.min_oob})",
                out_png=f"overlay_force_std_rms_ecdf_minOOB{args.min_oob}.png",
                xlabel="Force std RMS (OOB)",
                out_dir=out_dir)

if __name__ == "__main__":
    main()
