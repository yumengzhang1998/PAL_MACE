# animate_energy_dist_simple.py — minimal version per spec
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

def assign_iter(df: pd.DataFrame, init_n: int, inc_n: int) -> pd.Series:
    """Return an 'iter' Series: first init_n rows -> 0; then every inc_n rows -> 1,2,3,..."""
    if "iter" in df.columns:
        # If an iter column already exists, respect it (coerce to int)
        it = pd.to_numeric(df["iter"], errors="coerce")
        if it.notna().any():
            return it.fillna(0).astype(int)
    n = len(df)
    idx = np.arange(n, dtype=int)
    it = np.zeros(n, dtype=int)
    mask_after_init = idx >= int(init_n)
    it[mask_after_init] = ((idx[mask_after_init] - int(init_n)) // int(inc_n)) + 1
    return pd.Series(it, index=df.index, dtype=int)

def make_bins(all_values: np.ndarray, n_bins: int = 100):
    if all_values.size == 0:
        return np.linspace(0.0, 1.0, n_bins+1)
    lo, hi = float(np.min(all_values)), float(np.max(all_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, n_bins+1)

def hist_density(x, bins):
    if x.size == 0:
        return np.zeros_like(bins[:-1])
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(bins[:-1])
    counts, _ = np.histogram(x, bins=bins)
    widths = np.diff(bins)
    area = (counts * widths).sum()
    return counts / (area / widths) if area > 0 else counts

def animate_dist(df: pd.DataFrame, bins, out_path: str, title: str, fps=2, dpi=120, bitrate=1800):
    """Animate cumulative histograms over increasing iter thresholds (start at iter=1)."""
    if df.empty:
        raise ValueError("Empty DataFrame for animation.")
    max_iter = int(df["iter"].max())
    # frames: 1..max_iter (inclusive)
    frames_list = list(range(1, max_iter + 1))

    # Precompute cumulative arrays for speed
    energies_by_iter = {}
    for k in frames_list:
        energies_by_iter[k] = df.loc[df["iter"] <= k, "energy"].to_numpy(float)

    densities = [hist_density(energies_by_iter[k], bins) for k in frames_list]
    # Optional quantiles
    q25 = [np.quantile(energies_by_iter[k], 0.25) if energies_by_iter[k].size else np.nan for k in frames_list]
    q50 = [np.quantile(energies_by_iter[k], 0.50) if energies_by_iter[k].size else np.nan for k in frames_list]
    q75 = [np.quantile(energies_by_iter[k], 0.75) if energies_by_iter[k].size else np.nan for k in frames_list]
    sizes = [energies_by_iter[k].size for k in frames_list]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(bins[0], bins[-1])
    ymax = max((d.max() if len(d) else 0.0) for d in densities) * 1.10
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax)
    ax.set_title(title)
    ax.set_xlabel("Energy"); ax.set_ylabel("Density")

    bars = ax.bar(bins[:-1], densities[0], width=np.diff(bins), align="edge")
    lq25 = ax.axvline(q25[0], linestyle=":", linewidth=1.0) if np.isfinite(q25[0]) else None
    lq50 = ax.axvline(q50[0], linewidth=1.2) if np.isfinite(q50[0]) else None
    lq75 = ax.axvline(q75[0], linestyle=":", linewidth=1.0) if np.isfinite(q75[0]) else None
    txt  = ax.text(0.02, 0.95, f"iter<=1 | n={sizes[0]}", transform=ax.transAxes, va="top")

    def update(i):
        d = densities[i]
        for rect, h in zip(bars, d):
            rect.set_height(h)
        if np.isfinite(q25[i]) and lq25 is not None: lq25.set_xdata([q25[i], q25[i]])
        if np.isfinite(q50[i]) and lq50 is not None: lq50.set_xdata([q50[i], q50[i]])
        if np.isfinite(q75[i]) and lq75 is not None: lq75.set_xdata([q75[i], q75[i]])
        txt.set_text(f"iter<={frames_list[i]} | n={sizes[i]}")
        return list(bars) + [o for o in (lq25, lq50, lq75) if o is not None] + [txt]

    anim = animation.FuncAnimation(fig, update, frames=len(frames_list), interval=1000//max(1,fps), blit=False)

    # Save, prefer MP4 with ffmpeg, fallback to GIF
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        try:
            Writer = animation.FFMpegWriter
            anim.save(out_path, writer=Writer(fps=fps, bitrate=bitrate), dpi=dpi)
        except Exception:
            out_gif = out_path.replace(".mp4", ".gif")
            Writer = animation.PillowWriter
            anim.save(out_gif, writer=Writer(fps=fps), dpi=dpi)
            print(f"FFmpeg not available; saved GIF instead: {out_gif}")
    else:
        Writer = animation.PillowWriter
        anim.save(out_path, writer=Writer(fps=fps), dpi=dpi)
    print("Saved:", out_path)

def main():
    ap = argparse.ArgumentParser(description="Animate energy distribution change by iterations (simple).")
    ap.add_argument("--prefix", type=str, required=True, help="Folder prefix, e.g., 'bi4-6'")
    ap.add_argument("--model_number", type=str, required=True, help="Model number, e.g., '56'")
    ap.add_argument("--train_inc", type=int, default=45)
    ap.add_argument("--val_inc", type=int, default=5)
    ap.add_argument("--train_init", type=int, default=450)
    ap.add_argument("--val_init", type=int, default=90)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--bitrate", type=int, default=1800)
    ap.add_argument("--cluster_name", type=str, default="", help=" Cluster name for title adjustments.")
    args = ap.parse_args()

    csv_path = os.path.join(args.prefix, f"{args.model_number}_added_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "type" not in df.columns:
        # try to guess the type column name
        tc = next((c for c in df.columns if c.lower().strip() == "type"), None)
        if tc is None:
            raise ValueError(f"'type' column not found in {csv_path}. Columns: {list(df.columns)}")
        df = df.rename(columns={tc: "type"})
    # normalize type labels
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # energy column
    if "energy" not in df.columns:
        ec = next((c for c in df.columns if "energy" in c.lower()), None)
        if ec is None:
            raise ValueError(f"No energy column in {csv_path}. Columns: {list(df.columns)}")
        df = df.rename(columns={ec: "energy"})
    # ---- NEW: shift energies so that the minimal energy is 0 ----
    energy_array = df["energy"].to_numpy(float)
    finite_mask = np.isfinite(energy_array)
    if finite_mask.any():
        e_min = float(energy_array[finite_mask].min())
        df["energy"] = df["energy"] - e_min
        print(f"Shifted energies by {e_min:.6f} so global minimum is 0.")
    else:
        print("Warning: no finite energies found; skipping energy shift.")
    # ------------------------------------------------------------

    # split into train/val
    train = df[df["type"] == "train"].reset_index(drop=True).copy()
    val   = df[df["type"] == "val"].reset_index(drop=True).copy()

    # assign iterations if missing
    train["iter"] = assign_iter(train, args.train_init, args.train_inc)
    val["iter"]   = assign_iter(val,   args.val_init,   args.val_inc)

    # Build global bins per split for stable axes
    train_bins = make_bins(train["energy"].to_numpy(float), n_bins=args.bins)
    val_bins   = make_bins(val["energy"].to_numpy(float),   n_bins=args.bins)

    # Output paths (avoid overwriting by using split suffixes)
    out_train = os.path.join(args.prefix, f"energy_dist_change_{args.model_number}_train.mp4")
    out_val   = os.path.join(args.prefix, f"energy_dist_change_{args.model_number}_val.mp4")

    # Animate starting from iter=1 cumulatively
    animate_dist(train, train_bins, out_train, title=f"{args.cluster_name} Train energy distribution — model {args.model_number}",
                 fps=args.fps, dpi=args.dpi, bitrate=args.bitrate)
    animate_dist(val,   val_bins,   out_val,   title=f"{args.cluster_name} Val energy distribution — model {args.model_number}",
                 fps=args.fps, dpi=args.dpi, bitrate=args.bitrate)

if __name__ == "__main__":
    main()