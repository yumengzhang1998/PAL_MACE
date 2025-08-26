#!/usr/bin/env python3
# max_dist_from_csv.py
import argparse
import ast
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

def max_atom_distance_scipy(coords):
    """Return (max_distance, (i, j)) for a list/array of 3D coordinates."""
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if len(X) < 2:
        return 0.0, (0, 0)

    d = pdist(X, metric="euclidean")  # condensed vector of distances
    k = int(np.argmax(d))             # index in condensed form

    # Map condensed index k -> (i, j) with i < j
    # Number of pairs in rows before row j is j*(j-1)/2
    j = int(np.floor((1 + np.sqrt(1 + 8*k)) / 2))
    i = k - j*(j-1)//2
    return float(d[k]), (int(i), int(j))

def parse_coords(field):
    """Safely parse the 'coord' field which is a Python-like literal string."""
    val = ast.literal_eval(field)
    return np.asarray(val, dtype=float)

def main():
    ap = argparse.ArgumentParser(description="Compute max interatomic distance for each structure in a CSV.")
    ap.add_argument("csv_path", help="Input CSV path with columns: Name, Energy, Forces, coord")
    ap.add_argument("-o", "--out", help="Optional output CSV to write results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "Name" not in df.columns or "coord" not in df.columns:
        raise SystemExit("CSV must contain at least 'Name' and 'coord' columns.")

    results = []
    for _, row in df.iterrows():
        name = row["Name"]
        coords = parse_coords(row["coord"])
        max_d, (i, j) = max_atom_distance_scipy(coords)
        results.append({
            "Name": name,
            "MaxDistance": max_d,
            "AtomIndexI": i,
            "AtomIndexJ": j
        })

    res_df = pd.DataFrame(results).sort_values("Name").reset_index(drop=True)
    print(res_df.to_string(index=False))

    if args.out:
        res_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
