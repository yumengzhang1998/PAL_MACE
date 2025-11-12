#!/usr/bin/env python3
# max_dist_from_csv.py
import argparse
import ast
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
def get_energy_distribution(prefix):
    """Read a CSV file and return the energy distribution.
    pth should lead to a CSV with an 'Energy' column.
    under the folder with name as {prefix}_parsed.csv
    Returns (max, min, median, mean)
    """
    energy_path = f"usr/initial_pyg/raw/{prefix}_parsed.csv"
    df = pd.read_csv(energy_path)
    if "total_energy" not in df.columns:
        raise ValueError("CSV must contain an 'Energy' column.")
    energies = df["total_energy"].values
    energies = [ast.literal_eval(energy)[0] for energy in energies]
    _maximum = np.max(energies)
    _minimum = np.min(energies)
    _median = np.median(energies)
    _mean = np.mean(energies)
    _q25 = np.percentile(energies, 25)
    _q75 = np.percentile(energies, 75)
    print(f"Energy Distribution:\n Max: {_maximum}\n Min: {_minimum}\n Median: {_median}\n Mean: {_mean}")
    return _maximum, _minimum, _median, _mean, _q25, _q75


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
    ap.add_argument("--csv_path", help="Input CSV path with columns: Name, Energy, Forces, coord")
    # ap.add_argument("--energy_path", action="store_true", help="If set, print energy distribution and exit")
    ap.add_argument("-o", "--out", help="Optional output CSV to write results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "Name" not in df.columns or "coord" not in df.columns:
        raise SystemExit("CSV must contain at least 'Name' and 'coord' columns.")

    results = []
    max_d_list = []
    num_atom_list = []
    median_list = []
    _q25_list = []
    _q75_list = []
    for _, row in df.iterrows():
        name = row["Name"]
        coords = parse_coords(row["coord"])
        max_d, (i, j) = max_atom_distance_scipy(coords)
        max_d_list.append(max_d)
        num_atom = coords.shape[0]
        print(f"{name}: Max Distance = {max_d:.4f} between atoms {i} and {j}, Num Atoms = {num_atom}")
        num_atom_list.append(num_atom)
        results.append({
            "Name": name,
            "MaxDistance": max_d,
            "AtomIndexI": i,
            "AtomIndexJ": j
        })
        maximum, minimum, median, mean, q25, q75 = get_energy_distribution( name.lower())
        median_list.append([median])
        _q25_list.append([q25])
        _q75_list.append([q75])




    res_df = pd.DataFrame(results).sort_values("Name").reset_index(drop=True)
    print(res_df.to_string(index=False))
    df['MaxDistance'] = max_d_list
    df['NumAtoms'] = num_atom_list
    
    df['energy_threshold'] = median_list
    df['q25'] = _q25_list
    df['q75'] = _q75_list
    df.to_csv(args.csv_path, index=False)  # overwrite original CSV with new column


    

    if args.out:
        res_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
