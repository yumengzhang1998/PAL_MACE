import os
import argparse
import pandas as pd
import numpy as np
import ast
def get_optimal_energy(path, cluster_type):
    opt_df = pd.read_csv(path)
    if "energy" not in opt_df.columns:
        ec = next((c for c in opt_df.columns if "energy" in c.lower()), None)
        if ec is None:
            raise ValueError(f"No energy column found in {path}. Columns: {list(opt_df.columns)}")
        opt_df = opt_df.rename(columns={ec: "energy"})
    opt_df = opt_df[opt_df['Name'] == cluster_type]
    if opt_df.empty:
        raise ValueError(f"No entry found for cluster type '{cluster_type}' in {path}.")
    energy = opt_df['energy'].values[0]
    s = str(energy).strip()

    # Remove surrounding brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    energy = float(s)
    return energy



def parse_symbols(value):
    """
    Turn things like:
      "[83, 83, 83, 83]"  -> ["83", "83", "83", "83"]
      "Bi Bi Bi Bi"       -> ["Bi", "Bi", "Bi", "Bi"]
      [83, 83, 83, 83]    -> ["83", "83", "83", "83"]
    into a clean list of strings.
    """
    if isinstance(value, (list, np.ndarray)):
        return [str(v) for v in value]

    s = str(value).strip()

    # Remove surrounding brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    # Replace commas by spaces, then split
    tokens = s.replace(",", " ").split()

    # Drop empty tokens just in case
    tokens = [t for t in tokens if t]

    return tokens

def save_xyz(df: pd.DataFrame, xyz_path: str):
    """Save structures from dataframe to an XYZ file."""
    required_cols = ["atoms", "node_feature"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    with open(xyz_path, "w") as f:
        for _, row in df.iterrows():
            symbols = parse_symbols(row["atoms"])
            coords = row["node_feature"] if "node_feature" in row else row["coords"]

            # Parse coordinates
            if isinstance(coords, str):
                coords = np.array(ast.literal_eval(coords))  # assume [[x,y,z],...] format
            coords = np.asarray(coords)

            if not isinstance(symbols, (list, np.ndarray)):
                # If symbols is a space-separated string
                symbols = str(symbols).split()

            if len(symbols) != len(coords):
                raise ValueError(
                    f"Mismatch: {len(symbols)} symbols vs {len(coords)} coords"
                )


            f.write(f"{len(symbols)}\n")
            f.write("OOD structure\n")
            for s, (x, y, z) in zip(symbols, coords):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Saved XYZ: {xyz_path}")

def main():
    ap = argparse.ArgumentParser(description="Filter all structures with higher energy than the optimal energy (lowest one).")
    ap.add_argument("--prefix", type=str, required=True, help="Folder prefix, e.g. 'bi4-6'")
    ap.add_argument("--model_number", type=str, required=True, help="Model number, e.g. '56'")
    ap.add_argument("--cluster_name", type=str, required=False, help="Cluster type to look for optimal energy, e.g. 'bi4-6'")
    args = ap.parse_args()

    csv_path = os.path.join(args.prefix, f"{args.model_number}_added_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "energy" not in df.columns:
        ec = next((c for c in df.columns if "energy" in c.lower()), None)
        if ec is None:
            raise ValueError(f"No energy column found in {csv_path}. Columns: {list(df.columns)}")
        df = df.rename(columns={ec: "energy"})

    if not np.isfinite(df["energy"]).any():
        raise ValueError("No valid energy values found.")

    e_min = get_optimal_energy("../optimized.csv", args.cluster_name) + 3.5
    print(f"Optimal (lowest) energy: {e_min:.6f}")

    df_ood = df[df["energy"] >= e_min].reset_index(drop=True)
    df_ood_train = df_ood[df_ood["type"] == "train"]
    df_ood_val = df_ood[df_ood["type"] == "val"]
    print(f"Found {len(df_ood_train)} structures with higher energy in training set.")
    print(f"Found {len(df_ood_val)} structures with higher energy in validation set.")
    types = ["train", "val"]
    for type in types:
        df_ood_type = df_ood[df_ood["type"] == type].reset_index(drop=True)
        out_csv = f"{args.prefix}_ood_{type}.csv"
        out_xyz = f"{args.prefix}_ood_{type}.xyz"

        df_ood_type.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")

    # Save coordinates if available
        if "node_feature" in df_ood_type.columns and "atoms" in df_ood_type.columns:
            save_xyz(df_ood_type, out_xyz)
        else:
            print("No 'coords' or 'symbols' columns found; XYZ not created.")

if __name__ == "__main__":
    main()
