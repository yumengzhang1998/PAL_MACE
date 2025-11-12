import pandas as pd
import ast
from ase.data import chemical_symbols

# Input CSV
input_file = "../results/bi4-6/added_data.csv"
output_file = "train.xyz"

# Read CSV
df = pd.read_csv(input_file)

with open(output_file, "w") as f:
    for idx, row in df.iterrows():
        if row["type"].strip() != "train":
            continue

        # Parse atoms and node_feature
        atoms = ast.literal_eval(row["atoms"])
        positions = ast.literal_eval(row["node_feature"])

        n_atoms = len(atoms)
        f.write(f"{n_atoms}\n")

        # Comment line with metadata
        f.write(f"energy={row['energy']} global_charge={row['global_charge']}\n")

        # Write atom coordinates
        for atom, pos in zip(atoms, positions):
            symbol = chemical_symbols[int(atom)]
            x, y, z = pos
            f.write(f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}\n")
