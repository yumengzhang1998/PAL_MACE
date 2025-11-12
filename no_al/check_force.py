import pandas as pd
import numpy as np
from ast import literal_eval

# === CONFIG ===
csv_path = "collected_data/bi11-3_samples.csv"   # change this to your actual path
force_threshold = 100.0                 # eV/Å, adjust as needed
save_cleaned = True                     # whether to save cleaned version
clean_path = csv_path.replace(".csv", "_clean.csv")

bad_rows = []

def safe_parse_force(cell):
    """Safely parse the forces string into a numpy array."""
    try:
        arr = np.array(literal_eval(str(cell)), dtype=float)
        return arr
    except Exception as e:
        print(f"⚠️ Parse error in row: {e}")
        return None

# Read CSV fully as text (no automatic type guessing)
df = pd.read_csv(csv_path, dtype=str)
max_force_ls = []
for i, row in df.iterrows():
    forces = safe_parse_force(row["forces"])
    if forces is None:
        bad_rows.append(i)
        continue

    # check shape
    if forces.ndim != 2 or forces.shape[1] != 3:
        print(f"⚠️ Wrong shape at row {i}: {forces.shape}")
        bad_rows.append(i)
        continue

    max_force = np.abs(forces).max()
    max_force_ls.append(max_force)
    if max_force > force_threshold:
        #print(f"❌ Row {i} has high force magnitude: {max_force:.2f}")
        bad_rows.append(i)

print("\n=== Summary ===")
print(f"Total rows checked: {len(df)}")
print(f"Bad rows found: {len(bad_rows)}")
print(f"Maximum force magnitudes stats: min={np.min(max_force_ls):.2f}, max={np.max(max_force_ls):.2f}, mean={np.mean(max_force_ls):.2f}")
print(f"Force id of maximum: {np.argmax(max_force_ls)} with value {np.max(max_force_ls):.2f}")
print(f"bad row index of maxiximum: {bad_rows[np.argmax(max_force_ls)] if bad_rows else 'N/A'}")
# if bad_rows:
#     print("Indices of bad rows:", bad_rows)

# Optionally save a cleaned version without the bad rows
if save_cleaned:
    df_clean = df.drop(index=bad_rows)
    df_clean.to_csv(clean_path, index=False)
    print(f"\n✅ Cleaned CSV saved to: {clean_path}")
