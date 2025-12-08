import os
import pandas as pd

# Base directory where your collected folders are
base_dir = "./collected_data"

# Prefix directories to process
prefixes = ["bi11-3_samples_bi2", "bi11-3_samples", "bi11-3",
            "bi7-3", "bi4-6", "bi4-2", "bi2-2"]

for prefix in prefixes:
    dir_path = os.path.join(base_dir, prefix)
    if not os.path.isdir(dir_path):
        print(f"⚠️ Directory not found: {dir_path}")
        continue

    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ No CSV files found in {dir_path}")
        continue

    dfs = []
    for f in csv_files:
        file_path = os.path.join(dir_path, f)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Read: {file_path} ({df.shape[0]} rows)")
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")

    # Merge all DataFrames
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(dir_path, f"{prefix}.csv")
        merged_df.to_csv(out_path, index=False)
        print(f"✅ Saved merged file: {out_path} ({merged_df.shape[0]} rows)")
    else:
        print(f"⚠️ No valid CSVs to merge in {dir_path}")
import os
import pandas as pd

# Base directory where your collected folders are
base_dir = "./collected_data"

# Prefix directories to process
prefixes = ["bi11-3_samples_bi2", "bi11-3_samples", "bi11-3",
            "bi7-3", "bi4-6", "bi4-2", "bi2-2"]

for prefix in prefixes:
    dir_path = os.path.join(base_dir, prefix)
    if not os.path.isdir(dir_path):
        print(f"⚠️ Directory not found: {dir_path}")
        continue

    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ No CSV files found in {dir_path}")
        continue

    dfs = []
    for f in csv_files:
        file_path = os.path.join(dir_path, f)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Read: {file_path} ({df.shape[0]} rows)")
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")

    # Merge all DataFrames
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        print("Removing duplicates... current row count:", merged_df.shape[0])
        merged_df.drop_duplicates(inplace=True)
        print("After removing duplicates, row count:", merged_df.shape[0])
        out_path = os.path.join(dir_path, f"{prefix}.csv")
        merged_df.to_csv(out_path, index=False)
        print(f"✅ Saved merged file: {out_path} ({merged_df.shape[0]} rows)")
    else:
        print(f"⚠️ No valid CSVs to merge in {dir_path}")
