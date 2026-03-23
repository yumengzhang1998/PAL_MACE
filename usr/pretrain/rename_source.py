import pandas as pd
import glob
import os

# 1. Define your mapping dictionary
source_map = {"real": 0, "synthesis_bi4": 1, "synthesis_bi2": 2}

# 2. Use glob to find all .csv files matching your path pattern
# The ** and recursive=True allows it to find files in any sample_{i} folder
path_pattern = "samples/bi11-3_samples/sample_*/*.csv"
files = glob.glob(path_pattern, recursive=True)

for file_path in files:
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # 3. Check if 'source' column exists
        if 'source' in df.columns:
            print(f"Processing: {file_path}")
            
            # 4. Map the strings to integers
            # .map() will replace the strings using your dictionary
            df['source'] = df['source'].map(source_map)
            
            # 5. Overwrite the file with the updated column
            df.to_csv(file_path, index=False)
        else:
            print(f"Skipping: {file_path} (No 'source' column found)")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("--- All files processed ---")