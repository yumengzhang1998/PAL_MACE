import pandas as pd
import ast
import numpy as np

# Load your CSV


# Convert stringified list to actual NumPy array and flip sign
def parse_and_negate_force(force_str):
    force_array = np.array(ast.literal_eval(force_str))
    return (-1) * force_array


def parse(path, prefix):
    df = pd.read_csv(f"{path}/{prefix}.csv")
    # Apply to the 'forces' column
    df['forces'] = df['forces'].apply(parse_and_negate_force)
    # Convert back to string if saving
    df['forces'] = df['forces'].apply(lambda x: x.tolist())
    df.to_csv(f"{path}/{prefix}_parsed.csv", index=False)


if __name__ == "__main__":
    prefix = ['bi4-2', 'bi4-6', 'bi7-3', 'bi11-3', 'bi11-3_samples']
    prefix = ['bi0','bi11-3', 'bi11-3_samples']
    for p in prefix:
        parse(path = "raw/", prefix = p)
