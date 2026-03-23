import pandas as pd

# 1. Load the CSV files
# Replace these filenames if your actual files are named differently or are in a different folder
paths = ["samples/bi11-3_samples/sample_1/train.csv",
         "samples/bi11-3_samples/sample_1/val.csv",""
         "samples/bi11-3_samples/sample_0/val.csv",
         "samples/bi11-3_samples/sample_0/train.csv",
            "samples/bi11-3_samples/sample_2/train.csv",
            "samples/bi11-3_samples/sample_2/val.csv"]


main_df = pd.read_csv("raw/bi11-3_samples_parsed.csv")

for path in paths:
    train_df = pd.read_csv(path)
    # Optional: If train.csv accidentally retained an empty 'source' column, drop it first
    if 'source' in train_df.columns:
        train_df = train_df.drop(columns=['source'])

    # 2. Extract the mapping from the main dataframe
    # We only need 'coordinates' and 'source'. We drop duplicates to ensure a clean 1-to-1 match.
    mapping_df = main_df[['coordinates', 'source']].drop_duplicates(subset=['coordinates'])

    # 3. Merge the 'source' column into the train dataset
    # A 'left' join ensures all rows in train_df are kept exactly as they are.
    train_updated = pd.merge(train_df, mapping_df, on='coordinates', how='left')

    # 4. Save the updated dataframe to a new CSV file
    train_updated.to_csv(path, index=False)

    # Quick check to see if any rows failed to find a match
    missing_sources = train_updated['source'].isna().sum()
    print(f"Update complete! Saved to 'train_with_source.csv'.")
    print(f"Rows missing a source after merge: {missing_sources}")