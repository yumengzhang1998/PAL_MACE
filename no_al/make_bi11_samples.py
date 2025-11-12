import pandas as pd

df_bi2 = pd.read_csv("./collected_data/bi11-3_samples_bi2.csv")
df_bi4_2 = pd.read_csv("./collected_data/bi11-3_samples_bi4.csv")
df_bi11_3_samples = pd.read_csv("./collected_data/bi11-3.csv")

df_bi2['source'] = 'bi2'
df_bi4_2['source'] = 'bi4'
df_bi11_3_samples['source'] = 'bi11'

merged_df = pd.concat([df_bi2, df_bi4_2, df_bi11_3_samples], ignore_index=True).drop_duplicates()
merged_df.reset_index(drop=True, inplace=True)
# remove type column if exists
if 'type' in merged_df.columns:
    merged_df = merged_df.drop(columns=['type'])
merged_df.to_csv("./collected_data/bi11-3_samples.csv", index=False)
