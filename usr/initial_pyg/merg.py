import pandas as pd

# df_bi11 = pd.read_csv("raw/bi11-3_parsed.csv")
# df_bi11_samples_bi2 = pd.read_csv("raw/Bi11-3_samples_bi2_parsed.csv")
# df_bi11_samples_bi4 = pd.read_csv("raw/bi11-3_samples_bi4_parsed.csv")
# df_bi11_samples_bi4 = df_bi11_samples_bi4[df_bi11_samples_bi4['source'] == "synthesis"]
# df_bi11_samples_bi4['source'] = "synthesis_bi4"
# df_bi11_samples_bi2['source'] = "synthesis_bi2"
# df_bi11['source'] = "real"

# merged_df = pd.concat([df_bi11, df_bi11_samples_bi2, df_bi11_samples_bi4], ignore_index=True)

# merged_df.to_csv("raw/bi11-3_samples_parsed.csv", index=False)
import pandas as pd

def reorder_by_source(df, source_col="source"):
    # Separate 'real' and non-'real'
    real_df = df[df[source_col].astype(str).str.strip().eq("real")]
    non_real_df = df[~df[source_col].astype(str).str.strip().eq("real")]

    # Get unique non-"real" sources in the order they appear
    non_real_sources = non_real_df[source_col].unique().tolist()

    # Group non-real rows by source
    grouped = [non_real_df[non_real_df[source_col] == src] for src in non_real_sources]

    # Interleave non-real sources
    interleaved = []
    max_len = max(len(g) for g in grouped)
    for i in range(max_len):
        for g in grouped:
            if i < len(g):
                interleaved.append(g.iloc[i])

    interleaved_df = pd.DataFrame(interleaved)

    # Concatenate 'real' first, then interleaved others
    result = pd.concat([real_df, interleaved_df], ignore_index=True)
    return result

df = pd.read_csv("raw/bi11-3_samples_parsed.csv")
new_df = reorder_by_source(df)

new_df.to_csv("raw/bi11-3_samples_parsed.csv", index=False)