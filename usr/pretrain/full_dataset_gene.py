import pandas as pd

prefix = ["bi2-2", "bi4-2","bi4-6", "bi7-3","bi11-3_samples","bi11-3_samples_bi2", "bi11-3_samples"]
df_list = []
for p in prefix:
    df = pd.read_csv(f'raw/{p}.csv')
    # add a column to identify the source
    charge = int(p.split('-')[-1]) if 'samples' not in p else int(p.split('-')[-1].split('_')[0])
    if charge > 0:
        charge = -charge
    df['charge'] = charge
    df['num_atom'] = int(p.split('-')[0].replace('bi',''))
    df['source'] = p
    df_list.append(df)
    


df = pd.concat(df_list, ignore_index=True)
df.to_csv('raw/bi0.csv', index=False)


