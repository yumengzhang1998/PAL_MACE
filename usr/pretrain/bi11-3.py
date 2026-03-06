import pandas as pd

df = pd.read_csv('raw/bi11-3.csv')

energies_0 = [[e] for e in df['total_energy'].values]
print(energies_0)
df['total_energy'] = energies_0

df.to_csv('raw/bi11-3.csv', index=False)