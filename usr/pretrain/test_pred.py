import pandas as pd
import ast
import numpy as np
import torch
from torch_geometric.data import Data

from evaluation import evaluate

import matplotlib.pyplot as plt

def generate_data_list_from_df(df, charge):
    """
    Convert a pandas DataFrame containing atomic structure data into a list of PyTorch Geometric Data objects.

    Args:
        df (pd.DataFrame): A DataFrame with columns ["atoms", "coordinates", "total_energy", "forces"]

    Returns:
        List[Data]: A list of PyTorch Geometric Data objects
    """

    data_list = []
    
    for _, row in df.iterrows():
        num_atoms = len(row["atoms"])  # Number of atoms per structure
        
        data = Data(
            pos=torch.tensor(row["coordinates"], dtype=torch.float),  # Atomic positions
            z=torch.tensor([83] * num_atoms, dtype=torch.int32),  # Atomic numbers (assuming Bi=83)
            y=torch.tensor(row["total_energy"], dtype=torch.float),  # Energy
            forces=torch.tensor(row["forces"], dtype=torch.float),  # Forces
            charge=torch.tensor(charge, dtype=torch.get_default_dtype()),  # Placeholder charge
            atoms=[83] * num_atoms,  # Atomic numbers list
            atom_types=['Bi'] * num_atoms  # Atom type list
        )

        data_list.append(data)
    return data_list
    
def get_res(sample, test_data, prefix, batch_size):
    model = torch.load(f"full_data_charge_embed/{prefix}_logs/sample_{sample}/{prefix}.model")  
    model.eval()

    energies, forces_list, stresses_list, contributions_list = evaluate(model, 
                                                                        test_data,
                                                                        batch_size,
                                                                        default_dtype = "float64",
                                                                        device = "cpu",                 
                                                                        compute_stress = False,
                                                                        return_contributions = False
                                                                        ) 
    return energies, forces_list, stresses_list, contributions_list



prefix = 'bi4-2'
df = pd.read_csv(f"full_data_charge_embed/{prefix}_logs/{prefix}.csv")

columns_to_convert = ["atoms", "coordinates", "total_energy", "forces"]

for col in columns_to_convert:
    df[col] = df[col].apply(ast.literal_eval)

df['coordinates'] = df['coordinates'].apply(lambda x: np.array(x).reshape(-1, 3))
df['total_energy'] = df['total_energy'].apply(lambda x: x[0])
df['forces'] = df['forces'].apply(lambda x: np.array(x).reshape(-1, 3))

test_data = generate_data_list_from_df(df, charge=-2)
print(test_data[0])




pred_list = []

for i in range(2):
    energies, forces_list, stresses_list, contributions_list = get_res(i, test_data, prefix, 32)
    pred_list.append(energies)

array = np.array(pred_list)
std = np.std(array, axis=0)
percentile_value = np.percentile(std, 60)

# Find all values above this percentile
significant_values = std[std > percentile_value]
print(significant_values.min())
plt.hist(std, bins='auto', alpha=0.7, rwidth=0.85)
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.title('Distribution of Standard Deviations')
plt.grid(axis='y', alpha=0.75)
plt.savefig(f'chargeembed/{prefix}_logs/std_hist.png')
#clear plot to create new plot
plt.clf()


df = pd.read_csv(f'raw/{prefix}.csv')
if df['total_energy'].dtype == 'float64':
    print("Data is already in correct format.")
    total_energy = df['total_energy']
else:
    total_energy = [x[0] for x in df['total_energy'].apply(ast.literal_eval)]
# print(total_energy)
mu = np.mean(total_energy)  # Mean
sigma = np.std(total_energy)  # Standard deviation
median = np.median(total_energy)  # Median
std_lower_bound = mu - 2 * sigma
std_upper_bound = mu + 2 * sigma
Q1 = np.percentile(total_energy, 25)  # 25th percentile
Q3 = np.percentile(total_energy, 75)  # 75th percentile
IQR = Q3 - Q1

# Define thresholds
k = 1.5  # Adjust as needed
qu_lower_bound = Q1 - k * IQR
qu_upper_bound = Q3 + k * IQR

print(f"Median: {median}, Lower Bound: {qu_lower_bound}, Upper Bound: {qu_upper_bound}")
print(f"Mean: {mu}", f"Standard Deviation: {sigma}", f"Lower Bound: {std_lower_bound}", f"Upper Bound: {std_upper_bound}")
# Plot the data: distribution ofn total energy


plt.hist(total_energy, bins='auto', alpha=0.7, rwidth=0.85)
plt.xlabel('Total Energy')
plt.ylabel('Frequency')
plt.title('Distribution of Total Energies')
plt.grid(axis='y', alpha=0.75)

# Use ScalarFormatter to prevent abbreviation of x-axis labels
# ax = plt.gca()
# ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# ax.ticklabel_format(style='plain', axis='x')

plt.savefig(f'chargeembed/{prefix}_logs/total_energy_hist.png')
plt.show()