import pandas as pd
import ast
import numpy as np
import torch
from torch_geometric.data import Data

from charge_eval import evaluate

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
    model = torch.load(f"coulomb_10/{prefix}_logs/sample_{sample}/{prefix}.model")  
    model.eval()

    energies, forces_list, stresses_list, contributions_list, charge_list = evaluate(model, 
                                                                        test_data,
                                                                        batch_size,
                                                                        default_dtype = "float64",
                                                                        device = "cuda",                 
                                                                        compute_stress = False,
                                                                        return_contributions = False
                                                                        ) 
    return energies, forces_list, stresses_list, contributions_list, charge_list



atom = 'bi'
num_atom = 4
charge = -6
prefix = atom + str(num_atom)+ str(charge)
df = pd.read_csv(f"raw/{prefix}_parsed.csv")

columns_to_convert = ["atoms", "coordinates", "total_energy", "forces"]

for col in columns_to_convert:
    df[col] = df[col].apply(ast.literal_eval)

df['coordinates'] = df['coordinates'].apply(lambda x: np.array(x).reshape(-1, 3))
df['total_energy'] = df['total_energy'].apply(lambda x: x[0])
df['forces'] = df['forces'].apply(lambda x: np.array(x).reshape(-1, 3))

test_data = generate_data_list_from_df(df, charge=charge)
print(test_data[0])

energies, forces_list, stresses_list, contributions_list, charge_list = get_res(0, test_data, prefix, 1)
for i in charge_list:
    #sum latent charge which is an array
    print(i.sum())
    print(i)

    q = np.array(i)

    total_negative = np.sum(q)

    relative_ratios = charge * q / total_negative
    print(relative_ratios)
    # Plotting