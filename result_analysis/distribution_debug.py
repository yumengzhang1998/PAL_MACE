import pandas as pd
import ast
import numpy as np

def read_added_data(path):
    df = pd.read_csv(path)
    df['atoms'] = df['atoms'].apply(lambda x: ast.literal_eval(x))
    df['node_feature'] = df['node_feature'].apply(lambda x: ast.literal_eval(x))
    df['node feature'] = [np.array(x) for x in df['node_feature']]
    df['energy'] = df['energy'].apply(lambda x: float(x))
    df['force'] = df['force'].apply(lambda x: ast.literal_eval(x))
    df['force'] = [np.array(x) for x in df['force']]
    return df

def write_xyz(df,prefix,model):
    coords = df['node feature'].values
    coords = np.array([np.array(x) for x in coords])
    coords = [x.reshape(-1, 3) for x in coords]
    atoms = df['atoms'].values
    energies = df['energy'].values
    forces = df['force'].values
    with open(f'{prefix}_{model}_added_data.xyz', 'w') as f:
        for i in range(len(df)):
            f.write(f"{len(atoms[i])}\n")
            f.write(f"Energy: {energies[i]}\n")
            for j in range(len(atoms[i])):
                f.write(f"{atoms[i][j]} {coords[i][j][0]} {coords[i][j][1]} {coords[i][j][2]} {forces[i][j][0]} {forces[i][j][1]} {forces[i][j][2]}\n")

def plot_distribution(data, title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(data, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")


if __name__ == "__main__":
    prefix = 'bi7-3'
    model = 56
    path = f'../results/{prefix}/{model}_added_data.csv'
    df = read_added_data(path)
    print(len(df))
    # write_xyz(df,prefix,model)
    # Plot distributions
    energies = df['energy'].values
    forces = np.concatenate(df['force'].values, axis=0)
    plot_distribution(energies, f'Energy Distribution for {prefix} Model {model}', 'Energy', 'Frequency')
    plot_distribution(forces.flatten(), f'Force Distribution for {prefix} Model {model}', 'Force', 'Frequency')