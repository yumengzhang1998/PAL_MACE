import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_energy_distribution(csv_file, output_dir):
    df = pd.read_csv(csv_file)

    # Ensure 'type' and 'energy' columns exist
    if 'type' not in df.columns or 'energy' not in df.columns:
        raise ValueError("CSV file must contain 'type' and 'energy' columns.")

    # Separate data
    train_energies = df[df['type'] == 'train']['energy']
    val_energies = df[df['type'] == 'val']['energy']

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(train_energies, bins=50, alpha=0.6, label='Train', color='blue')
    plt.hist(val_energies, bins=50, alpha=0.6, label='Validation', color='orange')

    plt.xlabel('Energy (eV)')
    plt.ylabel('Frequency')
    plt.title('Energy Distribution: Train vs Validation')
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "energy_distribution.png")
    plt.savefig(output_path)
    plt.show()

    print(f"✅ Energy distribution plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot energy distribution from CSV")
    parser.add_argument("--num_atom", type=str, required=True, help="number of atoms in the system")
    parser.add_argument("--charge", type=str, default="plots", help="global charge of the system")
    parser.add_argument("--syn", type=str, default="False", help="if it is a synthetic dataset")
    args = parser.parse_args()
    if args.syn.lower() == "true":
        args.outdir = f"../results/bi{args.num_atom}{args.charge}_samples/"
        args.csv = f"../results/bi{args.num_atom}{args.charge}_samples/added_data.csv"
    else:
        args.outdir = f"../results/bi{args.num_atom}{args.charge}/"
        args.csv = f"../results/bi{args.num_atom}{args.charge}/added_data.csv"

    plot_energy_distribution(args.csv, args.outdir)
