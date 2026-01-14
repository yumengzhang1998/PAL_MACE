import pickle

import matplotlib.pyplot as plt

import random
import ast
import argparse
import pandas as pd
import os

def read_traj(traj_file):
    """Read a trajectory from a file."""
    with open(traj_file, "rb") as f:
        traj = pickle.load(f)
    return traj

def get_energy_list(traj):
    """Get the energy list from a trajectory."""
    return [float(frame[-3]) for frame in traj]

def plot_energy_trend(energy_trajs, prefix, traj_name):
    """Plot the energy trend of a trajectory."""
    # get a trend of energy by each time step

    time_steps = list(range(len(energy_trajs[0])))

    # Plot all 10 trajectories
    plt.figure(figsize=(8, 5))

    for i, energy_list in enumerate(energy_trajs):
        # plt.plot(time_steps, energy_list, label=f"Traj {i+1}", alpha=0.7)
        plt.plot(time_steps, energy_list, label=f"_nolegend_", alpha=0.7)

    # Labels and title
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Optimized Energy')

    plt.xlabel("Time Step")
    plt.ylabel("Predicted Energy - Optimized Energy(eV)")
    plt.title("Energy Change Trend Over Time for random Trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}/{traj_name}/pred_energy_traj.png")

    # Show the plot
    plt.show()


def read_optimimal_energy(prefix):
    df = pd.read_csv("optimized.csv")
    # Extract the energy values from the DataFrame
    df['Name'] = df['Name'].str.lower()
    for row in df.iterrows():
        name = row[1]['Name']
        energy = row[1]['Energy']
        if name == "bi11-3_nonsym":
            name = "bi11-3"
        if name == prefix:
            return ast.literal_eval(energy)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Separate batch trajectories.")
    parser.add_argument("--element", type=str, required=True, help="Element symbol (e.g., 'bi')")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the system (e.g., -2)")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms (e.g., 4)")
    parser.add_argument("--model_number", type=int, required=True, help="Model number (e.g., 25)")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to simulate")
    parser.add_argument("--synthesis", type=str, required=True, help="if the data is from synthesis")
    args = parser.parse_args()
    if args.synthesis == "True":
        prefix = f"{args.element}{args.num_atom}{args.charge}_samples"
    else:
        prefix = f"{args.element}{args.num_atom}{args.charge}"
    traj_name = f"{args.model_number}_{args.steps}steps"
    output_dir  = f"{prefix}/{traj_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    a = read_traj(f"{prefix}/{traj_name}_traj.pkl")
    all_traj = len(a)

    energy_trajs = [get_energy_list(traj) for traj in a]
    # sample from trajs

    random.seed(42)
    energy_trajs = random.sample(energy_trajs, 10)
    # get the energy of the optimized structure
    optimized_energy = read_optimimal_energy(prefix)
    
    print(f"Optimized energy: {optimized_energy}")
    energy_trajs = [[e - optimized_energy[0] for e in energy] for energy in energy_trajs]

    plot_energy_trend(energy_trajs, prefix, traj_name)
    print(f"✅ Energy trend plot saved in: {output_dir}/pred_energy_traj.png")



