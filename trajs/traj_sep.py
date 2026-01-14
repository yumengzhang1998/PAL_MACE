import pickle
import os
import argparse
def read_traj(traj_file):
    """Read a trajectory from a file."""
    trajs = []
    with open(traj_file, "rb") as f:
        trajs = pickle.load(f)
    return trajs
def traj_sep(traj_list, prefix, traj_name):
    """Separate a list of trajs: [traj1, traj2, ..., traj_n] and write each traj to a separate xyz file.
       Each traj is a list of lists consisting of:         
        [   coords_distorded, # pos
            geometry[1], # atom_numbers
            None, # true_energy
            true_force_empty, # true_forces
            geometry[4], # charge
            vec3_to_numpy(forces), # pred_forces 
            None,   # pred_energy
            geometry[-2], # patience
            vec3_to_numpy(velocity)].
    """
    # Create directory to store the separated trajs
    dir = traj_name
    output_dir = f"{prefix}/{dir}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, traj in enumerate(traj_list):
        print(f"Processing trajectory {i + 1}/{len(traj_list)}...")
        output_file = f"{output_dir}/traj_{i}.xyz"
        if i == 11:
        
            with open(output_file, "w") as f:
                num_atoms = len(traj[0][0])  # Number of atoms from the first frame
                # f.write(f"{num_atoms}\n")
                # f.write(f"Trajectory {i}\n")

                for frame in traj:
                    f.write(f"{num_atoms}\n")
                    f.write("Generated from NNFF\n")  # Placeholder comment

                    for atom in frame[0]:  # Atom positions
                        f.write(f"Bi {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n")  # Explicitly add "Bi"
            print(f"✅ Trajectory {i} saved in: {output_file}")

    print(f"✅ Trajectories saved in: {output_dir}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Separate batch trajectories.")
    parser.add_argument("--element", type=str, required=True, help="Element symbol (e.g., 'bi')")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the system (e.g., -2)")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms (e.g., 4)")
    parser.add_argument("--model_number", type=int, required=True, help="Model number (e.g., 25)")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to simulate")
    parser.add_argument("--synthesis", type=str, required=True, help="if the data is from synthesis")
    parser.add_argument("--project_name", type=str, required=True, help="Project name")
    args = parser.parse_args()
    if args.synthesis == "True":
        prefix = f"{args.element}{args.num_atom}{args.charge}_samples"
    else:
        prefix = f"{args.element}{args.num_atom}{args.charge}"
    project_name = args.project_name
    traj_name = f"{args.model_number}_{args.steps}steps"
    traj = f"{prefix}/{project_name}/{traj_name}_traj.pkl"
    a  = read_traj(traj)
    traj_sep(a, prefix, traj_name)