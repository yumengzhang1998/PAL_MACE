import pickle

import sys

sys.path.append("../../quantum_chem_python/")
import os
current_path = os.getcwd()
import re
from quantum_chem_python.api.settings import GeneralSettings, MultiProcessingSettings, XTBSettings, TurbomolSettings
from tqdm import tqdm  
from quantum_chem_python.api.xtb.xtb_api import XTBApi
from quantum_chem_python.api.turbomol.turbomol_api import TurbomolApi
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import random
import ast
import argparse
import pandas as pd
import os
import tempfile
import multiprocessing as mp
from contextlib import contextmanager
import numpy as np
@contextmanager
def set_threads(threads: int):
    # Tell OpenMP/BLAS libraries how many threads to use inside this process
    old = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")}
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    # (optional) nicer placement/binding
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES", "cores")
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def _one_dft(args):
    atoms, coords, charge, use_dft, threads = args
    with set_threads(threads):
        # give each worker its own scratch
        tmp = create_unique_scratch_dir(f"tmp_worker_{mp.current_process().pid}")
        return run_calc(atoms, coords, charge, use_dft)
def create_unique_scratch_dir(name):
    scratch_dir = os.path.join(tempfile.gettempdir(), f"scratch_{name}")
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir


tmp = create_unique_scratch_dir("tmp")
def run_calc(atoms, pos, charge, dft):
    """
    Run energy and force calculations.

    Args:
        atoms (list): List of atomic symbols.
        pos (numpy.ndarray): Atomic positions.
        charge (int): Molecular charge.
        dft (bool): Use DFT (True) or XTB (False).

    Returns:
        tuple: (energy, forces) or (None, None) if the calculation fails.
    """
    if not isinstance(pos, list):
        pos = [pos]
    try:
        settings = GeneralSettings(mp_settings=MultiProcessingSettings(mp_active=False, number_of_workers=1),
                                    output_dir_path=f"{tmp}/results/test_output",
                                    input_file_path=f"{tmp}/results/",
                                    delete_run_dir=True,
                                    load_from_file=False,
                                    coords=pos,
                                    elements=atoms)

        if dft:
            turbomol_settings = TurbomolSettings(basis="dhf-TZVP", 
                                                functional="tpss", 
                                                method="ridft",
                                                input_in_angstrom=True, 
                                                use_cosmo=True, 
                                                epsilon='infinity', 
                                                charge=charge)
            turbomol_api = TurbomolApi(general_settings=settings, turbomol_settings=turbomol_settings)
            energy, forces = turbomol_api.get_energy_and_gradient()
        else:
            xtb_settings = XTBSettings(binary_path="/home/yumeng/xtb-6.6.0/bin/xtb", charge=charge, solvent="Aniline", iterations=500, accuracy=500)
            xtb_api = XTBApi(general_settings=settings, xtb_settings=xtb_settings)
            energy, forces = xtb_api.get_energy_and_gradient()
        return energy, forces
    except Exception as e:
        print(f"Calculation failed: {e}")
        os.chdir(current_path)
        return None, None

def read_traj(traj_file):
    """Read a trajectory from a file."""
    with open(traj_file, "rb") as f:
        traj = pickle.load(f)
    return traj
def get_energy_list(traj):
    """Get the energy list from a trajectory."""
    return [float(frame[-3]) for frame in traj]
def get_coord_list(traj):
    """Get the coordinate list from a trajectory."""
    return [frame[0] for frame in traj]
def _valid_xy(y, x_step=1):
    """Return x,y with None/NaN filtered out. x grows by x_step."""
    xs, ys = [], []
    for k, v in enumerate(y):
        if v is not None and not (isinstance(v, float) and (v != v)):  # filter None/NaN
            xs.append(k * x_step)
            ys.append(v)
    return xs, ys

def plot_energy_trend(energy_trajs, prefix, traj_name, num_trajs):
    plt.figure(figsize=(8, 5))
    for i, energy_list in enumerate(energy_trajs):
        xs, ys = _valid_xy(energy_list, x_step=1)
        plt.plot(xs, ys, label=f"Traj {i+1}", alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Optimized Energy')
    plt.xlabel("Time Step")
    plt.ylabel("Predicted Energy - Optimized Energy (eV)")
    plt.title("Energy Change Trend Over Time for random Trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}/{traj_name}/pred_energy_traj_{num_trajs}.png")
    plt.close()

def plot_dft_energy_trend(energy_trajs, prefix, traj_name, num_trajs, interval=500):
    plt.figure(figsize=(8, 5))
    for i, energy_list in enumerate(energy_trajs):
        xs, ys = _valid_xy(energy_list, x_step=interval)
        plt.plot(xs, ys, label=f"Traj {i+1}", alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Optimized Energy')
    plt.xlabel("Time Step")
    plt.ylabel("DFT Energy - Optimized Energy (eV)")
    plt.title("Energy Change Trend Over Time for random Trajectories (DFT)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}/{traj_name}/dft_energy_traj_{num_trajs}.png")
    plt.close()

def plot_combined_energy_trend(pred_energy_trajs, dft_energy_trajs, prefix, traj_name, num_trajs, interval=500):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axs[0].set_title("Predicted Energy Change Trend")
    for i, energy_list in enumerate(pred_energy_trajs):
        xs, ys = _valid_xy(energy_list, x_step=1)
        axs[0].plot(xs, ys, label=f"Traj {i+1}", alpha=0.7)
    axs[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Optimized Energy')
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Energy - Optimized (eV)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("DFT Energy Change Trend")
    for i, energy_list in enumerate(dft_energy_trajs):
        xs, ys = _valid_xy(energy_list, x_step=interval)
        axs[1].plot(xs, ys, label=f"Traj {i+1}", alpha=0.7)
    axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Optimized Energy')
    axs[1].set_xlabel(f"Time Step (Interval = {interval})")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    combined_path = f"{prefix}/{traj_name}/combined_energy_trend.png"
    plt.savefig(combined_path)
    plt.close()
    print(f"✅ Combined energy trend plot saved in: {combined_path}")

def read_optimimal_energy(prefix):
    print(f"Reading optimized energy from {prefix}")
    df = pd.read_csv("optimized.csv")
    # print(df)
    # Extract the energy values from the DataFrame
    df['Name'] = df['Name'].str.lower()
    # print(df['Name'])   
    for row in df.iterrows():
        name = row[1]['Name']
        print("Checking name:", name)
        energy = row[1]['Energy']
        if prefix == "bi11-3_samples":
            prefix = "bi11-3"
        print("Comparing with prefix:", prefix)
        if name == prefix:
            print(f"Found optimized energy for {prefix}: {energy}")
            return ast.literal_eval(energy)


def get_dft_energy(coord_trajs, atoms, charge, interval):
    """
    Get the energy of the optimized structure.
    Args:
        coord_trajs (list): List of atomic coordinates.
        atoms (list): List of atomic symbols.
        charge (int): Molecular charge.
        interval (int): DFT energy trend interval.
    Returns:
        energy (float): Energy of the optimized structure.
    """
    energy_list = []
    forces_list = []
    for i in range(0, len(coord_trajs), interval):
        energy, force = run_calc(atoms, coord_trajs[i], charge, True)
        energy_list.append(energy)
        forces_list.append(force)
    return energy_list, forces_list
def _one_dft_indexed(t):
    idx, atoms, coords, charge, use_dft, threads = t
    with set_threads(threads):
        E, F = run_calc(atoms, coords, charge, use_dft)
        return idx, E, F

def get_dft_energy_parallel(coord_traj, atoms, charge, interval, n_jobs=4, threads_per_job=4):
    """Evaluate DFT every `interval` frames in parallel."""
    # pick frames
    frames = list(range(0, len(coord_traj), interval))

    # restore original order (imap_unordered returns arbitrary order)
    # safer: re-run with enumerate index
    tasks2 = [ (idx, atoms, coord_traj[i], charge, True, threads_per_job)
               for idx, i in enumerate(frames) ]


    energies = [None]*len(frames)
    forces   = [None]*len(frames)
    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        for idx, E, F in pool.imap_unordered(_one_dft_indexed, tasks2, chunksize=1):
            energies[idx] = E
            forces[idx]   = F

    return frames, energies, forces
def get_label_list(traj):
    """Get the label list from a trajectory."""
    return [frame[-2] for frame in traj]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Separate batch trajectories.")
    parser.add_argument("--element", type=str, required=True, help="Element symbol (e.g., 'bi')")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the system (e.g., -2)")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms (e.g., 4)")
    parser.add_argument("--model_number", type=int, required=True, help="Model number (e.g., 25)")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to simulate")
    parser.add_argument("--synthesis", type=str, required=True, help="if the data is from synthesis")
    parser.add_argument("--dft_interval", type=int, default=500, help="DFT energy trend interval")
    # add near your argparse
    parser.add_argument("--syn_base", type=str, default="bi4", help="Base name for synthesis data")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel DFT jobs (processes)")
    parser.add_argument("--threads_per_job", type=int, default=4, help="OpenMP threads per DFT job")
    parser.add_argument("--project", type=str, default=".", help="Project subdirectory name")

    args = parser.parse_args()
    dft_interval = args.dft_interval
    if args.synthesis == "True" and args.syn_base == "bi4":
        prefix = f"{args.element}{args.num_atom}{args.charge}_samples"
    elif args.synthesis == "True" and args.syn_base != "bi4":
        prefix = f"{args.element}{args.num_atom}{args.charge}_samples_{args.syn_base}"
    else:
        prefix = f"{args.element}{args.num_atom}{args.charge}"
    traj_name = f"{args.model_number}_{args.steps}steps"
    output_dir  = f"{prefix}/{traj_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    project_name = args.project
    a = read_traj(f"{prefix}/{project_name}/{traj_name}_traj.pkl")
    all_traj = len(a)
    print(f"Total number of trajectories: {all_traj}")
    print("Load the trajectories...")
    energy_trajs = [get_energy_list(traj) for traj in a]
    coord_trajs = [get_coord_list(traj) for traj in a]
    # sample from trajs
    # sample from trajs
    label_list = [get_label_list(traj) for traj in a]
    print("label list example:", label_list[0][0])
    has_labels = any(any(label is not None for label in labels) for labels in label_list)
    print("Sampling the trajectories by label...")

    n_per_label = 5  # <-- number of trajectories per label

    # Step 1: extract one representative label per trajectory
    traj_labels = []
    for labels in label_list:
        valid = [l for l in labels if l is not None]
        traj_labels.append(valid[0] if valid else None)

    traj_labels = np.array(traj_labels)

    # Step 2: group trajectory indices by label
    label_to_indices = {}
    for idx, lbl in enumerate(traj_labels):
        if lbl is None:
            continue
        label_to_indices.setdefault(lbl, []).append(idx)

    print("Label distribution:")
    for lbl, idxs in label_to_indices.items():
        print(f"  Label {lbl}: {len(idxs)} trajectories")

    # Step 3: sample n_per_label trajectories for each label
    selected_indices = []

    rng = np.random.default_rng(seed=42)

    for lbl, idxs in label_to_indices.items():
        if len(idxs) <= n_per_label:
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=n_per_label, replace=False).tolist()
        selected_indices.extend(chosen)

    # Optional: shuffle final list
    rng.shuffle(selected_indices)

    print(f"✅ Selected {len(selected_indices)} trajectories in total")
    print(f"✅ Selected indices: {selected_indices}")

    # Select the sampled trajectories: Selected indices: [20,3,0,23,8]
    
    energy_trajs = [energy_trajs[i] for i in selected_indices]
    
    coord_trajs = [coord_trajs[i] for i in selected_indices]
    # energy_trajs = random.sample(energy_trajs, 150)
    # get the energy of the optimized structure
    optimized_energy = read_optimimal_energy(prefix)
    num_of_selected_trajs = len(coord_trajs)

    print("Optimized energy read:", optimized_energy)
    
    print(f"Optimized energy: {optimized_energy}")
    energy_trajs = [[e - optimized_energy[0] for e in energy] for energy in energy_trajs]
    print("Calculating DFT energy...")
    # dft_energy_force_list = [get_dft_energy(coord_trajs[i], ["Bi"] * args.num_atom, args.charge, dft_interval) for i in range(num_of_selected_trajs)]
    atoms = ["Bi"] * args.num_atom
    # Prepare containers
    dft_frames_list = []   # to store the indices of frames that got DFT evaluation (every 500 steps)
    dft_energy_list = []   # to store DFT energies
    dft_force_list = []    # to store DFT forces

    # Loop over each selected trajectory
    for coord_traj in coord_trajs:
        frames_i, E_i, F_i = get_dft_energy_parallel(
            coord_traj,
            atoms,
            args.charge,
            dft_interval,
            n_jobs=args.n_jobs,
            threads_per_job=args.threads_per_job,
        )
        dft_frames_list.append(frames_i)
        dft_energy_list.append(E_i)
        dft_force_list.append(F_i)


    print("✅ DFT calculations finished.")
    #print(dft_energy_list)
    # Save the DFT energy list
    with open(f"{prefix}/{traj_name}/dft_energy_list.pkl", "wb") as f:
        for i in range(num_of_selected_trajs):
            pickle.dump(dft_energy_list[i], f)
    with open(f"{prefix}/{traj_name}/dft_force_list.pkl", "wb") as f:
        for i in range(num_of_selected_trajs):
            pickle.dump(dft_force_list[i], f)
    dft_energy_list = [
        None if energy is None else [
            None if e is None else e[0] - optimized_energy[0]
            for e in energy
        ]
        for energy in dft_energy_list
    ]
    plot_energy_trend(energy_trajs, prefix, traj_name, num_of_selected_trajs)
    plot_dft_energy_trend(dft_energy_list, prefix, traj_name, num_of_selected_trajs)
    plot_combined_energy_trend(
    pred_energy_trajs=energy_trajs,
    dft_energy_trajs=dft_energy_list,
    prefix=prefix,
    traj_name=traj_name,
    num_trajs=num_of_selected_trajs,
    interval=dft_interval
)

    print(f"✅ Energy trend plot saved in: {output_dir}/pred_energy_traj.png")
    print(f"✅ DFT energy trend plot saved in: {output_dir}/dft_energy_traj.png")
    print(f"✅ combined prediction and DFT energy trend plot saved in: {output_dir}/combined_energy_trend.png")



