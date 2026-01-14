import pickle
import h5py
import numpy as np
import argparse
import os

def convert_pkl_to_hdf5(pkl_path, h5_path):
    # Load pickled trajectories
    with open(pkl_path, "rb") as f:
        batch_trajs = pickle.load(f)
    print(f"✅ Loaded {len(batch_trajs)} trajectories from {pkl_path}")

    # Save to HDF5
    with h5py.File(h5_path, "w") as h5f:
        traj_grp = h5f.create_group("trajectories")
        for i, traj in enumerate(batch_trajs):
            try:
                coords = np.array([frame[0] for frame in traj])     # positions
                energys = np.array([frame[6] for frame in traj])    # predicted energies
                forces = np.array([frame[5] for frame in traj])     # predicted forces

                traj_grp.create_dataset(f"traj_{i}", data=coords, compression="gzip")
                traj_grp.create_dataset(f"energy_{i}", data=energys, compression="gzip")
                traj_grp.create_dataset(f"forces_{i}", data=forces, compression="gzip")
            except Exception as e:
                print(f"❌ Failed to write traj {i}: {e}")

    print(f"✅ Converted and saved to HDF5 at {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trajectory .pkl file to HDF5 format")
    parser.add_argument("--pkl", required=True, help="Path to input .pkl file")
    parser.add_argument("--out", required=True, help="Path to output .h5 file")
    args = parser.parse_args()

    convert_pkl_to_hdf5(args.pkl, args.out)
