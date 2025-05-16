import numpy as np
import pandas as pd
# from xtb.interface import Calculator
# from xtb.utils import get_method
import numpy as np
# from xtb.utils import get_solvent
import time
import sys
import pickle
import openmm
import h5py
sys.path.append("../../../quantum_chem_python/")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../initial_pyg")
import os
current_path = os.getcwd()
import re
from quantum_chem_python.api.settings import GeneralSettings, MultiProcessingSettings, XTBSettings, TurbomolSettings
from quantum_chem_python.api.turbomol.turbomol_api import TurbomolApi
from openmm import unit
from quantum_chem_python.api.xtb.xtb_api import XTBApi
from utils import Molecule, convert_to_data_object
import torch
import ast
import periodictable

from usr.initial_pyg.functions.config import ConfigLoader
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import argparse
from initial_pyg.evaluation import evaluate

def vec3_to_numpy(vec3_list):
    # Convert list of Vec3 to a NumPy array
    return np.array([[v.x, v.y, v.z] for v in vec3_list])
class Generate_TrajsBatch(object):
    def __init__(self, data_batch, result_path, model_number, prefix):
        # Initialize with a batch of data
        self.data_batch = data_batch
        self.temperature = 298.0 * unit.kelvin
        self.collision_rate = 91.0 / unit.picosecond
        self.timestep = 2.0 * unit.femtoseconds
        self.external_force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')         
        self.external_force.addPerParticleParameter('fx')
        self.external_force.addPerParticleParameter('fy')
        self.external_force.addPerParticleParameter('fz')
        self.config = ConfigLoader("../../config.yaml")
        self.model_number = model_number
        self.prefix = prefix
        PATH = f'{result_path}/model_{model_number}.pt'
        self.model = torch.load(PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔮 Loaded model {model_number} from {PATH} on device {self.device}")
        self.model.to(self.device)



    
    def set_up(self, data):
        atom_numbers = data[1]
        atom_types = [periodictable.elements[i].symbol for i in atom_numbers]
        coordinates = torch.tensor(data[0])
        coordinates -= coordinates.mean(axis=0)  # center at origin
        # reduced_density = 0.05

        # sigma = 3.4 * unit.angstroms
        # nparticles = len(atom_numbers)
        # number_density = reduced_density / sigma**3
        # volume = nparticles * (number_density ** -1)
        # box_edge = volume ** (1. / 3.)
        # box_vectors = np.diag([box_edge/unit.angstrom for i in range(3)]) * unit.angstroms
        molecule = Molecule(atom_types=atom_types, coordinates=coordinates)
        return molecule
    def custom_force_initilize(self, molecule):
        force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')
        force.addPerParticleParameter('fx')
        force.addPerParticleParameter('fy')
        force.addPerParticleParameter('fz')
        for i in range(molecule.system.getNumParticles()):
            force.addParticle(i, [0.0, 0.0, 0.0])
        return force

        
    def update_forces(self,force, predicted_forces):
        """
        Update the forces acting on each atom using the predicted forces.

        :param context: The OpenMM Context object managing the simulation state.
        :param force: The CustomExternalForce object used to apply custom forces.
        :param predicted_forces: A (4, 3) array of predicted forces, one row per atom.
                                Each row contains [Fx, Fy, Fz] force components.
        """
        # Loop over each atom and set the predicted force components
        conversion_factor = 418.4  # kcal/mol/Å -> kJ/mol/nm
        current_forces = -1 * predicted_forces * conversion_factor * unit.kilojoule_per_mole / unit.nanometer
        # current_forces = predicted_forces * unit.kilojoule_per_mole / unit.nanometer
        for i in range(predicted_forces.shape[0]):
            fx, fy, fz = current_forces[i]
            # fx = fx.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            # fy = fy.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            # fz = fz.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)
            force.setParticleParameters(i, i, [fx, fy, fz])
        return force
    
    def update(self, geometry, simulation):
        simulation.step(1)
        # print(geometry[0])
        current_positions = simulation.context.getState(getPositions=True).getPositions()
        # print(current_positions)
        # coords_distorded = np.array(current_positions)
        coords_distorded = np.array([pos.value_in_unit(unit.angstrom) for pos in current_positions])
        velosity = simulation.context.getState(getVelocities=True).getVelocities()
        forces = simulation.context.getState(getForces=True).getForces()
        true_force_empty = np.zeros(coords_distorded.shape)
        traj = [
            coords_distorded, # pos
            geometry[1], #atom_numbers
            None, # true_energy
            true_force_empty, # true_forces
            geometry[4], # charge
            geometry[5], # pred_forces 
            geometry[-3],   # pred_energy
            geometry[-2], # patience
            vec3_to_numpy(velosity)]
        return traj

    def set_up_simulations(self, data_batch):
        simulators = []
        for idx, data in enumerate(tqdm(data_batch, desc="Setting up simulations")):
            try:
                molecule = self.set_up(data)
                init_force = self.custom_force_initilize(molecule)
                molecule.system.addForce(init_force)
                # PBC not needed, skip box vectors
                platform = openmm.Platform.getPlatformByName('CPU')
                # integrator = openmm.VerletIntegrator(self.timestep)
                integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
                simulation = openmm.app.Simulation(molecule.get_Topology(), molecule.get_System(), integrator, platform)
                simulation.context.setPositions(data[0] * 0.1)
                simulation.context.setVelocitiesToTemperature(self.temperature)

                simulators.append((simulation, init_force))
            except Exception as e:
                print(f"❌ Failed to set up simulation #{idx}: {e}")
                continue
        return simulators


    def generate_batch_trajs(self, steps):
        print('Start batch simulation')
        start = time.time()
        simulators = self.set_up_simulations()
        print(f"Simulations set up in {time.time() - start:.2f} seconds")
        trajs = []
        for i in range(len(simulators)):
            trajs.append([self.data_batch[i]])
        print('Start generating batch trajectories')

        # Simulation loop
        for step in tqdm(range(steps), desc="Generating batch trajectories"):
            # Collect current coordinates from all simulations
            all_coords = []
            # for sim, _, _, _ in simulators:
            #     state = sim.context.getState(getPositions=True)
            #     coords = np.array([[pos.x, pos.y, pos.z] for pos in state.getPositions()])
            #     all_coords.append(coords)

            # Predict forces and energies using the model
            current_step_data = [traj[step] for traj in trajs]
            energies, forces_batch = self.get_predicted_energy_and_forces(current_step_data)

            # Update forces for each simulation
            for i, (sim, force) in enumerate(simulators):
                new_force = self.update_forces(force, forces_batch[i])
                current_step_data[i][-4] = forces_batch[i]
                current_step_data[i][-3] = energies[i]

                force.updateParametersInContext(sim.context)    
                next_step_data = self.update(current_step_data[i], sim)

                # Collect trajectory data
                trajs[i].append(next_step_data)
            if step % 1000 == 0:
                print(f"Step {step} completed.")
                # save the current state of trajs to a file
                with open(f'{self.dir_path}/{self.model_number}_{step}steps_traj.pkl', 'wb') as f:
                    pickle.dump(trajs, f)

        return trajs

    def generate_batch_trajs_setup_in_batch(self, steps, chunk_size=100, traj_dir='../../trajs/'):
        traj_dir = os.path.join(traj_dir, self.prefix)
        print('Start batch simulation')
        os.makedirs(traj_dir, exist_ok=True)

        all_trajs = []
        simulators = []
        data_slices = []

        # Step 1: Batch-wise simulator setup
        for batch_start in range(0, len(self.data_batch), chunk_size):
            data_batch = self.data_batch[batch_start: batch_start + chunk_size]
            print(f"🧪 Setting up batch {batch_start} to {batch_start + len(data_batch)}")

            try:
                sims = self.set_up_simulations(data_batch)
                simulators.extend(sims)
                data_slices.extend(data_batch)
            except Exception as e:
                print(f"❌ Failed during setup of batch {batch_start}: {e}")
                continue

        # Step 2: Initialize trajectory containers
        trajs = [[data] for data in data_slices]

        # Step 3: Run all simulations together
        print(f"🚀 Starting full trajectory generation for {len(simulators)} systems")
        for step in tqdm(range(steps), desc="Generating trajectories"):
            current_step_data = [traj[step] for traj in trajs]
            energies, forces_batch = self.get_predicted_energy_and_forces(current_step_data)

            for i, (sim, force) in enumerate(simulators):
                try:
                    self.update_forces(force, forces_batch[i])
                    current_step_data[i][-4] = forces_batch[i]
                    current_step_data[i][-3] = energies[i]

                    force.updateParametersInContext(sim.context)
                    next_data = self.update(current_step_data[i], sim)
                    trajs[i].append(next_data)
                except Exception as e:
                    print(f"⚠️ Failed at step {step} for traj {i}: {e}")
            if step % 1000 == 0:
                print(f"Step {step} completed.")
                # save the current state of trajs to a file
                with open(f'{traj_dir}/{self.model_number}_{step}steps_tmp_traj.pkl', 'wb') as f:
                    pickle.dump(trajs, f)

        return trajs

    def get_predicted_energy_and_forces(self, data_list):
        data_list = convert_to_data_object(data_list)
        for data in data_list:
            data.z = torch.tensor(data.z)
        # dataset = retrain_dataset(data_list, transforms=self.transform)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        y_pred, force_pred, _, _ = evaluate(model=self.model, eval_dataset=data_list, batch_size=32, default_dtype= 'float64', device=self.device, compute_stress=False, return_contributions=False)
        num_data = len(data_list)

        force_pred = np.stack(force_pred).reshape(num_data, -1, 3)
        return y_pred, force_pred


def convert_to_numpy_array(value):
    try:
        # Replace the 'tensor' part and the dtype specification to make it a valid list
        value = value.replace('tensor(', '').replace('dtype=torch.float32', '').replace(')', '').strip()
        # Use ast.literal_eval to safely evaluate the string to a list
        value = ast.literal_eval(value)
        # Convert the list to a NumPy array
        return np.array(value)
    except (ValueError, SyntaxError):
        return value
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate batch trajectories.")
    parser.add_argument("--element", type=str, required=True, help="Element symbol (e.g., 'bi')")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the system (e.g., -2)")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms (e.g., 4)")
    parser.add_argument("--model_number", type=int, required=True, help="Model number (e.g., 25)")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to simulate")
    parser.add_argument("--synthesis", type=str, default='False', help="Synthesis or not (e.g., True or False)")
    args = parser.parse_args()


    # Extract arguments
    element = args.element
    charge = args.charge
    num_atom = args.num_atom
    model_number = args.model_number
    steps = args.steps
    synthesis = args.synthesis
    # Convert synthesis to boolean
    if synthesis.lower() == 'true':
        synthesis = True
        prefix = f"{element}{num_atom}{charge}_samples"
    elif synthesis.lower() == 'false':
        synthesis = False
        prefix = f"{element}{num_atom}{charge}"
    else:
        raise ValueError("Invalid value for synthesis. Use 'True' or 'False'.")
    
    


    # Generate file prefix and result path
    
    # prefix = f"{element}{num_atom}{charge}_samples"
    print(prefix)
    result_path = f'../../results/{prefix}'
    print(result_path)
    validation_set = pd.read_csv(f'../initial_pyg/full_data_charge_embed/bi0_logs/bi0.csv')
    len_org_val = len(validation_set)
    print(len_org_val)
    # Load full added dataset
    df = pd.read_csv(f'{result_path}/added_data.csv', delimiter=',', on_bad_lines='skip')

    # Keep only validation-type entries
    df = df[df['type'] == 'val'].reset_index(drop=True)

    # Remove the first len_org_val entries (they are the original validation)
    df = df.iloc[len_org_val:].reset_index(drop=True)

    print(f"Newly generated validation samples: {len(df)}")
    df['node_feature'] = df['node_feature'].apply(lambda x: convert_to_numpy_array(x)) 

    df['node_feature'] = [x.reshape(num_atom, 3) for x in df['node_feature']]
    df['atoms'] = df['atoms'].apply(lambda x: convert_to_numpy_array(x))
    atom_number = df['atoms'][0]
    print(atom_number)
    coordinates_batch = df['node_feature'].to_list()
    data_batch = [
        [coords, atom_number, None, None, charge, None, None, 0, None]
        for coords in coordinates_batch
    ] 
    

    data_batch = data_batch[:200]
    traj_gen = Generate_TrajsBatch(data_batch, result_path, model_number, prefix)

    try:
        batch_trajs = traj_gen.generate_batch_trajs_setup_in_batch(steps, chunk_size=200, traj_dir=f'trajs_{prefix}')
    except KeyboardInterrupt:
        print("🛑 Caught Ctrl+C — cleaning up!") 

    # Define the directory path
    dir_path = f'../../trajs/{prefix}'

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    with open(f'{dir_path}/{model_number}_{steps}steps_traj.pkl', 'wb') as f:
        pickle.dump(batch_trajs, f)
    h5_path = f'{dir_path}/{model_number}_{steps}steps_traj.h5'
    with h5py.File(h5_path, "w") as h5f:
        traj_grp = h5f.create_group("trajectories")
        for i, traj in enumerate(batch_trajs):
            coords = np.array([frame[0] for frame in traj])  # frame[0] = positions
            energys = np.array([frame[6] for frame in traj])  # frame[6] = predicted energy
            forces = np.array([frame[5] for frame in traj])  # frame[5] = predicted forces
            # Save the trajectory da
            traj_grp.create_dataset(f"traj_{i}", data=coords, compression="gzip")
            traj_grp.create_dataset(f"energy_{i}", data=energys, compression="gzip")
            traj_grp.create_dataset(f"forces_{i}", data=forces, compression="gzip")
            # Save the trajectory data to HDF5
            # traj_grp.create_dataset(f"traj_{i}", data=traj, compression="gzip")

    print('Batch trajectory generation complete.')
