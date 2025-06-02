
from unittest import result
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
import copy
from openmm import Vec3

from usr.initial_pyg.functions.config import ConfigLoader

from tqdm import tqdm
import argparse
from initial_pyg.evaluation import evaluate
import h5py
def run_calc(input_to_orcl):
    """
    Run Oracle computation.
    Args:
        input: [
            coords_distorded, # pos
            geometry[1], #atom_numbers
            None, # true_energy
            true_force_empty, # true_forces
            geometry[4], # charge
            vec3_to_numpy(forces), # pred_forces 
            None,   # pred_energy
            geometry[-2], # patience
            vec3_to_numpy(velosity)]

    Returns:
        energy
    """
    ##### User Part #####
    # print('input_to_orcl', len(input_to_orcl))
    # input_to_orcl = reconstruct_from_metadata(input_to_orcl, self.meta_data)
    # #print('input_to_orcl', input_to_orcl)
    # atoms = atomic_number_to_symbol(input_to_orcl[1].tolist())
    # # atoms_list = re.sub(r'\b(\w+)\b', r"'\1'", atoms)
    

    time1 = time.time()
    settings = GeneralSettings(mp_settings=MultiProcessingSettings(mp_active=True, number_of_workers=1),
                            output_dir_path=f"{current_path}/results/TestRun/""test_output",
                            input_file_path=f"{current_path}/results/TestRun/",
                            delete_run_dir=True,
                            load_from_file=False,
                            coords=input_to_orcl[0],
                            elements=['Bi','Bi','Bi','Bi'])

    # xtb_settings = XTBSettings(binary_path="/home/yumeng/xtb-6.6.0/bin/xtb", charge= input_to_orcl[4], solvent = "Aniline", iterations = 500, accuracy = 500)

    turbomol_settings = TurbomolSettings(basis="dhf-TZVP ", 
                                    functional="tpss", 
                                    method="ridft",
                                    input_in_angstrom=True, 
                                    use_cosmo = True, 
                                    epsilon = 'infinity', 
                                    charge = input_to_orcl[4])

    # xtb_api = XTBApi(general_settings=settings, xtb_settings=xtb_settings)

    turbomol_api = TurbomolApi(general_settings=settings, turbomol_settings=turbomol_settings)
    work_path = os.getcwd()
    try:
        xtb_energies, xtb_forces = turbomol_api.get_energy_and_gradient()
        # input_to_orcl[2] = torch.tensor(xtb_energies)
        # input_to_orcl[3] = torch.tensor(xtb_forces[0])
        # #orcl_calc_res = convert_to_1d_float_array(input_to_orcl)
        # orcl_calc_res = torch.cat((torch.tensor(xtb_energies), torch.tensor(xtb_forces[0]).view(-1) ))
        # orcl_calc_res = orcl_calc_res.numpy()
        orcl_calc_res = [xtb_energies, xtb_forces]

    except:
        orcl_calc_res = None
        os.chdir(work_path)
        # Save the 'fail' object to a file
        with open('./results/xtbfail', 'ab') as f:  # 'ab' means append binary
            pickle.dump(orcl_calc_res, f)
        print('xtb failed', orcl_calc_res)

    # turbomol_energies = xtb_api.vibrational_analysis()

    #print("Time: ", time1 - time.time())
    # print("XTB Energies: ", xtb_energies)
    
    print('oracle is done')
    ##### User Part END #####
    return orcl_calc_res

def vec3_to_numpy(vec3_list):
    # Convert list of Vec3 to a NumPy array
    return np.array([[v.x, v.y, v.z] for v in vec3_list])
class Generate_TrajsBatch(object):
    def __init__(self, data_batch, result_path, model_number, dir_path = None):
        self.dir_path = dir_path
        # Initialize with a batch of data
        self.data_batch = data_batch
        self.model_number = model_number
        self.temperature = 298.0 * unit.kelvin
        self.collision_rate = 91.0 / unit.picosecond
        self.timestep = 2.0 * unit.femtoseconds
        self.external_force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')         
        self.external_force.addPerParticleParameter('fx')
        self.external_force.addPerParticleParameter('fy')
        self.external_force.addPerParticleParameter('fz')
        self.config = ConfigLoader("../../config.yaml")
        PATH = f'{result_path}/model_{model_number}.pt'
        self.model = torch.load(PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)



    
    def set_up(self, data):
        atom_numbers = data[1]
        atom_types = [periodictable.elements[i].symbol for i in atom_numbers]
        coordinates = torch.tensor(data[0])
        reduced_density = 0.05

        sigma = 3.4 * unit.angstroms
        nparticles = len(atom_numbers)
        number_density = reduced_density / sigma**3
        volume = nparticles * (number_density ** -1)
        box_edge = volume ** (1. / 3.)
        box_vectors = np.diag([box_edge/unit.angstrom for i in range(3)]) * unit.angstroms
        molecule = Molecule(atom_types=atom_types, coordinates=coordinates)
        return molecule, box_vectors
    def custom_force_initilize(self, molecule):
        force  = copy.deepcopy(self.external_force)
        # for i in range(molecule.system.getNumParticles()):
        #     force.addParticle(i, [0.0, 0.0, 0.0])    # initialize the force to 0
        # return force
        for i in range(molecule.system.getNumParticles()):

            force.addParticle(i, [0, 0, 0])

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
            vec3_to_numpy(forces), # pred_forces 
            geometry[-3],   # pred_energy
            geometry[-2], # patience
            vec3_to_numpy(velosity)]
        return traj

    def set_up_simulations(self):
        simulators = []
        forces = []
        for data in self.data_batch:
            molecule, box_vectors = self.set_up(data)
            init_force = self.custom_force_initilize(molecule)
            molecule.system.addForce(init_force)
            # molecule.system.setDefaultPeriodicBoxVectors(*box_vectors)
            
            integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
            simulation = openmm.app.Simulation(molecule.get_Topology(), molecule.get_System(), integrator)
            simulation.context.setPositions(data[0] * 0.1)
            simulation.context.setVelocitiesToTemperature(self.temperature)
            
            simulators.append((simulation, init_force))
        return simulators

    def generate_batch_trajs(self, steps):
        print('Start batch simulation')
        simulators = self.set_up_simulations()
        trajs = []
        for i in range(len(simulators)):
            trajs.append([self.data_batch[i]])

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

    def get_true_energy_and_forces(self, traj_list):
        '''
        Args:
            traj_list: list of traj
        Returns: 
            traj_list with true energy and forces
        '''
        for i in tqdm(range(len(traj_list)), desc="Generating DFT labels"):
            traj_list[i][2], traj_list[i][3] = run_calc(traj_list[i])
        return traj_list

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
    args = parser.parse_args()

    # Extract arguments
    element = args.element
    charge = args.charge
    num_atom = args.num_atom
    model_number = args.model_number
    steps = args.steps


    # Generate file prefix and result path
    prefix = f"{element}{num_atom}{charge}_samples"
    print(prefix)
    result_path = f'../../results/{prefix}'
    print(result_path)
    df = pd.read_csv(f'{result_path}/added_data.csv', delimiter=',', on_bad_lines='skip')

    df = df[df['type'] ==  'val'].reset_index(drop=True)
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
    # Define the directory path
    dir_path = f'../../trajs/{prefix}'

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    traj_gen = Generate_TrajsBatch(data_batch, result_path, model_number, dir_path)
    batch_trajs = traj_gen.generate_batch_trajs(steps)
    print("finished generating batch trajectories")

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

    print(f"✅ Saved {len(batch_trajs)} trajectories to {h5_path}")
    print('Batch trajectory generation complete and saved.')
