#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:27:15 2023

@author: chen
"""

from functools import partial
import gc
# from openmm import unit
import numpy as np
import os, pickle

import openmm
import openmm.app as app
from openmm import unit, Vec3
from usr.utils import get_specific_data, convert_to_1d_float_array, reconstruct_from_metadata, compute_flat_length
from usr.initial_pyg.functions.config import ConfigLoader
import copy
import random
from usr.utils import Molecule
import sys
import periodictable
# from openmm import Vec3
import pandas as pd
from al_setting import AL_SETTING
from tqdm import tqdm


def reset_simulation_state(simulation, coords_ang):
    """
    Reinitialize positions/velocities in an existing Simulation object
    without creating a new Simulation instance.
    coords_ang must be in Å.
    """
    # convert Å → nm
    coords_nm = coords_ang * 0.1

    # reset positions
    simulation.context.setPositions(coords_nm)

    # reset velocities (fresh Maxwell distribution)
    simulation.context.setVelocitiesToTemperature(298.0 * unit.kelvin)

def vec3_to_numpy(vec3_list):
    # Convert list of Vec3 to a NumPy array
    return np.array([[v.x, v.y, v.z] for v in vec3_list])
class UserGene(object):
    """
    User defined Generator. Receive prediction from Passive Learner and generate new data points.
    """
    def __init__(self, rank, result_dir):
        """
        initilize the generator.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
        """
        self.rank = rank
        self.result_dir = result_dir
        ##### User Part ######
        print(f"Initializing Generator {rank}")
        self.counter = 0
        self.limit = float('inf')
        self.save_path = os.path.join(self.result_dir, f"generator_data_{rank}")
        self.temperature = 298.0 * unit.kelvin
        self.collision_rate = 1.0 / unit.picosecond
        self.timestep = 2.0 * unit.femtoseconds
        config = ConfigLoader("config.yaml")
        self.full_dataset = config['full_dataset']
        self.metadata = config['metadata']
        self.patience_threshold = config['patience_threshold']
        self.prefix = config['prefix']
        self.num_atom = config['num_atom']
        self.stop = False
        pred_procs = AL_SETTING["pred_process"]
        self.gene_procs = AL_SETTING["gene_process"]
        gene_start = 2 + pred_procs
        self.counter = rank - gene_start
        self.sample_count  = 0
        if self.full_dataset:
            raise NotImplementedError("full_dataset option is not implemented yet.")
            df = pd.read_csv('usr/initial_pyg/raw/bi0_parsed.csv')
            df = df[df['source'] == self.prefix]
            df.to_csv(f'{self.result_dir}/{self.prefix}.csv', index=False)
            self.path = f'{self.result_dir}/{self.prefix}.csv'
        else:
            self.path = f'usr/initial_pyg/samples/{self.prefix}/sample_{self.sample_count}/train.csv'
        self.init_length = self.get_lenth()
        print(f"Generator {rank} initialized with path: {self.path}, init_length: {self.init_length}")
        
        self.external_force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')

        self.external_force.addPerParticleParameter('fx')
        self.external_force.addPerParticleParameter('fy')
        self.external_force.addPerParticleParameter('fz')
        self.num_generate = 0
        self.current_iteration = None
        self.num_traj_per_gene = int(config['num_traj_per_gene'])
        self.starting_point = [0] * self.num_traj_per_gene
        self.cluster_data_length = compute_flat_length(self.metadata)
        self.trajs = [None] * self.num_traj_per_gene  # list of per-trajectory dicts
        self.history = [[] for _ in range(self.num_traj_per_gene)]
    
    def set_up_simulations(self, data_batch):
        simulators = []
        for idx, data in enumerate(data_batch):
            try:
                molecule = self.set_up(data)
                init_force = self.custom_force_initilize(molecule)
                molecule.system.addForce(init_force)
                # PBC not needed, skip box vectors
                platform = openmm.Platform.getPlatformByName('CPU')
                # print(f"Setting up simulation #{idx} on platform: {platform.getName()}")
                integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
                # print(f"Integrator created: Langevin at {self.temperature}, collision rate {self.collision_rate}, timestep {self.timestep}")
                simulation = app.Simulation(molecule.get_Topology(), molecule.get_System(), integrator, platform)
                # print("Simulation object created.")
                coords = np.asarray(data[0], dtype=float)
                coords_nm = coords * 0.1
                simulation.context.setPositions(coords_nm * unit.nanometer)
                print("Initial positions set.")
                simulation.context.setVelocitiesToTemperature(self.temperature)
                print("Initial velocities set.")

                simulators.append((simulation, init_force))
                print(f"✅Rank {self.rank} Simulation #{idx} set up successfully.")
            except Exception as e:
                print(f"❌ Failed to set up simulation #{idx}: {e}")
                continue
        return simulators
    def read_in_data(self, counter):
        print(f'Rank {self.rank} reading in data number {counter} from {self.path} ')
        return get_specific_data(self.path, counter)
    def get_lenth(self):
        return sum(1 for _ in open(self.path)) -1

    def set_up(self, data):
        atom_numbers = data[1]
        atom_types = [periodictable.elements[i].symbol for i in atom_numbers.tolist()]
        coordinates = data[0]

        molecule = Molecule(atom_types=atom_types, coordinates=coordinates)
        return molecule


    def random_strcuture_in_space(self, original_coord):
        shape = original_coord.shape
        coords_distorded = original_coord + np.random.randn(shape[0], shape[1]) * 0.001
        # print(type(coords_distorded))
        return coords_distorded
    def custom_force_initilize(self, molecule):
        force  = copy.deepcopy(self.external_force)
        # for i in range(molecule.system.getNumParticles()):
        #     force.addParticle(i, [0.0, 0.0, 0.0])    # initialize the force to 0
        # return force
        for i in range(molecule.system.getNumParticles()):

            force.addParticle(i, [0, 0, 0])

        return force
        

    def update_forces(self, force, predicted_forces):
        """
        Update the forces in CustomExternalForce from predicted forces in eV/Å.
        """
        num_particles = self.num_atom
        if predicted_forces.shape[0] != num_particles:
            raise ValueError(
                f"predicted_forces has {predicted_forces.shape[0]} atoms, "
                f"but system has {num_particles}"
            )

        EV_PER_ANG_TO_KJMOL_PER_NM = 96.485 * 10.0

        # convert to kJ/mol/nm as plain floats
        current_forces = -1.0 * predicted_forces * EV_PER_ANG_TO_KJMOL_PER_NM  # (N,3)
        if not np.all(np.isfinite(current_forces)):
            raise ValueError("Non-finite forces encountered in update_forces")

        for i in range(num_particles):
            fx, fy, fz = current_forces[i]
            force.setParticleParameters(i, i, [float(fx), float(fy), float(fz)])

        return force



    def update(self, simulation, geometry):
        simulation.step(1)
        state_pos = simulation.context.getState(getPositions=True)
        state_vel = simulation.context.getState(getVelocities=True)

        current_positions = state_pos.getPositions()
        coords_distorded = np.array([pos.value_in_unit(unit.angstrom) for pos in current_positions])

        velosity = state_vel.getVelocities()
        true_force_empty = np.zeros(coords_distorded.shape)

        traj = [
            coords_distorded,          # pos
            geometry[1],               # atom_numbers
            None,                      # true_energy
            true_force_empty,          # true_forces
            geometry[4],               # charge
            geometry[5],               # pred_forces
            None,                      # pred_energy
            geometry[-2],              # patience
            vec3_to_numpy(velosity),   # velocities
        ]

        if geometry[7][0] < 0 or geometry[7][1] < 0:
            raise ValueError(f"[FATAL] Negative patience injected: {geometry[7]}")

        return traj



    def restart_traj(self, i):
        """
        Restart trajectory i safely:
        - do NOT allocate new Simulation()
        - reuse existing simulation + external force
        - reset coords/vels/steps
        """

        # pick a new geometry from initial dataset
        random.seed(self.rank) 
        idx = random.randint(0, self.init_length - 1)
        geom = self.read_in_data(counter=idx)
        coords_ang = np.asarray(geom[0], dtype=float)

        traj_state = self.trajs[i]
        simulation = traj_state["simulation"]
        force      = traj_state["force"]

        # reset simulation state IN PLACE
        reset_simulation_state(simulation, coords_ang)

        # reset counters
        traj_state["steps"] = 0
        self.starting_point[i] = 0

        # reset force parameters to zero
        N = len(coords_ang)
        zero_force = np.zeros((N, 3), dtype=float)
        self.update_forces(force, zero_force)
        force.updateParametersInContext(simulation.context)

        # produce a fresh MD step output
        sample = self.update(simulation, geom)

        return geom, sample

    def generate_new_data(self, data_to_gene):
        """
        Generate new data point for passive learner based on data_to_gene.
        
        Args:
            data_to_gene (1-D numpy.ndarray or None): data from passive learner through EXCHANGE process. (from UserModel.predict())
            
        Returns:
            stop (bool): flag to stop the active learning workflow. True for stop.
            data_to_pred (1-D numpy.ndarray): data to passive learner through EXCHANGE process. (to UserModel.predict())
        """
        stop = False
        data_to_pl = []  # list of traj samples (one per trajectory)

        # ---------- CASE 1: FIRST CALL, no previous geometries ----------
        if data_to_gene is None:
            sent = 0
            print(f"initializing data, PICKED randomly from initial data, "
                f"{self.num_traj_per_gene} trajectories")
            random.seed(self.rank) 
            indices = random.sample(range(0, self.init_length), self.num_traj_per_gene)
            data_batch = [self.read_in_data(counter=i) for i in indices]

            sims = self.set_up_simulations(data_batch)

            for j, ((simulation, force), geom) in enumerate(zip(sims, data_batch)):
                # store per-trajectory state
                self.trajs[j] = {
                    "simulation": simulation,
                    "force": force,
                    "steps": 0,
                }
                self.starting_point[j] = 0

                sample = self.update(simulation, geom)
                sample = convert_to_1d_float_array(sample)
                sample = np.concatenate(([float(sent)], sample))

                data_to_pl.append(sample)

            data_to_pred = np.concatenate(data_to_pl, axis=0)
            return stop, data_to_pred

        # ---------- CASE 2: SUBSEQUENT CALLS WITH geometries ----------
        # at this point data_to_gene is a 1D array:
        # [iteration_marker, sent0, flat0..., sent1, flat1..., ...]
        iteration_marker = int(data_to_gene[0])
        body = data_to_gene[1:]

        # each traj contributes (1 sent flag + cluster_data_length) entries
        per_traj_len = self.cluster_data_length + 1
        if len(body) % per_traj_len != 0:
            raise ValueError(
                f"Body length {len(body)} not divisible by per_traj_len={per_traj_len}"
            )
        num_traj = len(body) // per_traj_len
        if num_traj != self.num_traj_per_gene:
            raise ValueError(
                f"Received {num_traj} trajectories from PL but generator expects "
                f"{self.num_traj_per_gene}"
            )

        body_2d = np.reshape(body, (num_traj, per_traj_len))

        for i in range(num_traj):
            row = body_2d[i]
            sent_i_raw = int(row[0])
            sent_i = 0 if sent_i_raw > 10 else sent_i_raw

            flat_geom = row[1:]
            geometry = reconstruct_from_metadata(
                flat_geom, self.metadata, rank=f"generator {self.rank}"
            )

            # print every 1000 steps for this trajectory
            if self.starting_point[i] % 1000 == 0 and self.starting_point[i] != 0:
                print(f"Rank {self.rank}, traj {i}: MD has run for "
                    f"{self.starting_point[i]} steps")

            # check patience / max steps for THIS trajectory
            if (self.starting_point[i] >= 100000 or
                geometry[-2][0] > self.patience_threshold or
                geometry[-2][1] > self.patience_threshold):

                print(f"""Rank {self.rank} traj {i}: patience or step exceeded,
        steps = {self.starting_point[i]},
        energy patience = {geometry[-2][0]},
        rmsd patience = {geometry[-2][1]},
        model iteration = {iteration_marker},
        restarting trajectory {i}""")

                # restart only this trajectory
                new_geom, sample = self.restart_traj(i)
                sent_i = 0  # on restart, sent flag resets
            else:
                # normal MD step for this trajectory
                if self.current_iteration is not None and iteration_marker != self.current_iteration:
                    print(f"model update detected, current iteration: {self.current_iteration}, "
                        f"new iteration: {iteration_marker}")
                self.current_iteration = iteration_marker

                traj_state = self.trajs[i]
                sim = traj_state["simulation"]
                force = traj_state["force"]

                # update external force with predicted forces from PL
                force = self.update_forces(force, geometry[5])
                force.updateParametersInContext(sim.context)

                try:
                    sample = self.update(sim, geometry)
                    traj_state["steps"] += 1
                    self.starting_point[i] += 1
                except Exception as e:
                    # EXPLOSION HANDLING for THIS traj only
                    print(f"Trajectory {i} exploded in MD step: {e}. Restarting...")
                    new_geom, sample = self.restart_traj(i)
                    sent_i = 0

            self.history[i].append([self.current_iteration, [sent_i, sample]])
            sample = convert_to_1d_float_array(sample)
            sample = np.concatenate(([float(sent_i)], sample))
            
            data_to_pl.append(sample)

        # global generation counters
        self.num_generate += len(data_to_pl)
        if self.counter > self.limit:
            print("generation limit reached")
            stop = True

        if self.stop:
            stop = True

        if self.num_generate % 10000 == 0:
            print(f'{self.num_generate} Points are generated')

        data_to_pred =  np.concatenate(data_to_pl, axis=0)

        if len(self.history[-1]) % 100000 == 0:
            self.save_progress()

        return stop, data_to_pred

    def save_progress(self,stop_run = False):
        """
        Save the current state and progress.
        """
        ##### User Part #####
        
        # m = 'ab' if os.path.exists(self.save_path) else 'wb'
        # with open(self.save_path, m) as fh:
        #     if len(self.history) > 1:
        #         pickle.dump(self.history[:-1], fh)
        #     else:
        #         pickle.dump(self.history[0], fh)
        #         # print('save progress:', self.history[0])
        #     self.history = self.history[-1:]
        # save th length of each trajectory in history
        if stop_run:
            print(f'saving progress and stopping run, {len(self.history)} trajectories in history')
            m = 'ab' if os.path.exists(self.save_path) else 'wb'
            with open(self.save_path, m) as fh:
                if len(self.history) > 1:
                    pickle.dump(self.history[-1], fh)
                else:
                    pickle.dump(self.history[0], fh)
                    # print('save progress:', self.history[0])
                
        else:
            self.history = [[] for _ in range(self.num_traj_per_gene)]
            print(f'saving progress, history reset for {self.num_traj_per_gene} trajectories')
        


    def stop_run(self):
        """
        Stop the active learning workflow.
        """
        self.stop = True
        self.save_progress(self.stop)
        print('stop run')    
