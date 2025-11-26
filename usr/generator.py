#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:27:15 2023

@author: chen
"""

from functools import partial
import gc
from openmm import unit
import numpy as np
import os, pickle
# from openmm.app import *
# from openmm import *
# from openmm.unit import *
from simtk import unit
import openmm
from usr.utils import get_specific_data, convert_to_1d_float_array, reconstruct_from_metadata
from usr.initial_pyg.functions.config import ConfigLoader
import copy
import random
from usr.utils import Molecule
import sys
import periodictable
from openmm import Vec3
import pandas as pd
from al_setting import AL_SETTING
def block_print():
    sys.stdout = None


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
        self.history = [[],]
        self.save_path = os.path.join(self.result_dir, f"generator_data_{rank}")
        self.temperature = 298.0 * unit.kelvin
        self.collision_rate = 1.0 / unit.picosecond
        self.timestep = 2.0 * unit.femtoseconds
        config = ConfigLoader("config.yaml")
        self.full_dataset = config['full_dataset']
        self.metadata = config['metadata']
        self.patience_threshold = config['patience_threshold']
        self.prefix = config['prefix']
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
        self.starting_point = 0
        self.external_force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')

        self.external_force.addPerParticleParameter('fx')
        self.external_force.addPerParticleParameter('fy')
        self.external_force.addPerParticleParameter('fz')
        self.num_generate = 0
        self.current_iteration = None
    

    def read_in_data(self, counter=0):
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
        Update the forces acting on each atom using the predicted forces.

        :param context: The OpenMM Context object managing the simulation state.
        :param force: The CustomExternalForce object used to apply custom forces.
        :param predicted_forces: A (4, 3) array of predicted forces, one row per atom.
                                Each row contains [Fx, Fy, Fz] force components.
        """
        # Loop over each atom and set the predicted force components

        EV_PER_ANGSTROM_TO_KJMOL_PER_NM = 96.485 * 10

        current_forces = -1 * predicted_forces * EV_PER_ANGSTROM_TO_KJMOL_PER_NM * unit.kilojoule_per_mole / unit.nanometer
        # current_forces = predicted_forces * unit.kilojoule_per_mole / unit.nanometer
        for i in range(predicted_forces.shape[0]):
            fx, fy, fz = current_forces[i]
            force.setParticleParameters(i, i, [fx, fy, fz])

            
        return force
        # Update the force in the context to apply the new forces
        # force.updateParametersInContext(context)
        

    def update(self, geometry):
        self.simulation.step(1)
        current_positions = self.simulation.context.getState(getPositions=True).getPositions()
        coords_distorded = np.array([pos.value_in_unit(unit.angstrom) for pos in current_positions])
        velosity = self.simulation.context.getState(getVelocities=True).getVelocities()
        # forces = self.simulation.context.getState(getForces=True).getForces()

        true_force_empty = np.zeros(coords_distorded.shape)
        traj = [
            coords_distorded, # pos
            geometry[1], #atom_numbers
            None, # true_energy
            true_force_empty, # true_forces
            geometry[4], # charge
            geometry[5], # pred_forces 
            None,   # pred_energy
            geometry[-2], # patience
            vec3_to_numpy(velosity)]
        if geometry[7][0] < 0 or geometry[7][1] < 0:
            raise ValueError(f"[FATAL] Negative patience injected: {geometry[7]}")

        # print('traj', traj)
        # data = convert_to_1d_float_array(traj)
        # print(data)

        return traj
    
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
        data_to_pred = None
        ordered = True
        
        # please notice that data_to_gene is intinilized to be None for the first iteration.
        ##### User Part #####

        if self.starting_point % 1000 == 0:
            print(f"MD has ran for {self.starting_point} steps")
        if data_to_gene is not None:
            iteration_marker = int(data_to_gene[0])
            sent = 0 if int(data_to_gene[1]) > 10 else int(data_to_gene[1])
            # print(f"Rank {self.rank} received data from passive learner, iteration_marker: {iteration_marker}, sent to oracle? {sent}")
            #print(f"[DEBUG] RANK {self.rank}: {self.num_generate} | len(flat_array): {len(data_to_gene)} | data_to_gene: {data_to_gene}")
            geometry = reconstruct_from_metadata(data_to_gene[2:], self.metadata, rank = f"generator {self.rank}")

            if self.starting_point >= 10000 or geometry[-2][0] > self.patience_threshold or geometry[-2][1] > self.patience_threshold:
                data_to_gene = None
                print(f"""Rank {self.rank} patience or step exceeded,
                      trajecory is reached {self.starting_point} steps, 
                    energy patience is {geometry[-2][0]}, 
                    rmsd patience is {geometry[-2][1]},
                    current model is at iteration {iteration_marker}, 
                    start to generate new trajectory""")
                self.starting_point = 0
                self.history.append([])  # start a new history for the new trajectory
            else:
                if self.current_iteration is not None and iteration_marker != self.current_iteration:
                    print(f"model update detected, current iteration: {self.current_iteration}, new iteration: {iteration_marker}")
                self.current_iteration = iteration_marker  # <- save for later use if needed
                force = self.update_forces(self.iterative_force, geometry[5])
                force.updateParametersInContext(self.simulation.context)
                data_to_pl = self.update(geometry)
                sent  = sent +1 if sent != 0 else 0
                self.num_generate += 1
                self.starting_point += 1
                data_to_pl.insert(0, [sent])
                self.history[-1].append([self.current_iteration, data_to_pl[0]])
                

                # self.counter += 1
                if self.counter > self.limit:
                    print('generation limit reached')
                    stop = True
            
                    print(f"continuing from iteration {iteration_marker}, patience: {geometry[-2]}, starting point: {self.starting_point}")
            # initialize data: first iteration or when patience is exceeded
        if data_to_gene is None:
            sent = 0
            if ordered == True:
                print(f'initializing data, force to number {self.counter} in the initial data')
                if self.counter <= self.init_length - 1:
                    data_to_pl = self.read_in_data(counter = self.counter)
                else:
                    print(f'counter {self.counter} exceeds the initial data length {self.init_length}, start to randomly start')
                    data_to_pl = self.read_in_data(counter = random.randint(0, self.init_length - 1))
            else:
                print("initializing data, PICKED randomly from initial data")
                data_to_pl = self.read_in_data(counter = random.randint(0, self.init_length - 1))
            molecule = self.set_up(data_to_pl)
            num_particles = molecule.system.getNumParticles()
            init_force = self.custom_force_initilize(molecule)
            self.iterative_force = self.update_forces(init_force, data_to_pl[3].numpy())
            molecule.system.addForce(self.iterative_force)
            integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
            print('set up simulation')
            self.simulation = openmm.app.Simulation(molecule.get_Topology(), molecule.get_System(), integrator)
            print('add initial positions')
            self.simulation.context.setPositions(data_to_pl[0].numpy() * 0.1)
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            
            data_to_pl = self.update(data_to_pl)
            self.history.append([self.current_iteration, data_to_pl[0],])
            self.counter += self.gene_procs
            self.num_generate += 1
            print('counter:', self.counter)
            data_to_pl.insert(0, [sent])
            
            #read in data when initial data size exceeded and start to generate new data from beginning by random structure



        if self.stop:
            stop = True
        if self.num_generate % 10000 == 0:
            print(f'{self.num_generate} Points are generated')
        # print(data_to_pl)
        data_to_pred  = convert_to_1d_float_array(data_to_pl)
        # print(f"[DEBUG] RANK {self.rank} after update and flattening | len(flat_array): {len(data_to_pred)} | data_to_pred: {data_to_pred}")
        # print('flattened data', data_to_pred)


        del data_to_pl
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
            self.history = self.history[-1:]
            print(f'saving progress, {len(self.history)} trajectories in history')
        


    def stop_run(self):
        """
        Stop the active learning workflow.
        """
        self.stop = True
        self.save_progress(self.stop)
        print('stop run')    
