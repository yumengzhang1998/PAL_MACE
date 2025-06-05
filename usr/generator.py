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
        self.collision_rate = 91.0 / unit.picosecond
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
        if self.full_dataset:
            df = pd.read_csv('usr/initial_pyg/raw/bi0_parsed.csv')
            df = df[df['source'] == self.prefix]
            df.to_csv(f'{self.result_dir}/{self.prefix}.csv', index=False)
            self.path = f'{self.result_dir}/{self.prefix}.csv'
        else:
            self.path = f'usr/initial_pyg/raw/{self.prefix}_parsed.csv'
        self.init_length = self.get_lenth()
        self.starting_point = 0
        self.external_force = openmm.CustomExternalForce('fx * x + fy * y + fz * z')

        self.external_force.addPerParticleParameter('fx')
        self.external_force.addPerParticleParameter('fy')
        self.external_force.addPerParticleParameter('fz')
        self.num_generate = 0
    

    def read_in_data(self, counter=0):
        return get_specific_data(self.path, counter)
    def get_lenth(self):
        return sum(1 for _ in open(self.path)) -1

    def set_up(self, data):
        atom_numbers = data[1]
        atom_types = [periodictable.elements[i].symbol for i in atom_numbers.tolist()]
        coordinates = data[0]
        reduced_density = 0.05

        sigma = 3.4 * unit.angstroms
        nparticles = len(atom_numbers)
        number_density = reduced_density / sigma**3
        volume = nparticles * (number_density ** -1)
        box_edge = volume ** (1. / 3.)
        box_vectors = np.diag([box_edge/unit.angstrom for i in range(3)]) * unit.angstroms
        molecule = Molecule(atom_types=atom_types, coordinates=coordinates)
        return molecule, box_vectors


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

        conversion_factor = 418.4  # kcal/mol/Å -> kJ/mol/nm
        current_forces = -1 * predicted_forces * conversion_factor * unit.kilojoule_per_mole / unit.nanometer
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
        
        # please notice that data_to_gene is intinilized to be None for the first iteration.
        ##### User Part #####
        if self.starting_point % 1000 == 0:
            print(f"MD has ran for {self.starting_point} steps")
        if self.starting_point >= 100000:
            data_to_gene = None
            print('trajecory is reached 100000 steps, start to generate new trajectory')
            self.starting_point = 0
            self.history.append([])  # start a new history for the new trajectory
        # initialize data: first iteration or when patience is exceeded
        if data_to_gene is None:
            print(f'initializing data, force to number {self.counter} in the initial data')
            if self.counter <= self.init_length - 1:
                data_to_pl = self.read_in_data(counter = self.counter)
                molecule, box_vectors = self.set_up(data_to_pl)
                num_particles = molecule.system.getNumParticles()
                init_force = self.custom_force_initilize(molecule)
                self.iterative_force = self.update_forces(init_force, data_to_pl[3].numpy())
                molecule.system.addForce(self.iterative_force)
                molecule.system.setDefaultPeriodicBoxVectors(*box_vectors)
                integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
                print('set up simulation')
                self.simulation = openmm.app.Simulation(molecule.get_Topology(), molecule.get_System(), integrator)
                print('add initial positions')
                self.simulation.context.setPositions(data_to_pl[0].numpy() * 0.1)
                self.simulation.context.setVelocitiesToTemperature(self.temperature)
                
                data_to_pl = self.update(data_to_pl)
                self.history.append([data_to_pl[0],])
                self.counter += self.gene_procs
                self.num_generate += 1
                print('counter:', self.counter)
            #read in data when initial data size exceeded and start to generate new data from beginning by random structure
            else:
                # when all initila data have been used
                # data_to_pl = self.read_in_data(self.counter % self.init_length)
                # data_to_pl[-2] = 0
                # data_to_pl[2] = None
                # data_to_pl[0] = self.random_strcuture_in_space( data_to_pl[0])
                stop = True
                #data_to_pl = convert_to_1d_float_array([data_to_pl[0], data_to_pl[1], data_to_pl[2], data_to_pl[3], data_to_pl[4], data_to_pl[-2], data_to_pl[-1]])
            # self.history.append([data_to_pl,])
            # self.counter += 1
            # data_to_pred = data_to_pl 
            # del data_to_pl

        else:
            # take the last geomety from self.history : [{'data_list': Data(y=-9.739021563862, pos=[4, 3], z=[4], forces=[4, 3], charge=-2, atoms=[4])}]
            geometry = copy.deepcopy(data_to_gene)  
            geometry = reconstruct_from_metadata(geometry, self.metadata)
            if data_to_gene[-1] > self.patience_threshold:
                print(self.patience_threshold, 'patience is exceeded, new trajectory is generated')
                # print('patience is exceeded, new trajectory is generated')
                geometry[-1] = 0
                geometry[0] = self.random_strcuture_in_space(geometry[0])
                data_to_pl = copy.deepcopy(geometry)
                # data_to_pl = convert_to_1d_float_array([geometry[0], geometry[1], geometry[2], geometry[3], geometry[4], geometry[-2], geometry[-1]])
                del geometry
                self.num_generate += 1
                self.starting_point += 1
            else:
                force = self.update_forces(self.iterative_force, geometry[5])
                force.updateParametersInContext(self.simulation.context)
                data_to_pl = self.update(geometry)
                self.num_generate += 1
                self.starting_point += 1
            
            # geometry = self.history[-1][-1]  #I save all the data object completely in data_to_gene so data_to_gene is equivalent to self.history[-1][-1]
            # generate new data
            # data_to_pl = self.update(geometry)

            self.history[-1].append(data_to_pl[0])
 
            # self.counter += 1
            if self.counter > self.limit:
                print('generation limit reached')
                stop = True
        if self.stop:
            stop = True
        if self.num_generate % 10000 == 0:
            print(f'{self.num_generate} Points are generated')

        data_to_pred  = convert_to_1d_float_array(data_to_pl)
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
        m = 'ab' if os.path.exists(self.save_path) else 'wb'
        with open(self.save_path, m) as fh:
            if len(self.history) > 1:
                pickle.dump(self.history[:-1], fh)
            else:
                pickle.dump(self.history[0], fh)
                # print('save progress:', self.history[0])
            self.history = self.history[-1:]
    def stop_run(self):
        """
        Stop the active learning workflow.
        """
        self.stop = True
        self.save_progress()
        print('stop run')    
