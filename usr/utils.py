#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:44:34 2023

@author: chen
"""
import ast
from math import pi
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from usr.initial_pyg.functions.config import ConfigLoader
import matplotlib.pyplot as plt
from openmm import unit, Vec3
import random
import numpy as np

from usr.initial_pyg.functions.config import ConfigLoader
import time
import openmm.app as mmapp
import copy
import openmm as mm

random_seed = 42
random.seed(random_seed)

# Set a random seed for NumPy
np.random.seed(random_seed)

# Set a random seed for PyTorch
#torch.manual_seed(random_seed)
def compute_rmsd(P, Q):
    """Align P to Q and compute RMSD (P and Q are Nx3 arrays)"""
    # Subtract centroid
    P -= P.mean(axis=0)
    Q -= Q.mean(axis=0)

    # Kabsch alignment
    C = np.dot(P.T, Q)
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, W)))
    U = np.dot(V, np.dot(np.diag([1, 1, d]), W))
    P_aligned = np.dot(P, U)

    return np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
def prediction_check(list_data_to_pred, list_data_to_gene):
    """
    User defined predictions check function.
    Check the predictions from Prediction processes (e.g. STD). 
    
    Args:
        list_data_to_pred (list): list of data_to_pred gathered from all generators, sorted by the rank of generator.
                                  Source: list of data_to_pred from UserGene.generate_new_data()
                                  [1-D numpy.ndarray, 1-D numpy.ndarray, ...], size equal to number of generators.
        list_data_to_gene (list): list of data_to_gene gathered from all models in prediction kernel, sorted by the rank of model.
                                  Source: data_to_gene_list from UserModel.predict()
                                  [numpy.ndarray, numpy.ndarray, ...], array shape (n_pred, model output size), size equal to number of generators.

    Returns:
        list_input_to_orcl (list): list of user defined input to oracle to generate ground truth.
                                   Destination: list of input_for_orcl at UserOracle.run_calc().
                                   [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
        list_data_to_gene_checked (list): list of predictions distributed to generators.
                                  Destination: list of data_to_gene to UserGene.generate_new_data(), length must match the number of generators and should be sorted by the rank of generator.
                                  [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
    """
    config = ConfigLoader("config.yaml")
    metadata = config['metadata']

    num_generators = len(list_data_to_pred)
    input_to_orcl = []
    # print(len(list_data_to_pred[0])) 32
    input_list = [reconstruct_from_metadata(item, metadata) for item in list_data_to_pred]

    # pred_list = [k.tolist() for k in pred_list]
    list_data_to_gene = list_data_to_gene[0]

    pred_list, force_list = unflatten_predictions(list_data_to_gene)
   



    # Stack the arrays along a new dimension
    stacked_arrays = np.stack(force_list)

    # Compute the mean along the new dimension (axis=0)
    forces = np.mean(stacked_arrays, axis=0)
    forces = torch.tensor(forces)

    ##### User Part #####
    # Find the indices of the top 25% of standard deviations
    # print('pred_list:', pred_list)
    std = np.std(np.array(pred_list, dtype=float), axis=0, ddof=1)  # calculate std of PL predictions
    # print('std:', std)
    threshold = config['std_threshold']
    patience_threshold = config['patience_threshold']
    energy_threshold = config['energy_threshold']
    boundary = config['bound']
    optimal_coord = config['coord']

    upper_bound = energy_threshold + boundary
    lower_bound = energy_threshold - boundary
    avg_energy_pred = np.mean(np.array(pred_list, dtype=float), axis=0)
    if avg_energy_pred > upper_bound or avg_energy_pred < lower_bound:
        print('energy out of bound', avg_energy_pred)
        print(input_list[0][-1])
        input_list[0][-2] += 1
        # print('COORIDNATES:', input_list[0]['data_list'][0])
        # print('pred ENERGY:', avg_energy_pred)

    # std filter
    if std.ndim == 1:
        i_orcl_std = np.where(std >= threshold)[0]
    else:
        i_orcl_std = np.where((std >= threshold).any(axis=1))[0]
    # RMSD filter
    rmsd_threshold = 0.5
    optimal_coord = np.array(optimal_coord, dtype=float).reshape(input_list[0][0].shape)  # reshape to match the coordinates shape
    i_orcl_rmsd = [
        i for i, input_item in enumerate(input_list)
        if compute_rmsd(np.array(input_item[0]), optimal_coord) >= rmsd_threshold
    ]
    i_orcl = sorted(set(i_orcl_std).union(set(i_orcl_rmsd)))

    input_to_orcl = [convert_to_1d_float_array(input_list[i]) for i in i_orcl]

    pred_list = np.mean(np.array(pred_list, dtype=float), axis=0)  # take the mean of predictions to send to generator
    #pred_list[i_orcl] = 0  # for predictions with high std, send 0 instead to generator
    input_list[0][-3] = torch.tensor(pred_list[0])
    input_list[0][5] = forces
    if input_list[0][-2] > patience_threshold:
        data_to_gene = copy.deepcopy(input_list)
        print('patience reached')
    elif input_list[0][-2] is None:
        print('no patience in check function')
    else:
        data_to_gene = [convert_to_1d_float_array(k) for k in input_list]

    
    if data_to_gene is None:
        print('no data to gene')

    return input_to_orcl, data_to_gene

def adjust_input_for_oracle(to_orcl_buffer, pred_list):
    """
    User defined function to adjust data in oracle buffer based on the corresponding predictions in pred_list.
    Called only when dynamic_orcale_list is True in al_setting.
    
    Args:
        to_orcl_buffer (list): list of input for oracle labeling.
                               Source: list of input_to_orcl to UserOracle.run_calc().
                               [1-D numpy.ndarray, 1-D numpy.ndarray, ...], size equal to number of elements in the oracle buffer
        pred_list (list): list of corresponding predictions of to_orcl_buffer from retrained ML.
                          Source: UserModel.predict()
                          [1-D numpy.ndarray, 1-D numpy.ndarray, ...], size equal to number of elements in the oracle buffer
    Returns:
        to_orcl_buffer (list): list of adjusted input for oracle labeling. (list of input_to_orcl to UserOracle.run_calc())
                               Destination: list of input for oracle labeling.
                               [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
    """
    
    ##### User Part #####

    # print('to_orcl_buffer:', to_orcl_buffer) list of arrays representing the 1d arrya of data
    # print('pred_list:', pred_list) list of arrays representing the 1d array of predictions energy and flattened forces
    print('dynamic retraining data adjustment')
    config = ConfigLoader("config.yaml")
    threshold = config['std_threshold']
    pred_list = [k[:, 0] for k in pred_list]
    std = [np.std(k, axis=0, ddof=1) for k in pred_list]  # calculation std of predictions from retrained ML
    # remove data with prediction std not exceeding the threshold


    #std = np.std(np.array(pred_list, dtype=float), axis=0, ddof=1)  # calculation std of predictions from retrained ML
    # sort the to_orcl_buffer list based on the std
    if len(std) != len(to_orcl_buffer):
        raise ValueError(f"Mismatch: std length {len(std)} vs. to_orcl_buffer length {len(to_orcl_buffer)}")
    # Combine std_list and list1 element-wise using zip
    # combined_lists = list(zip(std, to_orcl_buffer))

    # # Sort the combined_lists based on the standard deviation values
    # sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
    sorted_indices = np.argsort(std)  # Get sorted index order
    print(f"Before sorting, to_orcl_buffer size = {len(to_orcl_buffer)}")

    # to_orcl_buffer = [to_orcl_buffer[i] for i in sorted_indices if std[i] > threshold]
    # to_orcl_buffer = [to_orcl_buffer[i] for i in sorted_indices]
    std_sorted = [std[i] for i in sorted_indices]
    print(f"After sorting, to_orcl_buffer size = {len(to_orcl_buffer)}")

    to_orcl_buffer = [np.asarray(item, dtype=np.float64) for item in to_orcl_buffer]
    print(f"After ensurance of type, to_orcl_buffer size = {len(to_orcl_buffer)}")
    # check if every item in the list has the same shape
    first_shape = to_orcl_buffer[0].shape  # Get the shape of the first item
    if all(item.shape == first_shape for item in to_orcl_buffer):
        print(" All items in to_orcl_buffer have the same shape:", first_shape)
    else:
        print(" Inconsistent shapes in to_orcl_buffer!")
        for i, item in enumerate(to_orcl_buffer):
            print(f"Item {i} shape: {item.shape}")


    # print('to_orcl_buffer:', to_orcl_buffer)

    #i_orcl_sorted = np.argsort(np.mean(std, axis=1), axis=0)[::-1]
    #to_orcl_buffer = np.array(to_orcl_buffer, dtype=float)[i_orcl_sorted]

    std = sorted(std)
    #to_orcl_buffer = list(to_orcl_buffer[np.nonzero((std > threshold).any(axis=1))[0]])  # remove data with prediction std not exceeding the threshold 
    # print(to_orcl_buffer)
    # pickle.dump(to_orcl_buffer, open('results/to_orcl_buffer.pkl', 'wb'))
    return to_orcl_buffer





####### dataset modules##########



def get_adjacency_matrix(coords, threshold=3.5):
    # Compute the adjacency matrix for a molecule
    # 3.5 is the default value
    pairwise_distances = pdist(coords)
    adjacency_matrix = (squareform(pairwise_distances) < threshold).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)  # Ensure no self-connections
    
    return adjacency_matrix







def shuffle_dataset(data_list):
    random.shuffle(data_list)
    data_list = [item for item in data_list if item is not None]
    
    # for item in data_list:
    #     if isinstance(item.pos, torch.Tensor):
    #         item.pos = item.pos.numpy()
    # 90% of the data is used for training, 10% for validation
    split = int(len(data_list) * 0.9)
    train_dataset = data_list[:split]
    val_dataset = data_list[split:]
    return train_dataset, val_dataset






#### model modules ####


def to_clean_repr(x):
    if isinstance(x, torch.Tensor):
        return repr(x.cpu().detach().tolist())
    elif isinstance(x, np.ndarray):
        return repr(x.tolist())
    elif isinstance(x, (list, tuple)):
        return repr(x)
    else:
        return x  # int, float, etc.


def save_data(data_list):
    ### save data_list to file, data_list is a list of data object
    node_feature = []
    atoms_list = []
    global_charge = []
    energy = []
    force = []
    patience = []
    for data in data_list:
        atoms = data[1]
        node_feature_row = data[0]
        global_charge_row = data[4]
        energy_row = data[2]
        patience_row = data[-2]
        force_row = data[3]

        atoms_list.append(to_clean_repr(atoms))
        node_feature.append(to_clean_repr(node_feature_row))
        global_charge.append(to_clean_repr(global_charge_row))
        energy.append(float(energy_row))  # force scalar float
        force.append(to_clean_repr(force_row))
        patience.append(patience_row)
    df = pd.DataFrame({'atoms': atoms_list, 'node_feature': node_feature, 'global_charge': global_charge, 'energy': energy, 'force': force,'patience': patience})
    return df
def generate_xyz(atoms, tensor):
    n_atoms = len(atoms)
    lines = [str(n_atoms), "Generated XYZ coordinates"]
    for atom, coords in zip(atoms, tensor):
        line = f"{atom} {coords[0]:.4f} {coords[1]:.4f} {coords[2]:.4f}"
        lines.append(line)
    return "\n".join(lines)


def convert_to_data_object(list):
    data_list = []
    for item in list:
        data = Data(
            pos=torch.tensor(item[0]),
            z=item[1],
            y=item[2],
            forces=item[3],
            charge=item[4],
            pred=item[-3]
            # patience=item[-2]
        )
        data_list.append(data)
    return data_list



class Predictor:
    def __init__(self, model):
        self.model = model
    def predict(self, data, bathc_size):
        if bathc_size == 1:
            data.global_charge = [data.global_charge]
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
        return out
    def predict_loader(self, loader):
        bathc_size = len(loader)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                out = self.predict(batch, bathc_size)
                predictions.append(out)
        predictions = torch.cat(predictions, dim=0)
        return predictions




class Molecule():

    def __init__(self, atom_types, coordinates):        
        self.topology = mmapp.Topology()
        self.system = mm.System()

        self.atom_types = atom_types
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates).to(dtype=torch.float32)
            print('convert to torch tensor')
        self.coordinates = coordinates


        chain = self.topology.addChain()
        residue = self.topology.addResidue('MOL', chain)


        # Add atoms to the chain with their respective element and coordinates
        self.atoms = []
        for atom_type, coord in zip(atom_types, coordinates):
            element = mmapp.Element.getBySymbol(atom_type)
            atom = self.topology.addAtom(atom_type, element, residue)
            self.atoms.append(atom)
            self.system.addParticle(element.__getattribute__('mass'))


    def get_Topology(self):
        return self.topology
    
    def get_Positions(self):
        return self.coordinates

    def get_System(self):
        return self.system
    def get_num_atoms(self):
        return len(self.atoms)
    
from torch.utils.data import Dataset
from torch_geometric.transforms import RadiusGraph
class retrain_dataset(Dataset):
    def __init__(self, data, transforms=None):
        if transforms is not None:
            self.data = [transforms(i) for i in data]
        self.data = data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    

    
from ast import literal_eval   
from ase.data import atomic_numbers
import re
def get_init_data(path):
    match = re.search(r"bi(\d+)(-?\d+)(?:_(?:samples|parsed))*\.csv", path)

    if match:
        num_atom = int(match.group(1))
        charge = int(match.group(2))
        print(f"num_atom: {num_atom}, charge: {charge}")
    else:
        print("Pattern not found")
    data = pd.read_csv(path)
    elements = data["atoms"].values
    elements = [literal_eval(e) for e in elements]
    elements_number = [[atomic_numbers[ei] for ei in e] for e in elements]
    coords = data["coordinates"].values
    coords = [np.array(np.matrix(c.replace('\n', ';'))).reshape((num_atom, 3)) for c in coords]
    energies_0 = data['total_energy'].values
    energies_0 = [literal_eval(e) for e in energies_0]
    # convert = lambda a: np.array(np.matrix(a.replace('\n', ';'))) if type(a) == str else a
    forces_0 = data['forces'].values
    forces_0 = [np.array(np.matrix(c.replace('\n', ';'))).reshape((num_atom, 3)) for c in forces_0]
    data_list = []  
    for i in range(len(coords)):

        data = [
            torch.tensor(coords[i]), 
            torch.tensor(elements_number[i]), 
            torch.tensor(energies_0[i]), 
            torch.tensor(forces_0[i]), 
            torch.tensor(charge, dtype=torch.int64), 
            torch.zeros(coords[i].shape),
            None, 
            0,
            torch.zeros(coords[i].shape)]
        data_list.append(data)
    # print('data_list:', data_list[0].forces)
    return data_list

def get_full_data_init(path):

    data = pd.read_csv(path)
    elements = data["atoms"].values
    elements = [literal_eval(e) for e in elements]
    # TODO: add num_atoms based on initial pyg way
    num_atoms = [len(e) for e in elements]  
    elements_number = [[atomic_numbers[ei] for ei in e] for e in elements]
    coords = data["coordinates"].values
    coords = [np.array(np.matrix(c.replace('\n', ';'))).reshape((num_atoms[i], 3)) for i, c in enumerate(coords)]
    energies_0 = data['total_energy'].values
    energies_0 = [literal_eval(e) for e in energies_0]
    # convert = lambda a: np.array(np.matrix(a.replace('\n', ';'))) if type(a) == str else a
    forces_0 = data['forces'].values
    forces_0 = [np.array(np.matrix(c.replace('\n', ';'))).reshape((num_atoms[i], 3)) for i, c in enumerate(forces_0)]
    charge = [int(str(val).split('(')[1].split(',')[0]) if 'tensor' in str(val) else int(val) for val in data['charge'].values]

    data_list = []  
    for i in range(len(coords)):

        data = [
            torch.tensor(coords[i]), 
            torch.tensor(elements_number[i]), 
            torch.tensor(energies_0[i]), 
            torch.tensor(forces_0[i]), 
            torch.tensor(charge[i], dtype=torch.int64), 
            torch.zeros(coords[i].shape),
            None, 
            0,
            torch.zeros(coords[i].shape)]
        data_list.append(data)
    # print('data_list:', data_list[0].forces)
    return data_list




import csv
def get_specific_data(file_path, line_number):
    match = re.search(r"bi(\d+)(-?\d+)(?:_(?:samples|parsed))*\.csv", file_path)

    if match:
        num_atom = int(match.group(1))
        charge = int(match.group(2))
        print(f"num_atom: {num_atom}, charge: {charge}")
    else:
        print("Pattern not found")
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header
        for i, row in enumerate(reader):
            if i == line_number:
                data = process_row(row, header, num_atom, charge)
                return data
    raise IndexError("Line number out of range")

def process_row(row, header, num_atom, charge):
    elements = literal_eval(row[header.index("atoms")])
    elements_number = [atomic_numbers[ei] for ei in elements]
    
    coords = np.array(np.matrix(row[header.index("coordinates")].replace('\n', ';'))).reshape((num_atom, 3))
    energies_0 = literal_eval(row[header.index("total_energy")])
    forces_0 = np.array(np.matrix(row[header.index("forces")].replace('\n', ';'))).reshape((num_atom, 3))

    data = [
        torch.tensor(coords),
        torch.tensor(elements_number),
        torch.tensor(energies_0),
        torch.tensor(forces_0),
        torch.tensor(charge, dtype=torch.int64),
        torch.zeros(coords.shape),  # Placeholder for 'pred_force'
        None, # Placeholder for 'pred_energy'
        0, 
        torch.zeros(coords.shape)
    ]
    
    return data


def reconstruct_from_metadata(flat_array, metadata, none_placeholder=99999999.0):
    reconstructed_data = []
    index = 0  # Start index for slicing flat_array

    for meta in metadata:
        if meta['type'] == 'array':
            # Reconstruct a NumPy array with the specified shape and dtype
            shape = tuple(meta['shape'])
            num_elements = np.prod(shape)
            # print(flat_array[index:index + num_elements])
            array_data = np.array(flat_array[index:index + num_elements], dtype=meta['dtype']).reshape(shape)
            reconstructed_data.append(array_data)
            index += num_elements

        elif meta['type'] == 'tensor':
            # Reconstruct a PyTorch tensor with specified shape and dtype
            shape = tuple(meta['shape'])
            dtype = getattr(torch, meta['dtype'].split('.')[1])  # e.g., 'torch.float64' to torch.float64
            num_elements = np.prod(shape) if shape else 1
            # print(flat_array[index:index + num_elements])
            tensor_data = torch.tensor(flat_array[index:index + num_elements], dtype=dtype).reshape(shape)
            reconstructed_data.append(tensor_data)
            index += num_elements
        elif meta['type'] == 'charge':
            reconstructed_data.append(torch.tensor(flat_array[index], dtype=torch.int64))
            index += 1
        elif meta['type'] == 'None':
            # print(reconstructed_data.append(flat_array[index]))
            # print(none_placeholder)
            # Convert the placeholder back to None
            if flat_array[index] == none_placeholder or flat_array[index] == int(none_placeholder):
                # print('here!!!!!!!!!!!')
                reconstructed_data.append(None)
            else:
                reconstructed_data.append(flat_array[index])
            index += 1  # Move index forward
        elif meta['type'] == 'scalar_nullable':
            # Check if the value is the placeholder; if so, replace it with None
            if flat_array[index] == none_placeholder:
                reconstructed_data.append(None)
            else:
                # Convert based on dtype
                if meta['dtype'] == 'int':
                    reconstructed_data.append(int(flat_array[index]))
                elif meta['dtype'] == 'float':
                    reconstructed_data.append(float(flat_array[index]))
            index += 1
        elif meta['type'] == 'scalar':
            # print(flat_array[index])
            # Handle scalar conversion based on specified dtype
            if meta['dtype'] == 'int':
                reconstructed_data.append(int(flat_array[index]))
            elif meta['dtype'] == 'float':
                reconstructed_data.append(float(flat_array[index]))
            index += 1

    return reconstructed_data


def convert_to_1d_float_array(data):
    flat_array = []

    for item in data:
        # if isinstance(item, torch.Tensor):
        #     # Flatten tensor and add its elements
        #     flat_array.extend(item.flatten().numpy())
        
        # elif isinstance(item, np.ndarray):
        #     # Flatten NumPy array and add its elements
        #     flat_array.extend(item.flatten())
        
        # elif item is None:
        #     # Replace None with a placeholder value
        #     flat_array.append(float(99999999.0))
        
        # elif isinstance(item, (int, float)):
        #     # Append scalar values directly
        #     flat_array.append(float(item))
        
        # elif isinstance(item, np.ndarray):  # For Quantities now as np.array
        #     flat_array.extend(item.flatten())
        
        # elif isinstance(item, list):
        #     # Recursively flatten lists
        #     flat_array.extend(flatten_to_1d_array(item))
        if isinstance(item, np.ndarray):
            flat_array.extend(item.ravel())  # Efficient flattening
        elif isinstance(item, torch.Tensor):
            flat_array.extend(item.cpu().numpy().ravel())  # Convert tensor -> NumPy -> Flatten
        elif isinstance(item, list):
            flat_array.extend(np.array(item, dtype=np.float64).ravel())  # Convert list -> NumPy -> Flatten
        elif isinstance(item, int) or isinstance(item, float):
            flat_array.append(float(item))  # Convert int/float directly
        elif item is None:
            flat_array.append(float(99999999.0))  # Placeholder for None
        else:
            raise TypeError(f"Unexpected type in data: {type(item)}")


    # Convert the flat_array to a NumPy array
    #print(f"Final flattened array size: {len(flat_array)}")
    # if len(flat_array) != 56:
    #     print('wrong size after flatten')

    return np.array(flat_array, dtype=np.float64)
def unflatten_predictions(flattened_preds):
    """
    Unflattens a numpy array of shape (n, 13) into lists of y_pred and force_pred in their original shapes.
    
    Args:
        flattened_preds (np.ndarray): The flattened predictions of shape (n, 13).
        
    Returns:
        list: List of y_pred values (each of shape (1, 1)).
        list: List of force_pred values (each of shape (4, 3)).
    """
    n = flattened_preds.shape[0]  # Get the number of predictions (rows)
    
    y_pred_list = []
    force_pred_list = []
    
    for i in range(n):
        # Slice the flattened array
        y_pred_flat = flattened_preds[i, 0]  # First element is y_pred
        force_pred_flat = flattened_preds[i, 1:]  # The remaining 12 elements are force_pred
        shape = force_pred_flat.shape[0]//3
        
        # Reshape back to original shapes
        y_pred = np.array([[y_pred_flat]])  # Shape (1, 1)
        force_pred = force_pred_flat.reshape(shape, 3)  # Shape (4, 3)
        
        # Append to lists
        y_pred_list.append(y_pred)
        force_pred_list.append(force_pred)
    
    return y_pred_list, force_pred_list


def kabsch_rmsd(P, Q):
    """
    Calculate the RMSD between two point sets P and Q using the Kabsch algorithm.
    Both P and Q must be NumPy arrays of shape (N, 3).
    """

    # Center both sets to their centroids
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    # Covariance matrix
    C = np.dot(P_centered.T, Q_centered)

    # Optimal rotation matrix using SVD
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    U = np.dot(V, np.dot(D, Wt))

    # Rotate P
    P_rotated = np.dot(P_centered, U)

    # Calculate RMSD
    diff = P_rotated - Q_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd
