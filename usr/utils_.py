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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from usr.initial_pyg.functions.config import ConfigLoader
import matplotlib.pyplot as plt
from openmm import unit, Vec3
import random
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform  # only if you want i,j
from usr.initial_pyg.functions.config import ConfigLoader
import time
import openmm.app as mmapp
import copy
import openmm as mm



random_seed = 42
random.seed(random_seed)

# Set a random seed for NumPy
np.random.seed(random_seed)
def _to_owned_vec148(x):
    import numpy as np, torch
    if isinstance(x, torch.Tensor):
        x = x.detach().clone().cpu().to(torch.float64).contiguous().numpy()
    else:
        x = np.asarray(x, dtype=np.float64)

    x = np.ascontiguousarray(x, dtype=np.float64).copy()  # 关键：copy() 断开底层别名
    if x.ndim != 1 or x.size != 148:
        raise ValueError(f"record must be (148,), got {x.shape}")
    x.setflags(write=False)  # 防止后续任何地方原地写坏
    return x

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

def compute_perm_invariant_rmsd(P, Q):
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    D = cdist(P_centered, Q_centered)
    row_ind, col_ind = linear_sum_assignment(D)

    P_matched = P_centered[row_ind]
    Q_matched = Q_centered[col_ind]

    # Kabsch: rotate Q_matched to align to P_matched
    C = Q_matched.T @ P_matched
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ W))
    U = V @ np.diag([1, 1, d]) @ W
    Q_aligned = Q_matched @ U

    rmsd = np.sqrt(np.mean(np.sum((Q_aligned - P_matched) ** 2, axis=1)))
    return rmsd

def convert_with_iteration_counter(input_list, iteration_counts,metadata=None):
    data_to_gene = []
    for i, k in enumerate(input_list):
        flat = convert_to_1d_float_array(k, metadata)
        flat_with_iter = np.insert(flat, 0, iteration_counts[i])  # prepend iteration count
        data_to_gene.append(flat_with_iter)
    return data_to_gene



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
    num_gene = config['num_gen_process']

    input_to_orcl = []
    # print("DEBUG | prediction_check | list_data_to_pred length:", len(list_data_to_pred))
    input_list = [reconstruct_from_metadata(item, metadata, rank=f"prediction check", return_dict=True) for item in list_data_to_pred]
    for rec in input_list:
        z = np.asarray(rec["atomic_numbers"])
        coords = np.asarray(rec["coords"])
        assert coords.shape == (z.size, 3), \
            f"coords shape {coords.shape} != ({z.size}, 3) during prediction_check"
    # print(f"DEBUG | prediction_check | input_list: {input_list[0]}")  # Debugging line to check the first item
    # above this long is correct

   
    # energy, forces = parse_list_data_to_gene(list_data_to_gene)
    # Remove the iteration marker before parsing
    # list_data_to_gene_cleaned = [item[:, 1:] for item in list_data_to_gene]
    # iteration_counts = [int(np.round(np.mean(item[:, 0]))) for item in list_data_to_gene_cleaned]  # optional if needed
    # print(iteration_counts)

    energy, forces = parse_list_data_to_gene(list_data_to_gene)


    force_std_vec = forces.std(axis=1, ddof=1)
    #max std for force
    max_force_std = force_std_vec.max(axis=1)  # shape: (generators, atoms)

    # Now compute L2 norm of std vectors per atom:
    # force_std_l2 = np.linalg.norm(force_std_vec, axis=2)  # shape: (generators, atoms)
    # Mean vector per atom across predictors: shape (generators, atoms, 3)
    force_mean_vec = forces.mean(axis=1)  # axis=1 → over predictors
    # max_force_norm = np.linalg.norm(force_mean_vec, axis=2).max(axis=1)
    # print("Max force norm per generator:", max_force_norm)
    ##### User Part #####
    # Find the indices of the top 25% of standard deviations
    # print('pred_list:', pred_list)
    std = energy.std(axis=1, ddof=1)  # shape: (num_generators,)
    std  = max_force_std
    energy_mean = energy.mean(axis=1)
    # print('std:', std)
    # print(energy_mean, std, force_mean_vec)
    threshold = config['std_threshold']
    patience_threshold = config['patience_threshold']
    energy_threshold = config['energy_threshold']
    boundary = config['bound']
    optimal_coord = config['coord']

    upper_bound = energy_threshold + boundary
    lower_bound = energy_threshold - boundary

    for i in range(len(input_list)):
        if input_list[i]["patience"][0] < 0:
            print('Energy patience went negative:', input_list[i])
        if energy_mean[i] > upper_bound or energy_mean[i] < lower_bound:
            print('energy out of bound', energy)
            print('energy out of bound', energy_mean[i])
            input_list[i]['patience'][0] += 1


    # std filter
    if std.ndim == 1:
        i_orcl_std = np.where(std >= threshold)[0]
    else:
        i_orcl_std = np.where((std >= threshold).any(axis=1))[0]
    # RMSD filter
    rmsd_threshold = float('inf')
    optimal_coord = np.array(optimal_coord, dtype=float).reshape(input_list[0]["coords"].shape)  # reshape to match the coordinates shape
    i_orcl_rmsd = []
    # i_orcl_rmsd = [
    #     i for i, input_item in enumerate(input_list)
    #     if compute_rmsd(np.array(input_item[0]), optimal_coord) >= rmsd_threshold
    # ]
    rmsd = [compute_perm_invariant_rmsd(np.array(input_item["coords"]), optimal_coord) for input_item in input_list]

    i_orcl_rmsd = np.where(np.array(rmsd) >= rmsd_threshold)[0]
    # add patience
    for i in i_orcl_rmsd:
        input_list[i]['patience'][1] += 1  # increment patience for items with high RMSD
    # if len(i_orcl_rmsd) != 0:
    #     print(f'RMSD filter: {len(i_orcl_rmsd)} items with RMSD >= {rmsd_threshold} A')
    #     print(f'RMSD values: {np.array(rmsd)[i_orcl_rmsd]}')
    i_orcl = sorted(set(i_orcl_std).union(set(i_orcl_rmsd)))
    # force_normalized = np.linalg.norm(forces, axis=1)  # calculate the norm of forces

            


    input_to_orcl = [ copy.deepcopy(input_list[i]) for i in i_orcl]
    

    # input_to_orcl = [convert_to_1d_float_array(item, metadata) for item in input_to_orcl]  # ensure type is float64
    normalized = []
    L = record_length_from_metadata(metadata)  # 可能为 None（遇到变长list）
    for obj in input_to_orcl:  # 确保这是 dict 列表
        arr = pack_from_metadata(obj, metadata)
        # 如果 L 有具体数值，做一次等长校验
        if L is not None and arr.size != L:
            raise ValueError(f"packed size {arr.size} != expected {L}")
        normalized.append(arr.copy())  # 立刻 copy，避免别名覆盖
        

    pred_list = energy_mean
    #pred_list[i_orcl] = 0  # for predictions with high std, send 0 instead to generator

    data_to_gene = []

    for i, rec in enumerate(input_list):
        rec["pred_energy"]  = float(energy_mean[i])
        rec["pred_forces"]  = np.asarray(force_mean_vec[i], dtype=np.float64)

    # Check patience status across all items
    if any(item['patience'][-1] is not None and item['patience'][-1] > patience_threshold for item in input_list):
        print('patience reached (at least one structure)')
    elif any(item['patience'][-1] is None for item in input_list):
        print('no patience info in check function')

    # data_to_gene = convert_with_iteration_counter(input_list, iteration_counts)
    data_to_gene = [convert_to_1d_float_array(item, metadata) for item in input_list]  # convert to 1D float arrays
    
    if data_to_gene is None:
        print('no data to gene')
    elif len(data_to_gene) != len(input_list):
        print(f'Warning: data_to_gene length {len(data_to_gene)} does not match input_list length {len(input_list)}')
    # print('data_to_gene length after prediction_check:', {len(item) for item in data_to_gene})
    return normalized, data_to_gene

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
    pred_list = [k[:, 1] for k in pred_list] #DEBUG: [[iteration, energy, forces], [iteration, energy, forces], ...]
    std = [np.std(k, axis=0, ddof=1) for k in pred_list]  # calculation std of predictions from retrained ML
    # remove data with prediction std not exceeding the threshold

    #std = np.std(np.array(pred_list, dtype=float), axis=0, ddof=1)  # calculation std of predictions from retrained ML
    # sort the to_orcl_buffer list based on the std
    if len(std) != len(to_orcl_buffer):
        raise ValueError(f"Mismatch: std length {len(std)} vs. to_orcl_buffer length {len(to_orcl_buffer)}")
    if len(to_orcl_buffer) != len(pred_list):
        print(f"[ERROR] Buffer size mismatch: {len(to_orcl_buffer)} vs {len(pred_list)}")
    # Combine std_list and list1 element-wise using zip
    # combined_lists = list(zip(std, to_orcl_buffer))

    # # Sort the combined_lists based on the standard deviation values
    # sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
    # sorted_indices = np.argsort(std)  # Get sorted index order
    # print(f"Before sorting, to_orcl_buffer size = {len(to_orcl_buffer)}")

    # # to_orcl_buffer = [to_orcl_buffer[i] for i in sorted_indices if std[i] > threshold]
    # to_orcl_buffer = [to_orcl_buffer[i] for i in sorted_indices]

    # std_sorted = [std[i] for i in sorted_indices]
    print(f"After sorting, to_orcl_buffer size = {len(to_orcl_buffer)}")

    # to_orcl_buffer = [np.asarray(item, dtype=np.float64) for item in to_orcl_buffer]
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

def parse_list_data_to_gene(list_data_to_gene):
    """
    Parses list_data_to_gene structured as:
    list of arrays: each array is shape (num_predictors, 1 + 3*num_atoms)

    Returns:
        energy: (G, P)
        forces: (G, P, N, 3)
    """
    data_array = np.array(list_data_to_gene)
    if data_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data_array.shape}")

    num_generators, num_predictors, total_dim = data_array.shape
    lengths = [[len(v) for v in row] for row in data_array]
    first_len = lengths[0][0]
    for gi, row in enumerate(lengths):
        for pi, L in enumerate(row):
            if L != first_len:
                raise ValueError(f"[parse_list_data_to_gene] Ragged predictor length at G{gi} P{pi}: {L} vs {first_len}")

    num_atoms = (total_dim - 1) // 3
    

    energy = data_array[:, :, 0]                             # shape (G, P)
    forces_flat = data_array[:, :, 1:]                       # shape (G, P, 3N)
    forces = forces_flat.reshape((num_generators, num_predictors, num_atoms, 3))

    return energy, forces




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
            [0,0],  # patience
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
            [0,0],  # patience
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
    print(f"Energy: {energies_0}")
    forces_0 = np.array(np.matrix(row[header.index("forces")].replace('\n', ';'))).reshape((num_atom, 3))

    data = [
        torch.tensor(coords),
        torch.tensor(elements_number),
        torch.tensor(energies_0),
        torch.tensor(forces_0),
        torch.tensor(charge, dtype=torch.int64),
        torch.zeros(coords.shape),  # Placeholder for 'pred_force'
        None, # Placeholder for 'pred_energy'
        [0,0],  # Placeholder for 'patience'
        torch.zeros(coords.shape)
    ]
    
    return data

# def reconstruct_from_metadata(flat_array, metadata, none_placeholder=99999999.0, rank = None):
#     reconstructed_data = []
#     index = 0  # Start index for slicing flat_array


#     for meta in metadata:
#         meta_type = meta['type']

#         if meta_type == 'array':
#             shape = tuple(meta['shape'])
#             num_elements = np.prod(shape)
#             array_data = np.array(flat_array[index:index + num_elements], dtype=meta['dtype']).reshape(shape)
#             reconstructed_data.append(array_data)
#             index += num_elements

#         elif meta_type == 'tensor':
#             shape = tuple(meta['shape'])
#             dtype = getattr(torch, meta['dtype'].split('.')[-1])  # e.g., 'torch.float64' → 'float64'
#             num_elements = np.prod(shape) if shape else 1
#             tensor_data = torch.tensor(flat_array[index:index + num_elements], dtype=dtype).reshape(shape)
#             reconstructed_data.append(tensor_data)
#             index += num_elements

#         elif meta_type == 'charge':
#             reconstructed_data.append(torch.tensor(flat_array[index], dtype=torch.int64))
#             index += 1

#         elif meta_type == 'None':
#             if flat_array[index] == none_placeholder or flat_array[index] == int(none_placeholder):
#                 reconstructed_data.append(None)
#             else:
#                 reconstructed_data.append(flat_array[index])
#             index += 1

#         elif meta_type == 'scalar_nullable':
#             if flat_array[index] == none_placeholder:
#                 reconstructed_data.append(None)
#             else:
#                 if meta['dtype'] == 'int':
#                     reconstructed_data.append(int(flat_array[index]))
#                 elif meta['dtype'] == 'float':
#                     reconstructed_data.append(float(flat_array[index]))
#             index += 1

#         elif meta_type == 'scalar':
#             if meta['dtype'] == 'int':
#                 reconstructed_data.append(int(flat_array[index]))
#             elif meta['dtype'] == 'float':
#                 reconstructed_data.append(float(flat_array[index]))
#             else:
#                 raise ValueError(f"Unsupported scalar dtype: {meta['dtype']}")
#             index += 1

#         elif meta_type == 'list':
#             shape = meta.get('shape')
#             if shape is None:
#                 raise ValueError(f"metadata field '{meta.get('name','<list>')}' missing 'shape'; "
#                                 "cannot reconstruct list deterministically.")
#             list_len = np.prod(shape)
#             dtype = meta.get('dtype', 'float')
#             segment = flat_array[index : index + list_len]

#             if dtype == 'int':
#                 # Check if all entries are close to integers before casting
#                 if not np.all(np.isclose(segment, np.round(segment), atol=1e-6)):
#                     print(f"[WARNING] rank {rank} Non-integer-like values in list intended as int: {segment}, {flat_array}")
#                     raise ValueError(f"List intended as int contains non-integer-like values, malform/mismatched data recieved. ")
#                 int_list = np.round(segment).astype(int).tolist()
#                 reconstructed_data.append(int_list)

#             elif dtype == 'float':
#                 float_list = segment.astype(float).tolist()
#                 reconstructed_data.append(float_list)

#             else:
#                 raise ValueError(f"Unsupported list dtype: {dtype}")

#             index += list_len

#         else:
#             raise ValueError(f"Unknown metadata type: {meta_type}")

#     return reconstructed_data


# def convert_to_1d_float_array(data, none_placeholder=99999999.0):
#     flat_array = []
#     coords = data[0]
#     atomic_numbers = data[1]
#     forces = data[3]
#     velocities = data[8]

#     if coords.shape != (11, 3):
#         raise ValueError(f"❌ Coordinates wrong shape: {coords.shape}")
#     if len(atomic_numbers) != 11:
#         raise ValueError(f"❌ Atomic numbers wrong length: {len(atomic_numbers)}")
#     if forces.shape != (11, 3):
#         raise ValueError(f"❌ Forces wrong shape: {forces.shape}")
#     if velocities.shape != (11, 3):
#         raise ValueError(f"❌ Velocities wrong shape: {velocities.shape}")
#     for item in data:
#         if isinstance(item, np.ndarray):
#             flat_array.extend(item.ravel().astype(np.float64))

#         elif isinstance(item, torch.Tensor):
#             flat_array.extend(item.cpu().numpy().ravel().astype(np.float64))

#         elif isinstance(item, list):
#             arr = np.array(item)

#             if arr.dtype.kind in ['i', 'u']:  # integer types
#                 # Convert safely to float64
#                 float_arr = arr.astype(np.float64)
#                 flat_array.extend(float_arr.ravel())

#             elif arr.dtype.kind in ['f']:  # float types
#                 flat_array.extend(arr.astype(np.float64).ravel())

#             else:
#                 raise TypeError(f"Unsupported list dtype: {arr.dtype} in item: {item}")

#         elif isinstance(item, int) or isinstance(item, float):
#             flat_array.append(float(item))

#         elif item is None:
#             flat_array.append(float(none_placeholder))

#         else:
#             raise TypeError(f"Unexpected type in data: {type(item)}")
#     return np.array(flat_array, dtype=np.float64)

# ---------- reconstruct (flat -> structured) ----------
def reconstruct_from_metadata(flat, metadata, none_placeholder=99999999.0, return_dict=False, rank=None):
    flat = np.array(flat, dtype=np.float64, copy=True).ravel()
    out, i = {}, 0
    INT_ATOL = 1e-6

    # (optional) total-length guard
    expected_len = 0
    for m in metadata:
        shp = tuple(m.get("shape", ()))
        expected_len += int(np.prod(shp)) if shp else 1
    if flat.size != expected_len:
        raise ValueError(f"[reconstruct][rank {rank}] total length mismatch: got {flat.size}, expected {expected_len}")

    def _take(n, field_name):
        nonlocal i
        if i + n > flat.size:
            raise ValueError(f"[reconstruct][rank {rank}] ran out of data while reading '{field_name}'")
        seg = flat[i:i+n]; i += n
        return seg

    for m in metadata:
        name = m["name"]
        t    = m["type"]
        shp  = tuple(m.get("shape", ()))
        n    = int(np.prod(shp)) if shp else 1

        if t in ("array", "tensor"):
            seg = _take(n, name)
            if t == "tensor":
                tdtype = getattr(torch, m["dtype"].split(".")[-1])
                out[name] = torch.tensor(seg, dtype=tdtype).reshape(shp)
            else:
                out[name] = np.array(seg, dtype=m.get("dtype", "float64"), copy=False).reshape(shp)

        elif t == "list":
            seg = _take(n, name)
            if m.get("dtype", "float") == "int":
                # int-like check (helps catch misalignment e.g. for 'patience')
                if not np.all(np.isclose(seg, np.round(seg), atol=INT_ATOL)):
                    bad = seg[np.logical_not(np.isclose(seg, np.round(seg), atol=INT_ATOL))][:6]
                    raise ValueError(f"[reconstruct][rank {rank}] field '{name}' expected integers, saw {bad}")
                ints = np.rint(seg).astype(np.int64)

                # *** KEY FIX: provide z as tensor, not a Python list ***
                if name in ("atomic_numbers", "z"):
                    out[name] = torch.tensor(ints, dtype=torch.int64)   # <- satisfies .numpy() downstream
                else:
                    out[name] = ints.tolist()
            else:
                out[name] = seg.astype(np.float64, copy=False).tolist()

        elif t == "scalar_nullable":
            v = _take(1, name)[0]
            out[name] = None if v == none_placeholder else float(v)

        elif t == "scalar":
            v = _take(1, name)[0]
            out[name] = int(v) if m.get("dtype") == "int" else float(v)

        elif t == "charge":
            v = _take(1, name)[0]
            if not np.isclose(v, round(v), atol=INT_ATOL):
                raise ValueError(f"[reconstruct][rank {rank}] field '{name}' must be integer-like, got {v}")
            out[name] = torch.tensor(int(round(v)), dtype=torch.int64)

        elif t == "None":
            v = _take(1, name)[0]
            out[name] = None if v == none_placeholder else v

        else:
            raise ValueError(f"[reconstruct][rank {rank}] unknown type '{t}' for field '{name}'")

    if i != flat.size:
        raise ValueError(f"[reconstruct][rank {rank}] leftover data: consumed {i} of {flat.size}")

    return out if return_dict else [out[m["name"]] for m in metadata]
# ---------- flatten (structured -> flat) ----------
def convert_to_1d_float_array(data, metadata, none_placeholder=99999999.0):
    """
    `data` can be a dict keyed by YAML names OR a list matching metadata order.
    """
    get = (lambda m, idx=[0]: data[m["name"]]) if isinstance(data, dict) \
          else (lambda m, idx=[-1]: data[(idx.__setitem__(0, idx[0]+1) or idx)[0]])

    buf = []
    for m in metadata:
        name = m["name"]
        t    = m["type"]
        x    = get(m)

        if t in ("array", "tensor"):
            buf.extend(np.asarray(x).ravel().astype(np.float64))

        elif t == "list":
            arr = np.asarray(x, dtype=np.int64 if m.get("dtype","float")=="int" else np.float64)
            buf.extend(arr.ravel().astype(np.float64))

        elif t == "scalar_nullable":
            buf.append(none_placeholder if x is None else float(x))

        elif t == "scalar":
            buf.append(float(x) if m.get("dtype")!="int" else float(int(x)))

        elif t == "charge":
            buf.append(float(int(x)))  # single int

        elif t == "None":
            buf.append(none_placeholder if x is None else float(x))

        else:
            raise ValueError(f"Unknown type '{t}' for field '{name}'")

    return np.array(buf, dtype=np.float64, copy=True)


def _flatten_one(field, meta, NONE_PLACEHOLDER=99999999.0):
    t = meta["type"]
    dt = meta.get("dtype")
    shp = tuple(meta.get("shape", ()))
    if t in ("array", "tensor"):
        arr = np.asarray(field, dtype=np.float64).reshape(-1)
        return arr
    elif t == "list":
        arr = np.asarray(field, dtype=np.float64).reshape(-1)
        return arr
    elif t == "scalar_nullable":
        v = NONE_PLACEHOLDER if (field is None) else float(field)
        return np.array([v], dtype=np.float64)
    elif t == "scalar":
        if dt == "int":
            return np.array([int(field)], dtype=np.float64)
        else:
            return np.array([float(field)], dtype=np.float64)
    elif t == "charge":
        return np.array([int(round(float(field)))], dtype=np.float64)
    elif t == "None":
        v = NONE_PLACEHOLDER if (field is None) else float(field)
        return np.array([v], dtype=np.float64)
    else:
        raise ValueError(f"unknown type {t}")

def pack_from_metadata(obj_dict, metadata):
    """严格按 metadata 顺序把一条记录打平成 1D float64 numpy 数组。"""
    pieces = []
    for m in metadata:
        name = m["name"]
        if name not in obj_dict:
            raise KeyError(f"missing field '{name}' while packing")
        pieces.append(_flatten_one(obj_dict[name], m))
    return np.concatenate(pieces, dtype=np.float64)

def record_length_from_metadata(metadata):
    """计算一条记录理论长度，用于 sanity check"""
    total = 0
    for m in metadata:
        t = m["type"]
        shp = tuple(m.get("shape", ()))
        if t in ("array", "tensor", "list"):
            n = int(np.prod(shp)) if shp else None  # list 没 shape 就不可预期
            if n is None:
                return None
            total += n
        else:
            total += 1
    return total



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
