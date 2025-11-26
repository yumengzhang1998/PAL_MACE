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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


random_seed = 42
random.seed(random_seed)

# Set a random seed for NumPy
np.random.seed(random_seed)

# Set a random seed for PyTorch
#torch.manual_seed(random_seed)
def max_atom_distance(coords):
    """Compute maximum pairwise distance in coords (N,3)."""
    d = pdist(np.asarray(coords, dtype=float), metric="euclidean")
    return float(np.max(d))

def is_unreasonable_structure_list(input_list, ref_max_dist, tolerance=0.2):
    """
    Check a list of structures and return indices of unreasonable ones.

    Args:
        input_list (list): list of reconstructed data, each with coords in input_item[0].
        ref_max_dist (float): reference maximum distance (Å).
        tolerance (float): allowed fractional deviation.

    Returns:
        tuple: (indices, deviations, max_distances)
            indices: np.ndarray of unreasonable indices
            deviations: list of deviations for each structure
            max_distances: list of max distances for each structure
    """
    max_distances = []
    deviations = []
    i_orcl_structural = []

    for idx, input_item in enumerate(input_list):
        coords = np.array(input_item[0], dtype=float)
        dmax = max_atom_distance(coords)
        if ref_max_dist > 0:
            deviation = (dmax - ref_max_dist) / ref_max_dist
        else:
            deviation = np.inf
        max_distances.append(dmax)
        deviations.append(deviation)
        if deviation > tolerance:
            i_orcl_structural.append(idx)

    return np.array(i_orcl_structural, dtype=int), deviations, max_distances
import numpy as np

def select_uncertain_forces_GPA3(
    forces: np.ndarray,
    tau_rms: float,
    tau_spike: float,
    energies: np.ndarray | None = None,
    tau_energy: float | None = None,
    reduce_generators: str | None = None,
):
    """
    forces:   ndarray [G, P, A, 3]  (generators, predictors(models), atoms, xyz)
    energies: OPTIONAL ndarray [G, P] or [G, P, 1] (per-generator energy predictions)
    tau_rms:      threshold for RMS across atoms & xyz (per generator)
    tau_spike:    threshold for max per-atom vector-norm (per generator)
    tau_energy:   OPTIONAL threshold for energy std (per generator)
    reduce_generators:
        None  -> return per-generator decisions (indices over G)
        'max'|'mean'|'p95' -> reduce metrics across generators to one scalar and
                              return a single boolean decision + metrics dict.

    Returns
    -------
    If reduce_generators is None:
        i_orcl_std : np.ndarray[int]  # generator indices marked UNCERTAIN
        i_certain  : np.ndarray[int]  # generator indices marked certain
        rms_g      : np.ndarray[float]  # [G]
        max_atom_g : np.ndarray[float]  # [G]
        energy_std_g : np.ndarray[float] | None  # [G] or None if energies not provided
    Else:
        uncertain_bool : bool
        metrics        : dict with keys 'rms', 'max_atom', and (if provided) 'energy_std'
    """
    G, P, A, _ = forces.shape

    # --- std across predictors(models) ---
    if P <= 1:
        std_gp = np.zeros((G, A, 3), dtype=forces.dtype)
    else:
        std_gp = forces.std(axis=1, ddof=1)  # [G, A, 3]

    # --- force metrics per generator ---
    # 1) RMS over atoms & coordinates (rotation-invariant)
    rms_g = np.sqrt(np.mean(std_gp**2, axis=(1, 2)))  # [G]
    # 2) Max per-atom vector-norm of std (rotation-invariant spike guard)
    max_atom_g = np.max(np.linalg.norm(std_gp, axis=-1), axis=1)  # [G]

    # --- energy std per generator (optional) ---
    energy_std_g = None
    if energies is not None:
        E = np.asarray(energies)
        if E.ndim == 3 and E.shape[-1] == 1:
            E = E[..., 0]
        if E.shape[:2] != (G, P):
            raise ValueError(f"energies shape {E.shape} incompatible with forces {forces.shape}")
        if P <= 1:
            energy_std_g = np.zeros(G, dtype=float)
        else:
            # sample std across predictors
            energy_std_g = E.std(axis=1, ddof=1).astype(float)

    # -------- per-generator decision --------
    if reduce_generators is None:
        uncertain_mask = (rms_g >= tau_rms) | (max_atom_g >= tau_spike)
        if (energy_std_g is not None) and (tau_energy is not None):
            uncertain_mask |= (energy_std_g >= tau_energy)

        i_orcl_std = np.where(uncertain_mask)[0]
        i_certain  = np.where(~uncertain_mask)[0]
        return i_orcl_std, i_certain, rms_g, max_atom_g, energy_std_g

    # -------- reduce across generators to one decision --------
    def _reduce(vec: np.ndarray, how: str) -> float:
        if how == 'max':
            return float(np.max(vec))
        if how == 'mean':
            return float(np.mean(vec))
        if how == 'p95':
            return float(np.percentile(vec, 95))
        raise ValueError("reduce_generators must be None, 'max', 'mean', or 'p95'.")

    rms_red = _reduce(rms_g, reduce_generators)
    max_atom_red = _reduce(max_atom_g, reduce_generators)
    energy_std_red = None
    if energy_std_g is not None:
        energy_std_red = _reduce(energy_std_g, reduce_generators)

    uncertain = (rms_red >= tau_rms) or (max_atom_red >= tau_spike)
    if (tau_energy is not None) and (energy_std_red is not None):
        uncertain = uncertain or (energy_std_red >= tau_energy)

    metrics = {"rms": rms_red, "max_atom": max_atom_red}
    if energy_std_red is not None:
        metrics["energy_std"] = energy_std_red
    return uncertain, metrics

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

    ##### User Part #####
    config = ConfigLoader("config.yaml")
    metadata = config['metadata']
    max_dist = config['max_dist']

    num_generators = len(list_data_to_pred)
    input_to_orcl = []
    # print(len(list_data_to_pred[0])) 32
    sent = [item[0] for item in list_data_to_pred]  # sent count
    list_data_to_pred = [item[1:] for item in list_data_to_pred]  # remove sent count
    input_list = [reconstruct_from_metadata(item, metadata) for item in list_data_to_pred]


    energy, forces, iter = parse_list_data_to_gene(list_data_to_gene, has_iter=True)
    # energy & forces part
    # force_std_vec = forces.std(axis=1, ddof=1)
    # max_force_std = force_std_vec.max(axis=1)
    force_mean_vec = forces.mean(axis=1)
    energy_std = energy.std(axis=1, ddof=1)  # shape: (num_generators,)
    # std = copy.deepcopy(max_force_std)
    # std  = max_force_std
    energy_mean = energy.mean(axis=1)

    for i, rec in enumerate(input_list):
        rec[-3]  = float(energy_mean[i]) # energy
        rec[5]  = np.asarray(force_mean_vec[i], dtype=np.float64) # forces

    energy_threshold = config['energy_std_threshold']
    q75_rms = config['force_rms_std']
    q75_max_atom = config['force_atom_max_std']
    q75_energy = config['energy_std_threshold']
    patience_threshold = config['patience_threshold']
    energy_threshold = config['energy_threshold']
    boundary = config['bound']
    # hard_bound = config['hard_bound']
    # soft_bound = config['soft_bound']
    # optimal_coord = config['coord']
    upper_bound = energy_threshold + boundary
    # lower_bound = energy_threshold - boundary
    for i in range(len(input_list)):
        if energy_mean[i] > upper_bound:
            print('energy out of bound, predicted energies are:', energy)
            print('energy out of bound, mean energy is:', energy_mean[i])
            input_list[i][-2][0] += 1
        # e = float(energy_mean[i])  # model mean energy for frame i
        # dev = abs(e - energy_threshold)
        # dev = e - energy_threshold

        # if dev <= soft_bound:
        #     pass  # normal: uncertainty decides
        # elif dev <= hard_bound:
        #     suspect exploration zone: increment patience, only send if persistent
        #     input_list[i][-2][0] += 1
        # else:
        #     ultra-OOD: drop / do not enqueue (optionally stop MD)
        #     input_list[i][-2][0] += config['patience_threshold']  # max out patience to block

    # to_orcl filter part

    ## STD filter
    # if energy_std.ndim == 1:
    #     i_orcl_std = np.where(energy_std >= energy_threshold)[0]
    # else:
    #     i_orcl_std = np.where((energy_std >= energy_threshold).any(axis=1))[0]
    i_orcl_std, i_certain, rms_g, max_atom_g, e_std_g = select_uncertain_forces_GPA3(
        forces, q75_rms, q75_max_atom, energies=energy, tau_energy=q75_energy, reduce_generators=None
    )


    ## structural filter
    i_orcl_structural = []
    # RMSD filter
    # rmsd_threshold = float('inf')
    # optimal_coord = np.array(optimal_coord, dtype=float).reshape(input_list[0][0].shape)  # reshape to match the coordinates shape
    # rmsd = [compute_perm_invariant_rmsd(np.array(input_item[0]), optimal_coord) for input_item in input_list]
    # i_orcl_structural = np.where(np.array(rmsd) >= rmsd_threshold)[0]

    # MAX distance filter
    i_orcl_structural, _, distance = is_unreasonable_structure_list(
        input_list, max_dist, tolerance=0.1
    )
    ########################
    for i in i_orcl_structural:
        input_list[i][-2][1] += 1
        print('structural deviation reached, max distance is:', distance[i])
    # i_orcl = sorted(set(i_orcl_std).union(set(i_orcl_structural)))
    i_orcl = sorted(set(i_orcl_std))
    # print('Indices selected by STD filter:', i_orcl_std)


    if any(item[-2][-1] is not None and item[-2][-1] > patience_threshold for item in input_list):
        print('structural patience reached (at least one structure)')
    elif any(item[-2][-1] is None for item in input_list):
        print('no patience info in check function')

    # data_to_gene & input_to_orcl conversion
    data_to_gene = copy.deepcopy(input_list)
    data_to_gene = [convert_to_1d_float_array(k) for k in data_to_gene]
    i_filtered  = []
    for i in i_orcl:
        if sent[i] == 0:
            # print('Selected index with sent=0 :', i)
            i_filtered.append(i)
            sent[i] +=1
    i_orcl = i_filtered
    # print('Selected indices for oracle (after filtering sent=0):', i_orcl)
    # print('Corresponding sent counts:', [sent[i] for i in i_orcl])


    if iter is not None:
        for i in range(len(data_to_gene)):
            iter_list = iter[i] # (num, predictors)
            # if numbers in iter_list are all the same, then only add one number
            if np.all(iter_list == iter_list[0]):
                mean_iter = int(np.mean(iter_list)) 
                
            else:
            #    raise ValueError(f"All iteration numbers are the same ({iter_list}), please check the input.")
                print(f"Warning: Iteration numbers are not the same ({iter_list}), using mean value.")
                mean_iter = int(np.mean(iter_list))
            data_to_gene[i] = np.concatenate(([sent[i]], data_to_gene[i]))
            data_to_gene[i] = np.concatenate(([mean_iter], data_to_gene[i]))

    
    input_to_orcl = [convert_to_1d_float_array(copy.deepcopy(input_list[i])) for i in i_orcl]
    # input_to_orcl = [convert_to_1d_float_array(input_list[i]) for i in i_orcl_std]
    
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
    print('dynamic retraining data adjustment')
    ranked = True  # set to True to rank by uncertainty, False to shuffle

    # thresholds from config.yaml
    config = ConfigLoader("config.yaml")
    energy_threshold = float(config['energy_std_threshold'])
    force_threshold  = float(config['force_rms_std'])

    if not to_orcl_buffer or not pred_list:
        print("Empty oracle buffer or prediction list; no adjustment made.")
        return to_orcl_buffer

    energy_std = []
    force_rms_std = []

    # Compute ensemble stds
    for k in pred_list:
        k = np.asarray(k)
        if k.ndim == 1:
            raise ValueError("pred_list must contain predictions from multiple models for each structure.")
        ddof = 1 if k.shape[0] > 1 else 0
        e_std = np.std(k[:, 0], ddof=ddof)
        f_std_vec = np.std(k[:, 1:], axis=0, ddof=ddof)
        f_std = float(np.sqrt(np.mean(f_std_vec**2)))
        energy_std.append(float(e_std))
        force_rms_std.append(float(f_std))

    # Filter by thresholds
    selected = [
        (i, e, f, to_orcl_buffer[i])
        for i, (e, f) in enumerate(zip(energy_std, force_rms_std))
        if (e > energy_threshold) or (f > force_threshold)
    ]

    # Rank or shuffle
    if ranked:
        selected.sort(key=lambda x: (x[2], x[1]), reverse=True)
        mode_msg = "ranked (desc by force_std, energy_std)"
    else:
        rng = np.random.default_rng(1234)
        rng.shuffle(selected)
        mode_msg = "shuffled"

    adjusted = [np.asarray(s[3], dtype=np.float64) for s in selected]

    print(f"After filtering: {len(adjusted)} kept out of {len(to_orcl_buffer)} → {mode_msg}")
    if adjusted and ranked:
        print("Top few (force_std, energy_std):",
              [(round(s[2], 4), round(s[1], 4)) for s in selected[:5]])

    return adjusted






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
# def get_init_data(path):
#     match = re.search(r"bi(\d+)(-?\d+)(?:_(?:samples|parsed))*\.csv", path)

#     if match:
#         num_atom = int(match.group(1))
#         charge = int(match.group(2))
#         print(f"num_atom: {num_atom}, charge: {charge}")
#     else:
#         print("Pattern not found")
#     data = pd.read_csv(path)
#     elements = data["atoms"].values
#     elements = [literal_eval(e) for e in elements]
#     elements_number = [[atomic_numbers[ei] for ei in e] for e in elements]
#     coords = data["coordinates"].values
#     coords = [np.array(np.matrix(c.replace('\n', ';'))).reshape((num_atom, 3)) for c in coords]
#     energies_0 = data['total_energy'].values
#     energies_0 = [literal_eval(e) for e in energies_0]
#     # convert = lambda a: np.array(np.matrix(a.replace('\n', ';'))) if type(a) == str else a
#     forces_0 = data['forces'].values
#     forces_0 = [np.array(np.matrix(c.replace('\n', ';'))).reshape((num_atom, 3)) for c in forces_0]
#     data_list = []  
#     for i in range(len(coords)):

#         data = [
#             torch.tensor(coords[i]), 
#             torch.tensor(elements_number[i]), 
#             torch.tensor(energies_0[i]), 
#             torch.tensor(forces_0[i]), 
#             torch.tensor(charge, dtype=torch.int64), 
#             torch.zeros(coords[i].shape),
#             None, 
#             [0,0],
#             torch.zeros(coords[i].shape)]
#         data_list.append(data)
#     # print('data_list:', data_list[0].forces)
#     return data_list

def get_full_data_init(path, source = None):
    if source is not None:
        data = pd.read_csv(path)
        data = data[data['source'] == source]
    else:
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
            [0,0],
            torch.zeros(coords[i].shape)]
        data_list.append(data)
    # print('data_list:', data_list[0].forces)
    return data_list




import csv
def get_specific_data(file_path, line_number):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

        # Filter out rows where source == "real" if "source" exists
        # if "source" in header:
        #     source_idx = header.index("source")
        #     rows = [r for r in rows if r[source_idx].strip().lower() != "real"]

        # Check range after filtering
        if type(line_number) == int:

            if line_number >= len(rows):
                raise IndexError("Line number out of range after filtering.")

            # Get the desired row
            row = rows[line_number]
            data = process_row(row, header)
        else:
            data = []
            for ln in line_number:
                if ln >= len(rows):
                    raise IndexError(f"Line number {ln} out of range after filtering.")
                row = rows[ln]
                processed_data = process_row(row, header)
                data.append(processed_data)
        return data

def process_row(row, header):
    elements = literal_eval(row[header.index("atoms")])
    elements_number = [atomic_numbers[ei] for ei in elements]
    charge = int(str(row[header.index("charge")]).split('(')[1].split(',')[0]) if 'tensor' in str(row[header.index("charge")]) else int(row[header.index("charge")])
    num_atom = len(elements)
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
        [0,0], 
        torch.zeros(coords.shape)
    ]
    
    return data


def compute_flat_length(metadata):
    total = 0
    for m in metadata:
        if m["type"] in ("array", "tensor"):
            # multiply all dims in shape
            size = 1
            for s in m["shape"]:
                size *= s
            total += size
        elif m["type"] in ("scalar", "scalar_nullable", "charge"):
            total += 1
        elif m["type"] == "list":
            total += m["shape"][0]
        else:
            raise ValueError(f"Unknown type {m['type']}")
    return total

def reconstruct_from_metadata(flat_array, metadata, none_placeholder=99999999.0, rank = None):
    reconstructed_data = []
    index = 0  # Start index for slicing flat_array


    for meta in metadata:
        meta_type = meta['type']

        if meta_type == 'array':
            shape = tuple(meta['shape'])
            num_elements = np.prod(shape)
            array_data = np.array(flat_array[index:index + num_elements], dtype=meta['dtype']).reshape(shape)
            reconstructed_data.append(array_data)
            index += num_elements

        elif meta_type == 'tensor':
            shape = tuple(meta['shape'])
            dtype = getattr(torch, meta['dtype'].split('.')[-1])  # e.g., 'torch.float64' → 'float64'
            num_elements = np.prod(shape) if shape else 1
            tensor_data = torch.tensor(flat_array[index:index + num_elements], dtype=dtype).reshape(shape)
            reconstructed_data.append(tensor_data)
            index += num_elements

        elif meta_type == 'charge':
            reconstructed_data.append(torch.tensor(flat_array[index], dtype=torch.int64))
            index += 1

        elif meta_type == 'None':
            if flat_array[index] == none_placeholder or flat_array[index] == int(none_placeholder):
                reconstructed_data.append(None)
            else:
                reconstructed_data.append(flat_array[index])
            index += 1

        elif meta_type == 'scalar_nullable':
            if flat_array[index] == none_placeholder:
                reconstructed_data.append(None)
            else:
                if meta['dtype'] == 'int':
                    reconstructed_data.append(int(flat_array[index]))
                elif meta['dtype'] == 'float':
                    reconstructed_data.append(float(flat_array[index]))
            index += 1

        elif meta_type == 'scalar':
            if meta['dtype'] == 'int':
                reconstructed_data.append(int(flat_array[index]))
            elif meta['dtype'] == 'float':
                reconstructed_data.append(float(flat_array[index]))
            else:
                raise ValueError(f"Unsupported scalar dtype: {meta['dtype']}")
            index += 1

        elif meta_type == 'list':
            shape = meta.get('shape')
            if shape is None:
                raise ValueError(f"metadata field '{meta.get('name','<list>')}' missing 'shape'; "
                                "cannot reconstruct list deterministically.")
            list_len = np.prod(shape)
            dtype = meta.get('dtype', 'float')
            segment = flat_array[index : index + list_len]

            if dtype == 'int':
                # Check if all entries are close to integers before casting
                if not np.all(np.isclose(segment, np.round(segment), atol=1e-6)):
                    print(f"[WARNING] rank {rank} Non-integer-like values in list intended as int: {segment}, {flat_array}")
                    raise ValueError(f"List intended as int contains non-integer-like values, malform/mismatched data recieved. ")
                int_list = np.round(segment).astype(int).tolist()
                reconstructed_data.append(int_list)

            elif dtype == 'float':
                float_list = segment.astype(float).tolist()
                reconstructed_data.append(float_list)

            else:
                raise ValueError(f"Unsupported list dtype: {dtype}")

            index += list_len

        else:
            raise ValueError(f"Unknown metadata type: {meta_type}")

    return reconstructed_data


def convert_to_1d_float_array(data):
    flat_array = []

    for item in data:
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


    return np.array(flat_array, dtype=np.float64)



import numpy as np

def parse_list_data_to_gene(list_data_to_gene, has_iter=False):
    """
    Parse list_data_to_gene structured as a list of arrays,
    each array with shape (num_predictors, 2 + 3*num_atoms) if has_iter=True
    (leading 'iteration' + 'energy' + flattened forces),
    or (num_predictors, 1 + 3*num_atoms) if has_iter=False.

    Returns:
        energy:   (G, P) float array
        forces:   (G, P, N, 3) float array
        iters:    (G, P) int array if has_iter=True, else None
    """
    data_array = np.array(list_data_to_gene)
    if data_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data_array.shape}")

    G, P, D = data_array.shape

    if has_iter:
        if (D - 2) % 3 != 0:
            raise ValueError(f"Last dim {D} not compatible with 'iter + energy + 3N'.")
        N = (D - 2) // 3

        # Ensure correct dtypes even if original array is object
        iters = data_array[:, :, 0].astype(int)
        energy = data_array[:, :, 1].astype(float)
        forces_flat = data_array[:, :, 2:].astype(float)
    else:
        if (D - 1) % 3 != 0:
            raise ValueError(f"Last dim {D} not compatible with 'energy + 3N'.")
        N = (D - 1) // 3

        iters = None
        energy = data_array[:, :, 0].astype(float)
        forces_flat = data_array[:, :, 1:].astype(float)

    forces = forces_flat.reshape(G, P, N, 3)
    return energy, forces, iters
