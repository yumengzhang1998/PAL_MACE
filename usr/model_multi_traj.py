#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:53:03 2023

@author: chen
"""
from copy import deepcopy
import gc
import logging
from math import e
import psutil
import numpy as np
import torch, time, os, json
from torch import nn
from PAL_MACE.usr import starting_point_pool
from usr.utils_multi_traj import  shuffle_dataset, save_data, get_full_data_init, compute_flat_length
from usr.initial_pyg.functions.config import ConfigLoader
from usr.initial_pyg.evaluation import evaluate
import sys
import pandas as pd
from ast import literal_eval
from ase.data import chemical_symbols

import matplotlib.pyplot as plt
from usr.utils_multi_traj import convert_to_data_object, reconstruct_from_metadata
import glob
import random
import sys
from sklearn.utils import resample
from mace.tools.load_from_var import (get_dataset_from_xyz_variable, 
                                      configure_model_without_scaleshift, 
                                      _build_model, 
                                      build_default_arg_parser_dict, 
                                      get_atomic_energies_from_data)
from mace.tools.scripts_utils import (
    LRScheduler,
    dict_to_array,
    get_avg_num_neighbors,
    get_config_type_weights,
    get_loss_fn,
    get_optimizer,
    get_params_options,
    get_swa,
    print_git_commit,
    setup_wandb,
    convert_to_json_format,
    extract_config_mace_model
)
from mace import data, tools, modules
from torch_ema import ExponentialMovingAverage
from typing import List, Optional
from torch.nn.parallel import DistributedDataParallel as DDP
from mace.tools import torch_geometric
from mace.tools.multihead_tools import (
    HeadConfig,
    assemble_mp_data,
    dict_head_to_dataclass,
    prepare_default_head,
)

import re
import ast
from mace.tools.utils import AtomicNumberTable
from torch.utils.data import ConcatDataset
from al_setting import AL_SETTING
def norm_from_serialized_force(x):
    if isinstance(x, str):
        arr = np.array(literal_eval(x))
    else:
        arr = np.array(x)
    return np.linalg.norm(arr)
def list_cuda_devices():
    if torch.cuda.is_available():
        print("Available CUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("❌ No CUDA devices available.")



def convert_atoms_column(df):
    def convert(atom_entry):
        # If it's a string, parse it
        if isinstance(atom_entry, str):
            atom_entry = ast.literal_eval(atom_entry)

        return [chemical_symbols[int(z)] for z in atom_entry]

    df["atoms"] = df["atoms"].apply(convert)
    return df

def reset_logging():
    """Reset logging to prevent duplicate messages in loops."""
    root_logger = logging.getLogger()
    
    # Remove only our own handlers, avoiding interference with external loggers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

def recursive_to(model, device, dtype=None):
    for name, module in model.named_modules():
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, torch.Tensor) and attr.device != device:
                try:
                    setattr(module, attr_name, attr.to(device=device, dtype=dtype or attr.dtype))
                except Exception:
                    pass  # some attributes are properties or immutable


def extract_e0_dict_from_log(log_path):
    with open(log_path, "r") as f:
        for line in f:
            if "Atomic Energies used" in line:
                # Extract the dictionary-like string
                match = re.search(r"\{.*\}", line)
                if match:
                    e0_str = match.group(0)
                    try:
                        e0_dict = ast.literal_eval(e0_str)
                        return e0_dict
                    except Exception as e:
                        raise ValueError(f"Failed to parse E0s: {e0_str}") from e
    raise ValueError("No atomic energy dictionary found in log.")

def block_print():
    sys.stdout = None

def log_memory_usage(rank):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Process {rank} - Memory Usage: RSS = {mem_info.rss / (1024 ** 2)} MB, VMS = {mem_info.vms / (1024 ** 2)} MB")
def tensor_to_serializable_force(force_list):
    """
    Converts a list of force tensors or arrays (shape: [n_atoms, 3]) to plain list-of-lists.
    Safe for molecules with varying atom counts.
    """
    serializable = []
    for force in force_list:
        if hasattr(force, "detach"):  # PyTorch tensor
            force = force.detach().cpu().numpy()
        elif hasattr(force, "numpy"):  # NumPy array
            force = force.astype(float)
        serializable.append(force.tolist())
    return serializable

def tensor_to_serializable_energy(energy_list):
    """
    Converts energy predictions to a list of floats.
    """
    return [float(e) for e in energy_list]

def flatten_and_concatenate(pred_list):
    """
    Flattens and concatenates all elements in a list of numpy arrays.
    
    Args:
        pred_list (list): A list of numpy arrays with the same shape.
        
    Returns:
        numpy.ndarray: A flattened 1D array containing all elements.
    """
    flattened_arrays = [pred.flatten() for pred in pred_list]
    return np.concatenate(flattened_arrays)
def combine_predictions_to_numpy_with_iter(y_pred, force_pred, iteration):
    """
    Combines y_pred and force_pred into a list of 1D numpy arrays.
    Each array will have length 14: 
    1 iteration index + 1 scalar energy + 12 flattened force values.

    Args:
    - y_pred (torch.Tensor): Tensor of shape (n,) or (n, 1) for predicted energies.
    - force_pred (torch.Tensor): Tensor of shape (n, 4, 3) for force predictions.
    - iteration (int): Iteration index to prepend.

    Returns:
    - result_list (list of np.ndarray): Each element is a 1D numpy array of length 14.
    """
    result_list = []
    for i in range(y_pred.shape[0]):
        iter_arr = np.array([iteration], dtype=int)           # shape (1,)
        energy = y_pred[i].reshape(1).detach().cpu().numpy()  # shape (1,)
        forces = force_pred[i].reshape(-1).detach().cpu().numpy()  # shape (12,)
        combined = np.concatenate((iter_arr, energy, forces))  # shape (14,)
        result_list.append(combined)

    return result_list

def combine_predictions_to_numpy(y_pred, force_pred):
    """
    Combines y_pred and force_pred into a list of 1D numpy arrays.
    Each array will have length 13: 1 scalar energy + 12 flattened force values.

    Args:
    - y_pred (torch.Tensor): Tensor of shape (n,) or (n, 1) for predicted energies.
    - force_pred (torch.Tensor): Tensor of shape (n, 4, 3) for force predictions.

    Returns:
    - result_list (list of np.ndarray): Each element is a 1D numpy array of length 13.
    """
    result_list = []
    for i in range(y_pred.shape[0]):
        energy = y_pred[i].reshape(1).detach().cpu().numpy()  # shape (1,)
        forces = force_pred[i].reshape(-1).detach().cpu().numpy()  # shape (12,)
        combined = np.concatenate((energy, forces))  # shape (13,)
        result_list.append(combined)

    return result_list

def poisson_bootstrap_indices(n, lam=1.0, ensure_inclusion=True, rng=None):
    """
    Return array of indices for a Poisson-weighted bootstrap.
    Every sample appears Poisson(lam) times; +1 ensures inclusion.
    """
    if rng is None:
        rng = np.random.default_rng()
    counts = rng.poisson(lam, size=n)
    if ensure_inclusion:
        counts = counts + 1
    # e.g. counts=[2,1,3,...] -> indices=[0,0,1,2,2,2,...]
    return np.repeat(np.arange(n, dtype=np.int64), counts)
class UserModel(object):
    """
    User defined model for both Passive Learner and Machine learning.
    Passive Learner:
        Receive inputs from Generator and make predictions.
        Receive model parameters from ML and update the model.
    Machine Learning:
        Receive inputs from Oracle and retrain the model.
        Output model parameters sent to PL.
    """
    def __init__(self, rank, result_dir, i_gpu, mode):
        """
        Initilize the model.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
            i_gpu (int): GPU index.
            mode (str): 'predict' for Passive Learner and 'train' for Machine Learning.
        """
        # set up model and basic settings
        self.rank = rank
        self.result_dir = result_dir
        self.mode = mode
        self.i_gpu = i_gpu
        self.boot_strap = True
        self.add_all_cluster = False
        self.fix_e0 = True
        self.iter_log = True
        self.num_retraining_instances = 0 # count how many times retraining happened for retraining
        self.counter = 0 # count how many times retraining happened for predictor
        # loss = "huber_energy_forces"

        pred_procs = AL_SETTING["pred_process"]
        orcl_procs = AL_SETTING["orcl_process"]
        gene_procs = AL_SETTING["gene_process"]
        ml_procs = AL_SETTING["ml_process"]

        self.metrcis_dir = f"{self.result_dir}/metrics_{rank}"
        os.makedirs(self.metrcis_dir, exist_ok=True)


        if mode == "predict":
            pred_start = 2  # After exchange (0) and manager (1)
            number = self.rank - pred_start
        elif mode == "train":
            ml_start = 2 + pred_procs + orcl_procs + gene_procs
            self.ml_start = ml_start
            number = self.rank - ml_start
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not (0 <= number < pred_procs):  # assuming pred_procs == ml_procs
            raise ValueError(f"Rank {self.rank} gave invalid model index {number} for mode {mode}")

        self.ml_device = torch.device(f"cuda:{number}" if torch.cuda.is_available() else 'cpu')
        # self.ml_device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # self.pred_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.pred_device = torch.device(f"cuda:{number}")
        else:
            self.pred_device = torch.device("cpu")
        self.device = self.ml_device if self.mode == "train" else self.pred_device
        print(f"Rank {self.rank}: Device is {self.device}")


        self.config = ConfigLoader("config.yaml")
        self.transforms = None
        # self.number_of_generators = self.config["number_of_generators"]
        self.metadata = self.config["metadata"]
        self.cluster_data_length = compute_flat_length(self.metadata)
        data_type = self.config["prefix"]
        if self.config["full_dataset"]:
            self.prefix = "bi0"
            log_file = f"usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/sample_{number}/logs/{self.prefix}_run-123.log" 
        else:
            self.prefix = self.config["prefix"]
            log_file = f"usr/initial_pyg/results/charge_embedding/{self.prefix}_logs/sample_{number}/logs/{self.prefix}_run-123.log"
        args = build_default_arg_parser_dict(self.config['args_dict']) 
        
        if self.fix_e0:
            args.E0s = extract_e0_dict_from_log(log_file)
        else :
            args.E0s = "average"
        self.args, input_log_messages = tools.check_args(args)
        self.args.heads = prepare_default_head(args)
        # self.args.loss = loss
        self.batch_size = self.args.batch_size
        compute_virials = self.args.loss in ("stress", "virials", "huber", "universal")
        self.args.compute_energy = True
        self.args.compute_forces = True
        self.args.compute_stress = False
        self.args.compute_dipole = False
        self.output_args = {
                "energy": self.args.compute_energy,
                "forces": self.args.compute_forces,
                "virials": compute_virials,
                "stress": self.args.compute_stress,
                "dipoles": self.args.compute_dipole,
            }
        # name and directory
        self.args.name = f'{self.prefix}_{self.rank}'
        self.args.results_dir = os.path.join(self.result_dir, f'rank_{self.rank}')
        args.checkpoints_dir = f"{self.args.results_dir}/checkpoints" 
        args.log_dir = f"{self.args.results_dir}/logs"
        args.model_dir = f"{self.args.results_dir}"
        if self.config["full_dataset"]:
            PATH = f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/sample_{number}/{self.prefix}.model'
        else:
            PATH = f'usr/initial_pyg/results/charge_embedding/{self.prefix}_logs/sample_{number}/{self.prefix}.model'
        self.load_model = bool(self.config['load_model'])
        self.load_dataset = bool(self.config['load_dataset'])
        if self.load_model:
            
            
            if self.mode == "train":
                PATH = f"results/{self.prefix}/model_{self.rank}.pt"
                print(f"[ML KERNEL]: Loading model from last AL run under {PATH}...")
            else:
                corresponding_ml_rank = self.rank + gene_procs + orcl_procs + pred_procs
                PATH = f"results/{self.prefix}/model_{corresponding_ml_rank}.pt"
                print(f"[Prediction process]: Loading model from last AL run under {PATH}...")
        print(f"✅ [Rank {self.rank}] ({mode}) → Loaded model {PATH}")
        self.model = torch.load(PATH, map_location=self.device)
        self.model = self.model.to(self.device)
        torch.set_default_dtype(torch.float64)
        recursive_to(self.model, device=self.device, dtype=torch.get_default_dtype())
        for param in self.model.parameters():
            param.data = param.data.to(dtype=torch.get_default_dtype())

        for buffer_name, buffer in self.model.named_buffers():
            if isinstance(buffer, torch.Tensor):
                setattr(self.model, buffer_name, buffer.to(dtype=torch.get_default_dtype(), device=self.device))
        if self.load_model and os.path.exists("al_state.json"):
            print(f"Rank {self.rank}: Loading AL state from previous run...")
            state = json.load(open("al_state.json"))
            self.pat_old = state["pat_old"]
            self.pat_new = state["pat_new"]
            self.best_mae = state["best_mae"]
            self.num_retraining_instances = state["num_retraining_instances"]
            self.counter = state["num_retraining_instances"]
        elif self.load_model and (not os.path.exists("al_state.json")):
            print("Rank {self.rank}: No previous AL state found, but load_model is True. Starting fresh.")
            self.pat_old = 0
            self.pat_new = 0
            self.best_mae = {
                                "init_e": float("inf"),
                                "init_f": float("inf"),
                                "added_e": float("inf"),
                                "added_f": float("inf"),
                            }
            self.num_retraining_instances = 0
            self.counter = 0
        else:
            self.num_retraining_instances = 0
            self.counter = 0
            print(f"Rank {self.rank}: No previous AL state found, starting fresh.")
        if self.mode == "predict":
            print('predicting', self.rank)
            # self.para_keys = list(self.model.state_dict().keys())
            self.batch_size = 128
        
        else:
            self.start_time = time.time()
            self.starting_pool_update = bool(self.config['starting_pool_update'])
            if not self.load_dataset:
                print('training', self.rank)
                if self.config["full_dataset"] and self.add_all_cluster:
                    print('full dataset')
                    init_data = get_full_data_init(f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/sample_{number}/train.csv')
                    self.val = get_full_data_init(f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/{self.prefix}.csv')
                    print(f'loading initial dataset that contains all cluster data')
                    print('initial dataset size', len(init_data))
                    print('validation dataset size', len(self.val))
                    print("Finished loading initial dataset")
                elif self.config["full_dataset"] and (not self.add_all_cluster):
                    print('full dataset but not all cluster')
                    print(f'loading initial dataset that only contains {data_type} data')
                    init_data = get_full_data_init(f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/sample_{number}/train.csv', source = data_type)
                    self.val = get_full_data_init(f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/{self.prefix}.csv', source= data_type)
                    print('initial dataset size', len(init_data))
                    print('validation dataset size', len(self.val))
                    print("Finished loading initial dataset")
                elif not self.config['full_dataset'] :
                    print('single cluster run')
                    init_data = get_full_data_init(f'usr/initial_pyg/samples/{self.prefix}/sample_{number}/train.csv')
                    self.val = get_full_data_init(f'usr/initial_pyg/samples/{self.prefix}/sample_{number}/val.csv')
                random.shuffle(init_data)

                self.train = init_data
            else:
                print(f"Loading dataset from last AL run under {self.result_dir}/{self.rank}_added_data.csv...")

                self.train = get_full_data_init(f"{self.result_dir}/{self.rank}_added_data.csv", source="train", source_column="type" )
                self.val = get_full_data_init(f"{self.result_dir}/{self.rank}_added_data.csv", source="val", source_column="type" )
                print(f"Finished loading dataset with {len(self.train)} training points and {len(self.val)} validation points.")
                

            self.val_split = 0.2
            self.history = {
                "MSE_train": [],
                "MSE_val": [],
                "start_MAE_train": [],
                "start_MAE_val": [],
                "init_val_mae": [],
                "new_val_mae": [],
                "new_val_force_mae": [],
                "init_val_force_mae": []
                }
            # --- Dual-patience with relative thresholds (Option B) ---
            self.pcfg = dict(
                old_limit=10000,            # patience for initial (protect forgetting)
                new_limit=10000,             # patience for added (drive AL progress)
                rel_added_delta=0.5/100, # require ≥0.5% improvement on added (energy OR force)
                init_tol=2.0/100,        # ≤2% worse on init = acceptable noise
                init_hi=5.0/100          # >5% worse on init = "worse a lot"
            )
            if not self.load_model:
                self.pat_old = 0
                self.pat_new = 0
                self.best_mae = {
                    "init_e": float("inf"),
                    "init_f": float("inf"),
                    "added_e": float("inf"),
                    "added_f": float("inf"),
                }

                self.init_val_size = len(self.val)
                self.init_train_size = len(self.train)
            elif self.load_model and os.path.exists("al_state.json"):
                self.init_train_size = state["init_train_size"]
                self.init_val_size = state["init_val_size"]
            else:
                self.init_train_size = len(self.train)
                self.init_val_size = len(self.val)


        self.para_keys = list(self.model.state_dict().keys())
        
        self.retrain_patience = 10
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        


        self.stop = False
        
            
    ##########################################
    #          Passive Learner Part          #
    ##########################################        
            
    def predict(self, list_data_to_pred):

        T = self.config["num_traj_per_gene"]
        flat_len = self.cluster_data_length
        per_traj_len = 1 + flat_len
        per_gen_len = T * per_traj_len

        # ---- Step 1: collect all flats ----
        item_info = []       # (format_type, n_flats)
        all_flats = []       # all flattened geometries
        flat_owner = []      # which original item each flat belongs to

        for item_idx, item in enumerate(list_data_to_pred):

            arr = np.asarray(item, dtype=float)

            # ---- Format A ----
            if arr.ndim == 1 and arr.size == per_gen_len:
                format_type = "A"
                blocks = arr.reshape(T, per_traj_len)
                flats = [b[1:] for b in blocks]

            # ---- Format B ----
            else:
                format_type = "B"
                if arr.ndim == 1:
                    flats = [arr]
                else:
                    flats = [np.asarray(x, dtype=float) for x in item]

            item_info.append((format_type, len(flats)))

            for f in flats:
                all_flats.append(f)
                flat_owner.append(item_idx)

        if len(all_flats) == 0:
            return []

        # ---- Step 2: reconstruct ALL at once ----
        traj_recs = [
            reconstruct_from_metadata(f, self.metadata, rank=f"predict {self.rank}")
            for f in all_flats
        ]

        data_objects = convert_to_data_object(traj_recs)

        # ---- Step 3: evaluate in batch ----
        y_pred, f_pred, _, _ = evaluate(
            self.model,
            data_objects,
            batch_size=256,  # now this finally matters
            device=self.device,
        )

        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        f_pred = np.asarray(f_pred, dtype=float).reshape(
            -1, self.config["num_atom"] * 3
        )

        # ---- Step 4: split back per item ----
        per_item_energy = [[] for _ in list_data_to_pred]
        per_item_force = [[] for _ in list_data_to_pred]

        for i, owner in enumerate(flat_owner):
            per_item_energy[owner].append(y_pred[i])
            per_item_force[owner].append(f_pred[i])

        final_outputs = []

        for item_idx, (format_type, _) in enumerate(item_info):

            pieces = []

            if format_type == "A":
                iter_val = int(self.counter) if self.iter_log else 0
                pieces.append(np.array([iter_val], float))

            for e, f in zip(per_item_energy[item_idx], per_item_force[item_idx]):
                pieces.append(np.array([e], float))
                pieces.append(f.reshape(-1))

            final_outputs.append(np.concatenate(pieces))

        return final_outputs


    def update(self, weight_array):
        """
        Update model/scalar with new weights in weight_array.
        """
        if self.iter_log:
            self.counter = int(weight_array[0])
            offset = 1
        else:
            offset = 0
        for k in self.para_keys:
            param = self.model.state_dict()[k]
            param_size = param.numel()
            if param.device != torch.device(self.device):
                param = param.to(self.device)
                # print(f"Rank {self.rank}: model updated on device {self.device} but parameters are on {param.device}")
            new_tensor = torch.tensor(
                weight_array[offset:offset + param_size], 
                dtype=param.dtype,
                device=param.device
            ).reshape(param.shape)
            self.model.state_dict()[k].copy_(new_tensor)
            offset += param_size

        print(f"Rank {self.rank}: model updated on device {self.device}, current iteration of retraining {self.counter}")

            
    def get_weight_size(self):
        """
        Return the size of model weight when unpacked as an 1-D numpy array.
        Used to send/receive weights through MPI.
        
        Returns:
            weight_size (int): size of model weight when unpacked as an 1-D numpy array.
        """
        weight_size = None
        
        ##### User Part #####
        if self.iter_log:
            weight_size = 1
        else:
            weight_size = 0
        # the last 4 key-item pairs are scalars
        for k in self.para_keys:
            weight_size += self.model.state_dict()[k].flatten().shape[0]
        return weight_size

    ###########################################
    #          Machine Learning Part          #
    ###########################################         

    def get_weight(self):
        """
        Return model/scalar weights as an 1-D numpy array.
        
        Returns:
            weight_array (numpy.ndarray): 1-D numpy array containing model/scalar weights. (to UserModel.update())
        """
        weight_array = None
        
        ##### User Part #####
        weight_array = []
        for k in self.para_keys:
            weight_array += self.model.state_dict()[k].detach().cpu().numpy().flatten().tolist()
        weight_array = np.array(weight_array, dtype=float)
        if self.iter_log:
            weight_array = np.concatenate((np.array([self.num_retraining_instances]), weight_array))
        return weight_array
    
    def add_trainingset(self, datapoints):
        """
        Increase the training set with set of data points.
        
        Args:
            datapoints (list): list of new training datapoints.
                               Format: [[input1 (1-D numpy.ndarray), target1 (1-D numpy.ndarray)], [input2 (1-D numpy.ndarray), target2 (1-D numpy.ndarray)], ...]
                               Source: input_for_orcl element of input_to_orcl_list from utils.prediction_check(). 
                                       orcl_calc_res from UserModel.run_calc().
        """
        ##### User Part #####
        # data_list = [item['data_list'] for item in datapoints if item['data_list'] is not None]
        data_list = []
        fail = 0
        for data in datapoints:
            if np.all(data[1] == 0):
                print('failed to get energy')
                # print('data:', data)
                fail += 1
                continue
            else:
                #input 1 array: (pos, z, energy, forces,charge, pred_forces, pred_energy, patience, velocity)
                #input 2 array: (y, forces)
                original_data = reconstruct_from_metadata(data[0], self.metadata)
                original_data[2] = torch.tensor(data[1][0]).reshape(-1)
                shape = original_data[3].shape
                original_data[3] = -1 * torch.tensor(data[1][1:].reshape(shape))
                del shape
                original_data[4] = torch.tensor(original_data[4], dtype=torch.int32)
                original_data[0] = torch.tensor(original_data[0], dtype=torch.get_default_dtype())
                #print(original_data)
                data_list.append(original_data)
        #data_list = [reconstruct_from_metadata(data, self.metadata) for data in datapoints]
        print('number of failed data in this iteration:', fail)
        for data in data_list:
            data[-3] = None           
        if self.boot_strap:
            train, val = shuffle_dataset(data_list)
            train = resample(train, replace=True, n_samples=len(train), random_state=123)
            self.train.extend(train)
            self.val.extend(val)
        else:
            train, val = shuffle_dataset(data_list)
            self.train.extend(train)
            self.val.extend(val)

            
        if len(self.train) > 1000000:
            print('1000000 training points reached')
            self.stop = True
        print(f"Rank {self.rank}: add_trainingset done. fail={fail}. train_len={len(self.train)} val_len={len(self.val)}", flush=True)

        print(f"Rank {self.rank}: training set size increased")
    
    def retrain(self, req_data):
        """
        Retrain the model with current training set.
        Retraining should stop before or when receiving new data points.
        
        Args:
            req_data (MPI.Request): MPI request object indicating status of receiving new data points.
        """
        ##### User Part #####
        print(f"Rank {self.rank}: retraining start...")
        self.model.to(self.ml_device) 
        if self.stop == True:
            stop_run = True
        for v in self.history.values():
            v.append([])
        
        # training datalaoader#
        # boot strap
        train = self.train.copy()
        val = self.val.copy()
        print('length of trainingset', len(train))
        print('length of the val set', len(val))
        # print('train                                              ', self.train[0].pos, self.train[0].z, self.train[0].charge, self.train[0].atoms, self.train[0].pred, self.train[0].y, self.train[0].forces)
        # print('val:                                                ', self.val[0].pos,  self.val[0].z, self.val[0].charge, self.val[0].atoms, self.val[0].pred, self.val[0].y, self.val[0].forces)
        train = convert_to_data_object(train)
        val = convert_to_data_object(val) 
        args = deepcopy(self.args)
        reset_logging()
        head_configs: List[HeadConfig] = []
        for head, head_args in args.heads.items():
            logging.info(f"=============    Processing head {head}     ===========")
            head_config = dict_head_to_dataclass(head_args, head, args)
            head_config.atomic_energies_dict = {}
            logging.info(
                f"Total number of configurations: train={len(train)}, valid={len(val)}, "
                # f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],"
            )
            head_configs.append(head_config)
        for head_config in head_configs:
            
            z_table_head = tools.get_atomic_number_table_from_zs_data(
                z
                for datas in (train,val)
                for data in datas
                for z in data.z
            )
            head_config.atomic_numbers = z_table_head.zs
            head_config.z_table = z_table_head
        # yapf: enable
        all_atomic_numbers = set()
        for head_config in head_configs:
            all_atomic_numbers.update(head_config.atomic_numbers)
        z_table = AtomicNumberTable(sorted(list(all_atomic_numbers)))
        logging.info(f"Atomic Numbers used: {z_table.zs}")
        atomic_energies_dict = {}
        for head_config in head_configs:
            assert head_config.E0s is not None, "Atomic energies must be provided"
            atomic_energies_dict[head_config.head_name] = get_atomic_energies_from_data(
                head_config.E0s, train, head_config.z_table
            )
        print(atomic_energies_dict)
        heads = list(args.heads.keys())
        atomic_energies = dict_to_array(atomic_energies_dict, heads)
        tools.set_seeds(args.seed)
        for head_config in head_configs:
            try:
                logging.info(f"Atomic Energies used (z: eV) for head {head_config.head_name}: " + "{" + ", ".join([f"{z}: {atomic_energies_dict[head_config.head_name][z]}" for z in head_config.z_table.zs]) + "}")
            except KeyError as e:
                raise KeyError(f"Atomic number {e} not found in atomic_energies_dict for head {head_config.head_name}, add E0s for this atomic number") from e
        valid_sets = {head: [] for head in heads}
        train_sets = {head: [] for head in heads}
        for head_config in head_configs:
            train_sets[head_config.head_name] = [
                data.AtomicData.from_data(
                    i, z_table=z_table, cutoff=args.r_max, heads=heads
                )
                for i in train
            ]
            valid_sets[head_config.head_name] = [
                    data.AtomicData.from_data(
                        k, z_table=z_table, cutoff=args.r_max, heads=heads
                    )
                    for k in val
                ]
            train_loader_head = torch_geometric.dataloader.DataLoader(
                dataset=train_sets[head_config.head_name],
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
                generator=torch.Generator().manual_seed(args.seed),
            )
            head_config.train_loader = train_loader_head
        # concatenate all the trainsets
        train_set = ConcatDataset([train_sets[head] for head in heads])
        train_sampler, valid_sampler = None, None

        train_loader = torch_geometric.dataloader.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=(train_sampler is None),
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        valid_loaders = {heads[i]: None for i in range(len(heads))}
        if not isinstance(valid_sets, dict):
            valid_sets = {"Default": valid_sets}
        for head, valid_set in valid_sets.items():
            valid_loaders[head] = torch_geometric.dataloader.DataLoader(
                dataset=valid_set,
                batch_size=args.valid_batch_size,
                sampler=None,
                shuffle=False,
                drop_last=False,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
                generator=torch.Generator().manual_seed(args.seed),
            )

        args.avg_num_neighbors = get_avg_num_neighbors(head_configs, args, train_loader, self.device)


        print('Optimizer init')
        param_options = get_params_options(args, self.model)
        optimizer: torch.optim.Optimizer
        optimizer = get_optimizer(args, param_options)  
        tag = tools.get_tag(name=args.name, seed=args.seed)        

        dipole_only = False
        loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
        lr_scheduler = LRScheduler(optimizer, self.args)
        self.train_sampler, self.valid_sampler = None, None
        checkpoint_handler = starting_point_pool.DummyCheckpointHandler(
                        directory=args.checkpoints_dir,
                        tag=tag,
                        keep=args.keep_checkpoints,
                        swa_start=args.start_swa,
                    )
        tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=0)
        logger = tools.MetricsLogger(
                directory=args.results_dir, tag=tag + "_train"
            )
        
        valid_sets = {head: [] for head in heads}
        train_sets = {head: [] for head in heads}

        start_epoch = 0
        ema: Optional[ExponentialMovingAverage] = None
        if args.ema:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=self.args.ema_decay)
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = args.lr

        metrics = tools.al_train.train(
            model= self.model,
            loss_fn= loss_fn,
            train_loader=train_loader,
            valid_loaders=valid_loaders,
            optimizer= optimizer,
            lr_scheduler= lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            eval_interval=args.eval_interval,
            start_epoch=start_epoch,
            max_num_epochs=args.max_num_epochs,
            logger=logger,
            patience=args.patience,
            save_all_checkpoints=args.save_all_checkpoints,
            output_args=self.output_args,
            device=self.device,
            swa=None,
            ema=ema,
            max_grad_norm=args.clip_grad,
            log_errors=args.error_table,
            log_wandb=args.wandb,
            distributed=args.distributed,
            distributed_model=None,
            train_sampler=self.train_sampler,
            rank=int(0),
        )
        if args.ema:
            ema.copy_to(self.model.parameters())
        logging.info("")
        logging.info(f"===========RANK {self.rank} FINISHED TRAINING NUM.{self.num_retraining_instances} ===========")
        logging.info("eveluation")

        # trainer.fit(self.model, data_module)
        # print(self.model._nn_scaler._p_fit_atom_selection.dtype)
        
        
        train_start_mse = metrics["train"][0]["mae_e"]
        val_start_mse = metrics["validation"][0]["mae_e"]

    
        self.num_retraining_instances += 1
        self.counter += 1

        init_val_mae, init_val_force_mae, new_val_mae, new_val_force_mae,train_mae, train_force_mae, val_mae, val_force_mae =self.save_dataset(path = os.path.join(self.result_dir, f"{self.rank}_added_data.csv"))
        self.history['init_val_mae'][-1].append(init_val_mae)
        self.history['init_val_force_mae'][-1].append(init_val_force_mae)
        self.history['new_val_mae'][-1].append(new_val_mae)
        self.history['new_val_force_mae'][-1].append(new_val_force_mae)

        self.history["MSE_val"][-1].append(val_mae)
        self.history['start_MAE_val'][-1].append(val_start_mse)
        self.history["MSE_train"][-1].append(train_mae)
        self.history['start_MAE_train'][-1].append(train_start_mse)
        self.save_progress()
    
        with open(os.path.join(self.metrcis_dir, f"metrics_{self.num_retraining_instances}.json"), 'w') as fh:
            json.dump(metrics, fh)
        # ----- Dual-patience early stopping (relative thresholds) -----
        init_e = float(init_val_mae)
        init_f = float(init_val_force_mae)
        add_e  = float(new_val_mae)        if new_val_mae is not None        else float("inf")
        add_f  = float(new_val_force_mae)  if new_val_force_mae is not None  else float("inf")

        # required relative improvement on added set (either energy OR force)
        added_improve = (
            self._rel_improved(add_e, self.best_mae["added_e"], self.pcfg["rel_added_delta"]) or
            self._rel_improved(add_f, self.best_mae["added_f"], self.pcfg["rel_added_delta"])
        )

        # relative worsening on init vs best so far
        rel_worse_init_e = self._rel_increase(init_e, self.best_mae["init_e"])
        rel_worse_init_f = self._rel_increase(init_f, self.best_mae["init_f"])

        init_worse_mild = (rel_worse_init_e >= self.pcfg["init_tol"]) or (rel_worse_init_f >= self.pcfg["init_tol"])
        init_worse_high = (rel_worse_init_e >= self.pcfg["init_hi"])  or (rel_worse_init_f >= self.pcfg["init_hi"])

        # did init improve at all this epoch (either energy or force)?
        init_improve = (init_e < self.best_mae["init_e"]) or (init_f < self.best_mae["init_f"])

        # ----- Rules -----
        if (not init_worse_mild or init_improve) and added_improve:
            # initial stable/improving AND added improving → reset both
            self.pat_old = 0
            self.pat_new = 0
            status = "🎯 init stable/improving & added improving → reset both"
        elif init_worse_high and added_improve:
            # protect initial distribution more
            self.pat_old += 1
            self.pat_new = 0
            status = "⚖️ init worsened a lot but added improved → pat_old+1, pat_new→0"
        elif init_worse_mild and not added_improve:
            # worse on init, no progress on added
            self.pat_old += 1
            self.pat_new += 1
            status = "🛑 init worsened & added no improvement → pat_old+1, pat_new+1"
        elif (not init_improve) and added_improve:
            # added improves, init flat (within tol)
            self.pat_new = 0
            status = "✅ added improving, init flat → pat_new→0"
        elif init_improve and not added_improve:
            # init improving, added flat → nudge AL progress
            self.pat_new += 1
            status = "➡️ init improving, added flat → pat_new+1"
        else:
            # both flat within noise
            self.pat_old += 1
            self.pat_new += 1
            status = "⏸️ both flat → pat_old+1, pat_new+1"

        print(f"[early-stop] {status} | pat_old={self.pat_old}/{self.pcfg['old_limit']} "
            f"pat_new={self.pat_new}/{self.pcfg['new_limit']} | "
            f"init(e,f)=({init_e:.4f},{init_f:.4f}) add(e,f)=({add_e:.4f},{add_f:.4f}) "
            f"| rel_worse_init(e,f)=({rel_worse_init_e:.3%},{rel_worse_init_f:.3%})")

        # Update bests AFTER deciding patience
        self._update_bests(init_e, init_f, add_e, add_f)

        # ----- Size gates (unchanged) -----
        num_data_minimum = int(self.config['retrain_size']) * 15 + self.init_train_size
        K = 0.1 # least 10% of initial training data
        num_data_max = (self.init_train_size) / K

        should_stop_dual = (self.pat_old >= self.pcfg["old_limit"]) or (self.pat_new >= self.pcfg["new_limit"])
        if should_stop_dual and len(self.train) >= num_data_minimum:
            print(f"Rank {self.rank}: dual-patience reached (pat_old={self.pat_old}, pat_new={self.pat_new})")
            self.stop = True

        if len(self.train) >= num_data_max:
            print(f"Rank {self.rank}: maximum training data size {num_data_max:.0f} reached")
            self.stop = True




        print(f"Rank {self.rank}: retraining stop.")
        stop_run = self.check_stop()
            
        return stop_run
    
    def _rel_increase(self, cur, best):
        """Relative increase vs best: (cur-best)/best, floor at 0. If best not set, return 0."""
        if not np.isfinite(best) or best <= 0.0:
            return 0.0
        return max(0.0, (cur - best) / best)

    def _rel_improved(self, cur, best, rel_delta):
        """Return True if (best-cur)/best > rel_delta. If best not set, accept any decrease."""
        if not np.isfinite(best) or best <= 0.0:
            return cur < best
        return ((best - cur) / best) > rel_delta

    def _update_bests(self, init_e, init_f, add_e, add_f):
        if np.isfinite(init_e) and init_e < self.best_mae["init_e"]:  self.best_mae["init_e"]  = init_e
        if np.isfinite(init_f) and init_f < self.best_mae["init_f"]:  self.best_mae["init_f"]  = init_f
        if np.isfinite(add_e)  and add_e  < self.best_mae["added_e"]: self.best_mae["added_e"] = add_e
        if np.isfinite(add_f)  and add_f  < self.best_mae["added_f"]: self.best_mae["added_f"] = add_f

            
    def save_progress(self, stop_run = False):
        """
        Save the current progress/data/state.
        Called everytime after retraining and receiving new data points.
        """
        ##### User Part #####
        with open(os.path.join(self.result_dir, f"retrain_history_{self.rank}.json"), 'w') as fh:
            json.dump(self.history, fh)
        with open(os.path.join(self.result_dir, f"retrain_history_{self.rank}_log.txt"), 'w') as file:
            file.write(f'retraining the {self.counter}th time\n')
        if self.mode == "train":
            PATH = os.path.join(self.result_dir, f"model_{self.rank}.pt")
            torch.save(self.model, PATH)
            print(f"Rank {self.rank}: model saved")
            al_state = {
                "num_retraining_instances": self.num_retraining_instances,
                "counter": self.counter,
                "pat_old": self.pat_old,
                "pat_new": self.pat_new,
                "best_mae": self.best_mae,
                "init_train_size": self.init_train_size,
                "init_val_size": self.init_val_size,
            }
            with open(f"{self.result_dir}/al_state_{self.rank}.json", "w") as f:
                json.dump(al_state, f, indent=2)

            
        if self.stop == True:
            self.save_dataset(path = os.path.join(self.result_dir, f"{self.rank}_added_data.csv"))

    def save_dataset(self, path):
        print("Saving dataset...")

        # Prepare DataFrames for train and val sets
        train_df = save_data(self.train)
        val_df = save_data(self.val)
        
        train_energy = _to_numeric_energy(train_df['energy'])
        val_energy = _to_numeric_energy(val_df['energy'])
        train_force = _to_numeric_forces(train_df['forces'])
        val_force = _to_numeric_forces(val_df['forces'])
        # Predict on train and val sets
        train_en_pred, train_force_pred, _, _ = evaluate(
            self.model,
            convert_to_data_object(self.train),
            batch_size=self.batch_size,
            device=self.device
        )
        val_en_pred, val_force_pred, _, _ = evaluate(
            self.model,
            convert_to_data_object(self.val),
            batch_size=self.batch_size,
            device=self.device
        )

        # Convert predictions to serializable formats
        train_en_pred = tensor_to_serializable_energy(train_en_pred)
        val_en_pred = tensor_to_serializable_energy(val_en_pred)

        train_force_pred = tensor_to_serializable_force(train_force_pred)
        val_force_pred = tensor_to_serializable_force(val_force_pred)

        # full MAE
        true_train_en = train_energy
        true_train_force = np.vstack(train_force)
        train_mae = np.mean(np.abs(true_train_en - train_en_pred))
        train_force_mae = np.mean(np.abs(true_train_force - np.vstack(train_force_pred)))
        print(f'training data MAE: {train_mae} eV')
        val_mae = np.mean(np.abs(val_energy - val_en_pred))
        true_val_force = np.vstack(val_force)
        val_force_mae = np.mean(np.abs(true_val_force - np.vstack(val_force_pred)))
        print(f'validation data MAE: {val_mae} eV')
        # initial vaidation data MAE
        init_true_val_en = val_energy.copy()[:self.init_val_size]
        init_pred_val_en = val_en_pred.copy()[:self.init_val_size]
        init_val_mae = np.mean(np.abs(init_true_val_en - init_pred_val_en))
        force_size = self.init_val_size * self.config['num_atom']
        init_true_val_force = true_val_force.copy()[:force_size]
        init_pred_val_force = np.vstack(val_force_pred.copy()[:self.init_val_size])
        init_val_force_mae = np.mean(np.abs(init_true_val_force - init_pred_val_force))
        print(f'initial validation data MAE: {init_val_mae} eV')
        # new validation data MAE
        new_true_val_en = val_energy.copy()[self.init_val_size:]
        new_pred_val_en = val_en_pred.copy()[self.init_val_size:]
        new_true_val_force = true_val_force.copy()[force_size:]
        new_pred_val_force = np.vstack(val_force_pred.copy()[self.init_val_size:])
        if len(new_true_val_en) > 0:
            new_val_mae = np.mean(np.abs(new_true_val_en - new_pred_val_en))
            new_val_force_mae = np.mean(np.abs(new_true_val_force - new_pred_val_force))
            print(f'new validation data MAE: {new_val_mae} eV')
        
        # Add predictions to dataframes
        train_df["pred_energy"] = train_en_pred
        train_df["pred_forces"] = train_force_pred
        train_df["type"] = ["train"] * len(train_df)
        train_df["init"] = [1] * self.init_train_size + [0] * (len(train_df) - self.init_train_size)

        val_df["pred_energy"] = val_en_pred
        val_df["pred_forces"] = val_force_pred
        val_df["type"] = ["val"] * len(val_df)
        val_df["init"] = [1] * self.init_val_size + [0] * (len(val_df) - self.init_val_size)

        # add starting point pool
        if self.starting_pool_update and self.rank == self.ml_start:
            print(f"rank {self.rank}: updating starting pool with top 10 largest force error data points")
            # build empty df with coloumn ["atoms", "coordinates", "total_energy", "forces", "charge"]
            start_pool_df = pd.DataFrame(columns=["atoms", "coordinates", "total_energy", "forces", "charge"])
            added_df  = train_df[train_df["init"] == 0]
            force_error = np.abs(
                added_df["forces"].apply(norm_from_serialized_force)
                - added_df["pred_forces"].apply(norm_from_serialized_force)
            )            # gte the top 10 largest force error rows as starting pool
            top_error_df = added_df.loc[force_error.nlargest(10).index] 
            # extract atoms, coordinates, total_energy, forces, charge from top_error_df and add to start_pool_df
            start_pool_df["atoms"] = top_error_df["atoms"]
            # change atoms number to atom type with periodic table
            start_pool_df = convert_atoms_column(start_pool_df)
            start_pool_df["coordinates"] = top_error_df["coordinates"]
            start_pool_df["total_energy"] = [[i] for i in top_error_df["energy"]]
            start_pool_df["forces"] = top_error_df["forces"]
            start_pool_df["charge"] = top_error_df["charge"]
            file_path = os.path.join(self.result_dir, f"starting_point_pool.csv")
            write_header = not os.path.exists(file_path)

            start_pool_df.to_csv(
                file_path,
                index=False,
                mode='a',
                header=write_header
            )
            # start_pool_df.to_csv(os.path.join(self.result_dir, f"starting_point_pool.csv"), index=False, mode = 'a')


        # Plot results
        self.plot(train_df, val_df)
        self.plot_force(train_df, val_df)

        # Concatenate and save
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        full_df.to_csv(path, index=False)

        print("Dataset saved at:", path)
        return init_val_mae, init_val_force_mae, new_val_mae if len(new_true_val_en) > 0 else None, new_val_force_mae if len(new_true_val_en) > 0 else None, train_mae, train_force_mae, val_mae, val_force_mae
        

    def check_stop(self):
        if time.time() - self.start_time >= 7200000000:
            print('time limit reached')
            self.stop = True
        if self.stop:
            print('stop signal received')
            print("save now the final dataset.....")
            self.save_dataset(path = os.path.join(self.result_dir, f"{self.rank}_added_data_finished.csv"))
            # save history
            self.save_progress(stop_run = True)
            return True
        print('continue running')
        return False

    def stop_run(self):
        """
        Called before the Training/Prediction process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        # self.save_dataset(path = os.path.join(self.result_dir, f"added_data.csv"))
        print(f'rank {self.rank} done')
    def plot(self, train_df, val_df):

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        axs[0].scatter(train_df['energy'], train_df['pred_energy'], alpha=0.5)
        axs[0].plot([train_df['energy'].min(), train_df['energy'].max()],
                    [train_df['energy'].min(), train_df['energy'].max()], 'r')
        axs[0].set_title("Training: True vs Predicted Energy")
        axs[0].set_xlabel("True Energy")
        axs[0].set_ylabel("Predicted Energy")
        
        axs[1].scatter(val_df['energy'], val_df['pred_energy'], alpha=0.5)
        axs[1].plot([val_df['energy'].min(), val_df['energy'].max()],
                    [val_df['energy'].min(), val_df['energy'].max()], 'r')
        axs[1].set_title("Validation: True vs Predicted Energy")
        axs[1].set_xlabel("DFT Energy")
        axs[1].set_ylabel("Predicted Energy")
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/{self.rank}_energy_pred.png")

    # def plot_force(self, train_df, val_df, reduce="norm"):
    #     # 生成可画的标量对
    #     tr_true, tr_pred = _pairs_from_series(train_df["force"], train_df["pred_forces"], reduce=reduce)
    #     va_true, va_pred = _pairs_from_series(val_df["force"],  val_df["pred_forces"],  reduce=reduce)

    #     fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    #     # Train
    #     axs[0].scatter(tr_true, tr_pred, alpha=0.5)
    #     lo = min(tr_true.min(), tr_pred.min())
    #     hi = max(tr_true.max(), tr_pred.max())
    #     axs[0].plot([lo, hi], [lo, hi])  # 不指定颜色，避免样式冲突
    #     axs[0].set_title("Training: True vs Predicted Force")
    #     axs[0].set_xlabel("True Force" + (" (‖F‖)" if reduce=="norm" else " (components)"))
    #     axs[0].set_ylabel("Predicted Force")

    #     # Val
    #     axs[1].scatter(va_true, va_pred, alpha=0.5)
    #     lo = min(va_true.min(), va_pred.min())
    #     hi = max(va_true.max(), va_pred.max())
    #     axs[1].plot([lo, hi], [lo, hi])
    #     axs[1].set_title("Validation: True vs Predicted Force")
    #     axs[1].set_xlabel("True Force" + (" (‖F‖)" if reduce=="norm" else " (components)"))
    #     axs[1].set_ylabel("Predicted Force")

    #     plt.tight_layout()
    #     plt.savefig(f"{self.result_dir}/{self.rank}_force_pred.png")
    #     plt.close(fig) 

    def plot_force(self, train_df, val_df):
        tr_true, tr_pred = _flatten_pair(train_df, 'forces', 'pred_forces')
        va_true, va_pred = _flatten_pair(val_df,   'forces', 'pred_forces')

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # --- Train ---
        axs[0].scatter(tr_true, tr_pred, alpha=0.5, s=6)
        if tr_true.size and tr_pred.size:
            lo = float(min(tr_true.min(), tr_pred.min()))
            hi = float(max(tr_true.max(), tr_pred.max()))
            axs[0].plot([lo, hi], [lo, hi])
        axs[0].set_title("Training: True vs Predicted Force (all components)")
        axs[0].set_xlabel("True Force (components)")
        axs[0].set_ylabel("Predicted Force (components)")

        # --- Val ---
        axs[1].scatter(va_true, va_pred, alpha=0.5, s=6)
        if va_true.size and va_pred.size:
            lo = float(min(va_true.min(), va_pred.min()))
            hi = float(max(va_true.max(), va_pred.max()))
            axs[1].plot([lo, hi], [lo, hi])
        axs[1].set_title("Validation: True vs Predicted Force (all components)")
        axs[1].set_xlabel("True Force (components)")
        axs[1].set_ylabel("Predicted Force (components)")

        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/{self.rank}_force_pred.png")
        plt.close(fig)
def _to_numeric_forces(col):
    """
    col: pandas Series where each row is either:
         - np.ndarray shape (n_i, 3) of floats, or
         - list of lists, or
         - string representation of the above.
    Returns: np.ndarray of shape (sum_i n_i, 3)
    """
    arrays = []
    for x in col.values:
        if isinstance(x, str):
            x = ast.literal_eval(x)  # parse string -> python list
        a = np.asarray(x, dtype=float)
        # ensure shape (..., 3)
        a = a.reshape(-1, 3)
        arrays.append(a)
    return np.vstack(arrays) if arrays else np.zeros((0, 3), dtype=float)

def _to_numeric_energy(col):
    """
    col: pandas Series with float or stringified float.
    Returns: 1D float np.ndarray
    """
    vals = []
    for x in col.values:
        if isinstance(x, str):
            x = ast.literal_eval(x) if x.strip().startswith('[') else x
        vals.append(float(x))
    return np.asarray(vals, dtype=float)

def _pairs_from_series(true_col, pred_col, reduce="norm"):
    """将(可能是标量或数组)的两列转换为成对的一维标量数组."""
    t_list, p_list = [], []
    for t, p in zip(true_col, pred_col):
        t = np.asarray(t)
        p = np.asarray(p)
        # 标量
        if t.ndim == 0 and p.ndim == 0:
            t_list.append(float(t))
            p_list.append(float(p))
        else:
            # 统一形状
            t = t.reshape(-1)
            p = p.reshape(-1)
            if reduce == "norm":
                # 如果每条是 3N 分量，可按(…,3)求范数；否则按整体范数
                if t.size % 3 == 0 and p.size % 3 == 0:
                    t = np.linalg.norm(t.reshape(-1, 3), axis=1)
                    p = np.linalg.norm(p.reshape(-1, 3), axis=1)
                else:
                    t = np.linalg.norm(t)
                    p = np.linalg.norm(p)
                # 如果是整体范数，上面得到标量；若是每原子范数，上面得到多个标量
                if np.ndim(t) == 0:
                    t_list.append(float(t))
                    p_list.append(float(p))
                else:
                    t_list.extend(t.tolist())
                    p_list.extend(p.tolist())
            elif reduce == "components":
                # 直接把所有分量展开比较
                t_list.extend(t.tolist())
                p_list.extend(p.tolist())
            else:
                raise ValueError("Unknown reduce method")
    return np.array(t_list, dtype=float), np.array(p_list, dtype=float)


def _to_1d_array(v):
    """Coerce a cell to a 1D float NumPy array.
       Handles real arrays/lists and stringified lists like '[[...],[...]]'."""
    if v is None:
        return np.empty((0,), dtype=float)
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)   # parse string -> Python list
        except Exception:
            # fallback: try to parse comma-separated scalars
            try:
                return np.fromstring(v, sep=',', dtype=float)
            except Exception:
                return np.empty((0,), dtype=float)
    a = np.asarray(v)
    if a.size == 0:
        return np.empty((0,), dtype=float)
    # If it looks like (..., 3) force-components; otherwise just ravel
    return a.reshape(-1).astype(float, copy=False)

def _flatten_pair(df, true_col, pred_col):
    # Flatten each cell, concatenate all rows
    x_chunks, y_chunks = [], []
    for tx, py in zip(df[true_col].values, df[pred_col].values):
        x_chunks.append(_to_1d_array(tx))
        y_chunks.append(_to_1d_array(py))
    x = np.concatenate(x_chunks) if x_chunks else np.array([], dtype=float)
    y = np.concatenate(y_chunks) if y_chunks else np.array([], dtype=float)
    # align and filter finite
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]