#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:53:03 2023

@author: chen
"""
from copy import deepcopy
import logging
import psutil
import numpy as np
import torch, time, os, json
from torch import nn
from usr.utils import get_init_data, shuffle_dataset, save_data, get_full_data_init
from usr.initial_pyg.functions.config import ConfigLoader
from usr.initial_pyg.evaluation import evaluate
import sys
import pandas as pd

import matplotlib.pyplot as plt
from usr.utils import convert_to_data_object, reconstruct_from_metadata
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

def list_cuda_devices():
    if torch.cuda.is_available():
        print("Available CUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("❌ No CUDA devices available.")


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

# def combine_predictions_to_numpy(y_pred, force_pred):
#     """
#     Combines y_pred and force_pred into a list of 1D numpy arrays.
#     Each array will have length 13, where the first element is y_pred 
#     and the rest are the flattened force_pred values.
    
#     Args:
#     - y_pred (torch.Tensor): Tensor of shape (n, 1) for predicted values.
#     - force_pred (torch.Tensor): Tensor of shape (n, 4, 3) for force predictions.
    
#     Returns:
#     - result_list (list of np.ndarray): A list where each element is a 1D numpy array 
#                                         of length 13.
#     """
#     n = y_pred.shape[0]  # Number of predictions
#     result_list = []
#     if force_pred.ndim == 2 :
#         force_pred = np.expand_dims(force_pred, axis=0)
#     if y_pred.ndim == 1:
#         y_pred = np.expand_dims(y_pred, axis=0)
#     for i in range(n):
#         # Flatten force_pred[i] (shape (1, 4, 3) -> (12,))
#         force_pred_flat = force_pred[i].reshape(-1)
        
#         # Concatenate y_pred[i] (scalar) with flattened force_pred[i]
#         combined = np.concatenate((y_pred[i].reshape(-1), force_pred_flat))
        
#         # Convert to numpy array and append to the list
#         result_list.append(combined)
    
#     return result_list
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
            number = self.rank - ml_start
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not (0 <= number < pred_procs):  # assuming pred_procs == ml_procs
            raise ValueError(f"Rank {self.rank} gave invalid model index {number} for mode {mode}")

        self.ml_device = torch.device(f"cuda:{number}" if torch.cuda.is_available() else 'cpu')
        # self.ml_device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # self.pred_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_device = torch.device('cpu')
        self.device = self.ml_device if self.mode == "train" else self.pred_device
        print(f"Rank {self.rank}: Device is {self.device}")


        self.config = ConfigLoader("config.yaml")
        self.transforms = None
        # self.number_of_generators = self.config["number_of_generators"]
        self.metadata = self.config["metadata"]
        if self.config["full_dataset"]:
            self.prefix = "bi0"
        else:
            self.prefix = self.config["prefix"]
        args = build_default_arg_parser_dict(self.config['args_dict']) 
        log_file = f"usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/sample_{number}/logs/{self.prefix}_run-123.log" 
        args.E0s = extract_e0_dict_from_log(log_file)
        self.args, input_log_messages = tools.check_args(args)
        self.args.heads = prepare_default_head(args)
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
        PATH = f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/sample_{number}/{self.prefix}.model'
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


        if self.mode == "predict":
            print('predicting', self.rank)
            # self.para_keys = list(self.model.state_dict().keys())
            self.batch_size = 16
        
        else:
            self.start_time = time.time()
            self.counter = 0
            
            print('training', self.rank)
            if self.config["full_dataset"]:
                print('full dataset')
                init_data = get_full_data_init(f'usr/initial_pyg/raw/{self.prefix}_parsed.csv')
                self.val = get_full_data_init(f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/{self.prefix}.csv')
                print("Finished loading initial dataset")
            else:

                init_data = get_init_data(f'usr/initial_pyg/raw/{self.prefix}_parsed.csv')
                self.val = get_init_data(f'usr/initial_pyg/full_data_charge_embed/{self.prefix}_logs/{self.prefix}.csv')
            random.shuffle(init_data)

            self.train = init_data
            

            self.val_split = 0.2
            
            self.history = {
                "MSE_train": [],
                "MSE_val": []
                }




        self.para_keys = list(self.model.state_dict().keys())
        self.num_retraining_instances = 0
        self.retrain_patience = 10
        self.best_val_loss = float('inf')
        self.patience_counter = 0


        self.stop = False
        
            
    ##########################################
    #          Passive Learner Part          #
    ##########################################        
            
    def predict(self, list_data_to_pred):
        """
        Make prediction for list of inputs from Generator.
        
        Args:
            list_data_to_pred (list): list of user defined model inputs. [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
                               size is equal to number of generator processes
                               Source: list of data_to_pred from UserModel.generate_new_data().
            
        Returns:
            data_to_gene_list (list): predictions returned to Generator. [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
                                      size should be equal to number of generator processes
                                      Destination: list of data_to_gene at UserModel.generate_new_data().
        """
        data_to_gene = None

        ##### User Part #####
        # print(list_data_to_pred)
        data_list = [reconstruct_from_metadata(data, self.metadata) for data in list_data_to_pred]
        # print('data_list', data_list)
        data_list = convert_to_data_object(data_list)
        # dataset = retrain_dataset(data_list, transforms=self.transforms)
        # test_loader = DataLoader(dataset, batch_size = 64)
        

        y_pred, force_pred, _, _= evaluate(self.model, data_list, batch_size = self.batch_size, device = self.device)
        a = torch.tensor(y_pred)
        b = torch.tensor(force_pred)
        # print(a)
        # print(b)
        # flattened_y_pred = flatten_and_concatenate(y_pred)
        # flattened_force_pred = flatten_and_concatenate(force_pred)


        # data_to_gene = np.concatenate([flattened_y_pred, flattened_force_pred])

        # #TODO: check if this is correct
        # print("energy", a, a.shape)
        # print("forces", b, b.shape)
        #print('length of list_data_to_pred', len(list_data_to_pred))
        # num_data = len(list_data_to_pred)
        # # reshape force_pred to (num_data, 4, 3)
        # force_pred = b.reshape(num_data, -1, 3)
        data_to_gene = combine_predictions_to_numpy(a, b)
        
        return data_to_gene
    
    # def update(self, weight_array):
    #     """
    #     Update model/scalar with new weights in weight_array.
        
    #     Args:
    #         weight_array (numpy.ndarray): 1-D numpy array containing model/scalar weights. (from UserModel.get_weight())
    #     """
    #     ##### User Part #####
    #     for k in self.para_keys:
    #         self.model.state_dict()[k] = weight_array[:self.model.state_dict()[k].flatten().shape[0]].reshape(self.model.state_dict()[k].shape)
    #     print(f"Rank {self.rank}: model updated")
    def update(self, weight_array):
        """
        Update model/scalar with new weights in weight_array.
        """
        offset = 0
        for k in self.para_keys:
            param = self.model.state_dict()[k]
            param_size = param.numel()
            if param.device != self.device:
                param = param.to(self.device)
                print(f"Rank {self.rank}: model updated on device {self.device} but parameters are on {param.device}")
            new_tensor = torch.tensor(
                weight_array[offset:offset + param_size], 
                dtype=param.dtype,
                device=param.device
            ).reshape(param.shape)
            self.model.state_dict()[k].copy_(new_tensor)
            offset += param_size

        print(f"Rank {self.rank}: model updated on device {self.device}")

            
    def get_weight_size(self):
        """
        Return the size of model weight when unpacked as an 1-D numpy array.
        Used to send/receive weights through MPI.
        
        Returns:
            weight_size (int): size of model weight when unpacked as an 1-D numpy array.
        """
        weight_size = None
        
        ##### User Part #####
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
        return np.array(weight_array, dtype=float)
    
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
                
        train, val = shuffle_dataset(data_list)

        self.train.extend(train)
        self.val.extend(val)
        if len(self.train) > 10000:
            print('10000 training points reached')
            self.stop = True
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
        self.counter += 1
        # training datalaoader#
        # boot strap
        if self.boot_strap:
            print('bootstrapping')
            train = resample(self.train, n_samples = len(self.train))
            val = resample(self.val, n_samples = len(self.val))
        else: 
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
        checkpoint_handler = tools.CheckpointHandler(
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
        logging.info("")
        logging.info(f"===========RANK {self.rank} FINISHED TRAINING NUM.{self.num_retraining_instances} ===========")
        logging.info("eveluation")

        # trainer.fit(self.model, data_module)
        # print(self.model._nn_scaler._p_fit_atom_selection.dtype)
        
        train_mse = metrics["train"][-1]["mae_e"]

        val_mse =  metrics["validation"][-1]["mae_e"]

        self.history["MSE_val"][-1].append(val_mse)
        self.history["MSE_train"][-1].append(train_mse)
        print(self.history)
        self.num_retraining_instances += 1


        with open(os.path.join(self.metrcis_dir, f"metrics_{self.num_retraining_instances}.json"), 'w') as fh:
            json.dump(metrics, fh)
        if val_mse < self.best_val_loss:
            self.best_val_loss = val_mse
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.retrain_patience and len(self.train) >= 4000:
            print(f"Rank {self.rank}: retraining patience reached")
            self.stop = True


        print(f"Rank {self.rank}: retraining stop.")
        stop_run = self.check_stop()

        self.save_dataset(path = os.path.join(self.result_dir, f"added_data.csv"))
        self.save_progress()
            
        return stop_run
            
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
            
        if self.stop == True:
            self.save_dataset(path = os.path.join(self.result_dir, f"added_data.csv"))

    def save_dataset(self, path):
        print("Saving dataset...")

        # Prepare DataFrames for train and val sets
        train_df = save_data(self.train)
        val_df = save_data(self.val)

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

        # Add predictions to dataframes
        train_df["pred_energy"] = train_en_pred
        train_df["pred_forces"] = train_force_pred
        train_df["type"] = ["train"] * len(train_df)

        val_df["pred_energy"] = val_en_pred
        val_df["pred_forces"] = val_force_pred
        val_df["type"] = ["val"] * len(val_df)

        # Plot results
        self.plot(train_df, val_df)

        # Concatenate and save
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        full_df.to_csv(path, index=False)

        print("Dataset saved at:", path)

    def check_stop(self):
        if time.time() - self.start_time >= 36000:
            print('time limit reached')
            self.stop = True
        if self.stop:
            print('stop signal received')
            print("save now the final dataset.....")
            self.save_dataset(path = os.path.join(self.result_dir, f"added_data_finished.csv"))
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
        axs[1].set_xlabel("True Energy")
        axs[1].set_ylabel("Predicted Energy")
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/{self.rank}_energy_pred.png")