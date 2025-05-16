
from math import log
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.dont_write_bytecode = True
from functions.config import ConfigLoader
from data import  big_list, split_data
import pickle
import pandas as pd
import random
from sklearn.utils import resample
from sklearn.metrics import r2_score
###########################################################################################
# Training script for MACE
# Authors: Yumeng Zhang
# This script use random data to debug, load data from Variable instead of xyz file or h5 file
# PROBELM 1: The strcuture is redundant, I don't need fine-tuning part and multihead part
# PROBELM 2: MACE model takes partial charges as charge input, but for me I need global charges
# PROBELM 3: The simplified version of code: currently MACE convert config to Data object, I am already using data object, I need to directly convert my data object to MACE strcutre data
###########################################################################################
import argparse
import ast
import glob
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
import numpy as np
from torch_geometric.data import Data


import torch.distributed
import torch.nn.functional
from e3nn.util import jit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from torch_ema import ExponentialMovingAverage
import mace
from mace import data, tools, modules
from mace.calculators.foundations_models import mace_mp, mace_off
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from mace.tools.multihead_tools import (
    HeadConfig,
    assemble_mp_data,
    dict_head_to_dataclass,
    prepare_default_head,
)
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
from mace.tools.slurm_distributed import DistributedEnvironment
from mace.tools.tables_utils import create_error_table
from mace.tools.utils import AtomicNumberTable

from mace.tools.load_from_var import get_dataset_from_xyz_variable, configure_model_without_scaleshift, _build_model, build_default_arg_parser_dict, get_atomic_energies_from_data

from evaluation import evaluate
from plot import plot_distribution, plot_true_vs_predicted

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


def reset_logging():
    """Reset logging to prevent duplicate messages in loops."""
    root_logger = logging.getLogger()
    
    # Remove only our own handlers, avoiding interference with external loggers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    





def rmse_forces(forces_true, forces_pred):
    return np.sqrt(np.mean(np.linalg.norm(forces_true - forces_pred, axis=1) ** 2))

def run(args: argparse.Namespace, train, val, test = None, latent = False, charge_penalty = None) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    reset_logging()
    args = deepcopy(args)
    train = deepcopy(train)
    val = deepcopy(val)
    tag = tools.get_tag(name=args.name, seed=args.seed)
    args, input_log_messages = tools.check_args(args)


    if args.device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError as e:
            raise ImportError(
                "Error: Intel extension for PyTorch not found, but XPU device was specified"
            ) from e
    if args.distributed:
        try:
            distr_env = DistributedEnvironment()
        except Exception as e:  # pylint: disable=W0703
            logging.error(f"Failed to initialize distributed environment: {e}")
            return
        world_size = distr_env.world_size
        local_rank = distr_env.local_rank
        rank = distr_env.rank
        if rank == 0:
            print(distr_env)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = int(0)

    # Setup
    tools.set_seeds(args.seed)
    if latent:
        print(f"Using latent charge and penalty {charge_penalty}")
        args.loss = "plus_charge"
        args.charge_penalty_weight = charge_penalty
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)

    if args.distributed:
        torch.cuda.set_device(local_rank)
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    commit = print_git_commit()
    model_foundation: Optional[torch.nn.Module] = None
    args.multiheads_finetuning = False

    if args.heads is not None:
        if isinstance(args.heads, str):
            args.heads = ast.literal_eval(args.heads)  # Convert string to dict
        print('heads')
    else:
        args.heads = prepare_default_head(args)
        print('default')

    logging.info("===========LOADING INPUT DATA===========")
    heads = list(args.heads.keys())
    logging.info(f"Using heads: {heads}")
    head_configs: List[HeadConfig] = []
    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        head_config = dict_head_to_dataclass(head_args, head, args)
        # if head_config.statistics_file is not None:

        # Data preparation

        config_type_weights = get_config_type_weights(
            head_config.config_type_weights
        )
        # collections, atomic_energies_dict = get_dataset_from_xyz_variable(
        #     train_list = train,
        #     valid_list = val,
        #     valid_fraction = head_config.valid_fraction,
        #     seed = args.seed,
        #     config_type_weights = config_type_weights,
        #     head_name = head_config.head_name,
        # )

        # I don't have atomic energies, so this is {}
        head_config.atomic_energies_dict = {}
        logging.info(
            f"Total number of configurations: train={len(train)}, valid={len(val)}, "
            # f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],"
        )
        head_configs.append(head_config)



    # Atomic number table
    # yapf: disable
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
    if args.foundation_model_elements and model_foundation:
        z_table = AtomicNumberTable(sorted(model_foundation.atomic_numbers.tolist()))
    logging.info(f"Atomic Numbers used: {z_table.zs}")

    # Atomic energies
    atomic_energies_dict = {}
    for head_config in head_configs:
        assert head_config.E0s is not None, "Atomic energies must be provided"
        atomic_energies_dict[head_config.head_name] = get_atomic_energies_from_data(
            head_config.E0s, train, head_config.z_table
        )
    print(atomic_energies_dict)

    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            args.compute_dipole = True
            args.compute_energy = True
            args.compute_forces = True
            args.compute_virials = False
            args.compute_stress = False
        else:
            args.compute_energy = True
            args.compute_dipole = False
        # atomic_energies: np.ndarray = np.array(
        #     [atomic_energies_dict[z] for z in z_table.zs]
        # )
        atomic_energies = dict_to_array(atomic_energies_dict, heads)
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
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        valid_samplers = {}
        for head, valid_set in valid_sets.items():
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
            valid_samplers[head] = valid_sampler
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
            sampler=valid_samplers[head] if args.distributed else None,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

    loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
    args.avg_num_neighbors = get_avg_num_neighbors(head_configs, args, train_loader, device)

    # Model

    model_config, output_args = configure_model_without_scaleshift(args, atomic_energies, model_foundation, heads, z_table)
    if args.scaling == "no_scaling":    
        args.std = 1.0
        logging.info("No scaling selected")
    elif (args.mean is None or args.std is None) and args.model != "AtomicDipolesMACE":
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
                 )
        logging.info(f"Mean: {args.mean}, Std: {args.std}")

    else:
        print("Using provided mean and std")
        logging.info(f"Using provided mean: {args.mean}, std: {args.std}")
    
    if not args.scale:
        args.std = 1.0

    if not args.shift:
        args.mean = 0.0
    print(args.mean, args.std)
    logging.info(f"Scale: {args.std}, Shift: {args.mean}")
    model = _build_model(args, model_config, model_config_foundation = None, heads = heads)
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Cueq
    if args.enable_cueq:
        logging.info("Converting model to CUEQ for accelerated training")
        assert model.__class__.__name__ in ["MACE", "ScaleShiftMACE"]
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
    # Optimizer
    param_options = get_params_options(args, model)
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)
    if args.device == "xpu":
        logging.info("Optimzing model and optimzier for XPU")
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swa, swas = get_swa(args, model, optimizer, swas, dipole_only)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if args.wandb:
        setup_wandb(args)
    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None

    metrcis = tools.al_train.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        train_sampler=train_sampler,
        rank=rank,
    )
    pickle.dump(metrcis, open(f"{args.results_dir}/metrics.pkl", "wb"))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print( model.scale_shift.scale,  model.scale_shift.shift)
    logging.info("")
    logging.info("===========FINISHED TRAINING===========")
    logging.info("eveluation")
    logging.info("The scale is {} and shift is {}".format(args.std, args.mean))
    energy, forces, stress, contribution = evaluate(model=model, eval_dataset=val, batch_size=args.batch_size, default_dtype=args.default_dtype, device=device, compute_stress=False, return_contributions=False)
    true_energy = [data.y for data in val]
    true_energy = torch.cat(true_energy, dim=0).cpu().tolist()
    logging.info(f"Energy Mean: {np.mean(np.array(true_energy))}, Variance: {np.var(np.array(true_energy))}")
    true_forces = [data.forces for data in val]
    true_forces = torch.cat(true_forces, dim=0).cpu()
    plot_true_vs_predicted(true_energy,energy, args.results_dir ,'Energy')
    #calculate energy RMSE and R2, energy is a list

    energy = torch.tensor(energy, dtype=torch.float32)
    plot_distribution(energy.cpu().tolist(), args.results_dir, 'prediction energy')
    true_energy = torch.tensor(true_energy, dtype=torch.float32)
    energy_rmse = torch.nn.MSELoss()(energy, true_energy).sqrt().item()
    energy_r2 = r2_score(true_energy, energy)
    energy_mae = torch.nn.L1Loss()(energy, true_energy).item()
    logging.info(f"Energy RMSE: {energy_rmse}")
    logging.info(f"Energy R2: {energy_r2}")
    logging.info(f"Energy MAE: {energy_mae}")
    
    #calculate forces RMSE and R2
    num_val = len(val)
    print(num_val)

    true_forces = true_forces.reshape(num_val, -1, 3)
    forces = np.array(forces).reshape(num_val, -1, 3)
    print(forces.shape)
    print(true_forces.shape)
    true_forces = torch.tensor(true_forces, dtype=torch.float32)
    forces_rmse = rmse_forces(true_forces, forces)
    forces_r2 = r2_score(true_forces.flatten(),forces.flatten())
    logging.info(f"Forces RMSE: {forces_rmse}")
    logging.info(f"Forces R2: {forces_r2}")   
    # train evaluation
    energy, forces, stress, contribution = evaluate(model=model, eval_dataset=train, batch_size=args.batch_size, default_dtype=args.default_dtype, device=device, compute_stress=False, return_contributions=False)
    true_energy = [data.y for data in train]
    true_energy = torch.cat(true_energy, dim=0).cpu().tolist()
    logging.info(f"Energy Mean: {np.mean(np.array(true_energy))}, Variance: {np.var(np.array(true_energy))}")
    true_forces = [data.forces for data in train]
    true_forces = torch.cat(true_forces, dim=0).cpu()
    plot_true_vs_predicted(true_energy,energy, args.results_dir ,'Train_Energy')



    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        if args.distributed:
            distributed_model = DDP(model, device_ids=[local_rank])
        model_to_evaluate = model if not args.distributed else distributed_model
        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        for param in model.parameters():
            param.requires_grad = False


        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            model_to_save = deepcopy(model)
            if args.enable_cueq:
                print("RUNING CUEQ TO E3NN")
                print("swa_eval", swa_eval)
                model_to_save = run_cueq_to_e3nn(deepcopy(model), device=device)
            if args.save_cpu:
                model_to_save = model_to_save.to("cpu")
            torch.save(model_to_save, model_path)
            extra_files = {
                "commit.txt": commit.encode("utf-8") if commit is not None else b"",
                "config.yaml": json.dumps(
                    convert_to_json_format(extract_config_mace_model(model))
                ),
            }
            if swa_eval:
                torch.save(
                    model_to_save, Path(args.model_dir) / (args.name + "_stagetwo.model")
                )
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_stagetwo_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    pass
            else:
                torch.save(model_to_save, Path(args.model_dir) / (args.name + ".model"))
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata to {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    pass

        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()

def boot_train(prefix, num_samples, config_path, res_dir="nocharge", latent = False, charge_penalty = None):

    import re
    numbers = re.findall(r"-?\d+", prefix)
    numbers = list(map(int, numbers))

    config = ConfigLoader(config_path)
    args_dict = config['args_dict']
    args = build_default_arg_parser_dict(args_dict)   
    print(args.swa)

    results_dir = f"{res_dir}/{prefix}_logs/"
    samples_dir = f"samples/"
    os.makedirs(results_dir, exist_ok=True)
    if os.path.exists(samples_dir):
        print(f"Results directory {samples_dir} already exists. Exiting.")
        
    else:
        os.makedirs(samples_dir, exist_ok=True)
        print(f"Results directory {samples_dir} created.")
    # Load raw dataset
    dataset = big_list(raw_data_path=f'./raw/{prefix}_parsed.csv',
                       num_atom=numbers[0],
                       charge=numbers[1],
                       transform=None,
                       pre_transform=None,
                       pre_filter=None)

    # Split raw into fixed validation set and trainable pool
    validation_size = 50
    data_list = dataset.data_list.copy()
    random.shuffle(data_list)
    val_data, train_data = data_list[:validation_size], data_list[validation_size:]



    # CSV version of val set for readability
    val_df = pd.DataFrame([{
        'atoms': data.atoms,
        'coordinates': data.pos.numpy().tolist(),
        'total_energy': data.y.numpy().tolist(),
        'forces': data.forces.numpy().tolist()
    } for data in val_data])
    val_df.to_csv(f'{results_dir}/{prefix}.csv', index=False, header=['atoms', 'coordinates', 'total_energy', 'forces'])

    csv_data = pd.read_csv(f'./raw/{prefix}_parsed.csv')
    has_source = 'source' in csv_data.columns
    if has_source:
        for data_obj, source_value in zip(data_list, csv_data['source']):
            data_obj.source = source_value
    del csv_data

    for i in range(num_samples):
        current_sample_dir = f"samples/{prefix}/sample_{i}/"
        result_of_sample_dir = f"{results_dir}/sample_{i}/"
        if not os.path.exists(current_sample_dir):
            os.makedirs(current_sample_dir)

        sample_file = f"{current_sample_dir}/train_val.pkl"

        if os.path.exists(sample_file):
            print(f"Loading saved bootstrap sample {i}")
            with open(sample_file, "rb") as f:
                train, val = pickle.load(f)
        else:
            print(f"Creating bootstrap sample {i}")
            if has_source:
                synthetic = [d for d in train_data if d.source == 'synthesis']
                real = [d for d in train_data if d.source == 'real']
                synthetic_boot = resample(synthetic, replace=True, n_samples=len(synthetic), random_state=i)
                real_boot = resample(real, replace=True, n_samples=len(real), random_state=i)
                bootstrap_dataset = synthetic_boot + real_boot
                random.seed(i)
                random.shuffle(bootstrap_dataset)
            else:
                bootstrap_dataset = resample(train_data, replace=True, n_samples=len(train_data), random_state=i)

            train, val = split_data(bootstrap_dataset, valid_fraction=0.2, seed=1234)
            with open(sample_file, "wb") as f:
                pickle.dump((train, val), f)

            # CSV for readability
            pd.DataFrame([{
                'atoms': d.atoms,
                'coordinates': d.pos.numpy().tolist(),
                'total_energy': d.y.numpy().tolist(),
                'forces': d.forces.numpy().tolist()
            } for d in train]).to_csv(f"{current_sample_dir}/train.csv", index=False)

            pd.DataFrame([{
                'atoms': d.atoms,
                'coordinates': d.pos.numpy().tolist(),
                'total_energy': d.y.numpy().tolist(),
                'forces': d.forces.numpy().tolist()
            } for d in val]).to_csv(f"{current_sample_dir}/val.csv", index=False)

        train_energy = [d.y for d in train]
        train_energy = torch.cat(train_energy, dim=0).cpu().tolist()
        val_energy = [d.y for d in val]
        val_energy = torch.cat(val_energy, dim=0).cpu().tolist()

        print(f"Training model {i}")
        args.checkpoints_dir = f"{result_of_sample_dir}/checkpoints"
        args.results_dir = f"{result_of_sample_dir}/results"
        args.log_dir = f"{result_of_sample_dir}/logs"
        args.model_dir = f"{result_of_sample_dir}"
        args.name = prefix

        run(args, train, val, latent=latent, charge_penalty=charge_penalty)

        plot_distribution(train_energy, result_of_sample_dir, 'train energy')
        plot_distribution(val_energy, result_of_sample_dir, 'val energy')


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrapped Training")
    parser.add_argument("--atom", type=str, required=True, help="Element symbol, e.g., Bi")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms in the system")
    parser.add_argument("--charge", type=str, required=True, help="Charge of the system (can include suffixes like '_samples')")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of bootstrapped samples")
    parser.add_argument("--config", type=str, default="config.yaml", help="Directory for config files")
    parser.add_argument("--results_dir", type=str, default="coulomb", help="Directory for results")
    parser.add_argument("--latent", type=str, default="False", help="Logging level")
    parser.add_argument("--penalty", type=int, default=1, help="charge penalty")


    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    prefix = f"{args.atom.lower()}{args.num_atom}{args.charge}"
    config_path = f"configs/{args.config}"
    res_dir = f"{args.results_dir}"
    latent = True if args.latent == "True" else False
    if latent:
        charge_penalty = args.penalty
    else:
        charge_penalty = None
    boot_train(prefix, args.num_samples, config_path, res_dir=res_dir, latent=latent, charge_penalty=charge_penalty)