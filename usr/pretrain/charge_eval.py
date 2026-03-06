###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
from typing import List, Tuple, Union
import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils

def init_device(device):
    device_str = str(device)  # ✅ Ensure device is a string

    if "cuda" in device_str:
        return torch.device("cuda")
    elif "cpu" in device_str:
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device type: {device_str}")


def evaluate(   model, 
                eval_dataset,
                batch_size,
                default_dtype = "float64",
                device = "cpu",                 
                compute_stress = False,
                return_contributions = False
                 ) -> Tuple[np.ndarray, List[np.ndarray], Union[np.ndarray, None], Union[List[np.ndarray], None]]:
    torch_tools.set_default_dtype(default_dtype)
    device = init_device(device)

    # Load model
    # model = torch.load(model, map_location = device)
    model = model.to(device)  # shouldn't be necessary but seems to help with CUDA problems
    model.eval() 

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_data(
                eval_data, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for eval_data in eval_dataset
        ],
        batch_size = batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    stresses_list = []
    forces_collection = []
    charge_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its empty
        if "charges" in output or "latent_charges" in output:
            charge_key = "charges" if "charges" in output else "latent_charges"
            charges = torch_tools.to_numpy(output[charge_key])
            charges_split = np.split(
                charges, indices_or_sections=batch.ptr[1:], axis=0
            )
            charge_collection.append(charges_split[:-1]) 

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    charge_list = [
        charges for charge_list in charge_collection for charges in charge_list
    ]
    assert len(eval_dataset) == len(energies) == len(forces_list)
    if compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(eval_dataset) == stresses.shape[0]

    return energies, forces_list, stresses_list, contributions_list, charge_list
