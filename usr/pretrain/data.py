import os
import numpy as np
import torch
import ase
import ase.io
from ase.data import atomic_numbers
import pandas as pd
from ast import literal_eval
from torch_geometric.data import Data
from torch_geometric.data import  Dataset
from torch_geometric.data.lightning import LightningDataset

from functions.config import ConfigLoader
import functions.properties as ppt


class big_list:
    def __init__(self, raw_data_path, num_atom = 4, charge = -2, transform=None, pre_transform=None, pre_filter=None, gradeint_to_force = False) -> None:
        print("Reading data from: ", raw_data_path)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.num_atom = num_atom
        self.charge = charge
        self.reverse = gradeint_to_force
        if type(raw_data_path) == list:
            self.raw_paths = raw_data_path
        else:
            self.raw_paths = [raw_data_path]
        self.data_list = self.process()
        if self.transform is not None:
            self.data_list = [self.transform(data) for data in self.data_list]

    def process(self):
        # Read data into huge `Data` list.
        print(self.raw_paths)
        data_list = []
        coords = []
        elements = []
        energies_0 = []
        forces_0 = []

        for raw_path in self.raw_paths:
            print("Reading data from: ", raw_path)
            if not os.path.exists(raw_path):
                print("WARNING: Can not find data for path: '%s'." % raw_path)
            if raw_path.endswith(".csv"):
                data = pd.read_csv(raw_path)
                elements = data["atoms"].values
                elements = [literal_eval(e) for e in elements]
                elements_number = [[atomic_numbers[ei] for ei in e] for e in elements]
                coords = data["coordinates"].values
                coords = [np.array(np.matrix(c.replace('\n', ';'))).reshape((self.num_atom, 3)) for c in coords]
                energies_0 = data['total_energy'].values
                if type(energies_0[0]) == str:
                    energies_0 = [literal_eval(e) for e in energies_0]
                else:
                    energies_0 = [e for e in energies_0]
                print(energies_0[0])
                # convert = lambda a: np.array(np.matrix(a.replace('\n', ';'))) if type(a) == str else a
                forces_0 = data['forces'].values

                forces_0 = [np.array(np.matrix(c.replace('\n', ';'))).reshape((self.num_atom, 3)) for c in forces_0]
                if self.reverse:
                    forces_0 = [-1 * f for f in forces_0]
                print(forces_0[0].shape)



        for i in range(len(coords)):
            if not np.isnan(energies_0[i]):
                data = Data(
                    pos=torch.tensor(coords[i]),
                    z=torch.tensor(elements_number[i]),
                    y=torch.tensor(energies_0[i]),
                    forces=torch.tensor(forces_0[i]),
                    charge=torch.tensor(self.charge, dtype=torch.int32),
                    atoms = elements[i],
                )
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        return data_list
    
# split train and validation data
import torch
from torch.utils.data import random_split

def split_data(data_list, valid_fraction=0.1, seed=1234):
    """Splits data_list into training and validation sets randomly."""
    torch.manual_seed(seed)  # Set seed for reproducibility

    total_size = len(data_list)
    valid_size = int(total_size * valid_fraction)
    train_size = total_size - valid_size

    train_dataset, valid_dataset = random_split(data_list, [train_size, valid_size])
    
    return list(train_dataset), list(valid_dataset)






class full_data_list:
    def __init__(self, raw_data_path, gradeint_to_force = False) -> None:
        print("Reading data from: ", raw_data_path)


        self.reverse = gradeint_to_force
        if type(raw_data_path) == list:
            self.raw_paths = raw_data_path
        else:
            self.raw_paths = [raw_data_path]
        self.data_list = self.process()

    def process(self):
        # Read data into huge `Data` list.
        print(self.raw_paths)
        data_list = []
        coords = []
        elements = []
        energies_0 = []
        forces_0 = []
        charge = []

        for raw_path in self.raw_paths:
            print("Reading data from: ", raw_path)
            if not os.path.exists(raw_path):
                print("WARNING: Can not find data for path: '%s'." % raw_path)
            if raw_path.endswith(".csv"):
                data = pd.read_csv(raw_path)
                elements = data["atoms"].values
                elements = [literal_eval(e) for e in elements]
                elements_number = [[atomic_numbers[ei] for ei in e] for e in elements]
                coords = data["coordinates"].values
                coords = [np.array(np.matrix(c.replace('\n', ';'))).reshape((-1, 3)) for c in coords]
                energies_0 = data['total_energy'].values
                
                if type(energies_0[0]) == str:
                    energies_0 = [literal_eval(e) for e in energies_0]
                else:
                    energies_0 = [e for e in energies_0]
                print(energies_0[0])
                # convert = lambda a: np.array(np.matrix(a.replace('\n', ';'))) if type(a) == str else a
                forces_0 = data['forces'].values

                forces_0 = [np.array(np.matrix(c.replace('\n', ';'))).reshape((-1, 3)) for c in forces_0]
                charge = data['charge'].values
                charge = [int(c) for c in charge]
                if self.reverse:
                    forces_0 = [-1 * f for f in forces_0]
                print(forces_0[0].shape)



        for i in range(len(coords)):
            if not np.isnan(energies_0[i]):
                data = Data(
                    pos=torch.tensor(coords[i]),
                    z=torch.tensor(elements_number[i]),
                    y=torch.tensor(energies_0[i]),
                    forces=torch.tensor(forces_0[i]),
                    charge=torch.tensor(charge[i], dtype=torch.int32),
                    atoms = elements[i],
                )
                data_list.append(data)



        return data_list