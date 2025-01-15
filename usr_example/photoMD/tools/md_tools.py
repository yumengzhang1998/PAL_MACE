from PyRAI2MD.Utils.sampling import Element
from PyRAI2MD.Utils.coordinates import ReadInitcond
from PyRAI2MD.Dynamics.aimd import AIMD
from PyRAI2MD.Molecule.trajectory import Trajectory

import numpy as np
import pickle, gc
def read_inicond(init_coords, init_velc, elements):
    """
    Parameters
    ----------
    init_coords : numpy.ndarray
        Initial coordinates with shape (n_ensembles, natom, 3).
    init_velc : numpy.ndarray
        Initial velocities with shape (n_ensembles, natom, 3).
    elements : list
        List of elements.

    Returns
    -------
    ensemble : list of numpy.ndarray
        Initial conditions.

    """
    amass = []
    achrg = []
    natom = len(elements)
    for a in elements:
        amass.append(Element(a).getMass())
        achrg.append(Element(a).getNuc())
    
    atoms = np.array(elements)
    amass = np.array(amass)
    amass = amass.reshape((natom, 1))
    achrg = np.array(achrg)
    achrg = achrg.reshape((natom, 1))
    
    ensemble = [] # a list of sampled  molecules, ready to transfer to external module or print out
    
    for s in range(0, init_coords.shape[0]):
        inicond = np.concatenate((init_coords[s], init_velc[s]), axis=1)
        inicond = np.concatenate((atoms.reshape(-1, 1), inicond), axis=1)
        inicond = np.concatenate((inicond, amass[:, 0: 1]), axis = 1)
        inicond = np.concatenate((inicond, achrg), axis = 1)
        ensemble.append(inicond)
        del inicond
    return ensemble

def set_aimd(mol, job_keywords, bond_index, bond_limit):
    """
    Parameters
    ----------
    mol : list of numpy.ndarray
        Initial conditions.
    job_keywords : dict
        Dictionary containing job keywords.
    bond_index: Dict
        Dictionary containing atom index of chemical bonds.
    bond_limit: Dict
        Dictionary containing limitation of bond length.

    Returns
    -------
    aimd : AIMD object
        Object managing MD simulation.

    """
    atoms, xyz, velo = ReadInitcond(mol)
    traj = Trajectory(mol, keywords = job_keywords)
    method = None
    aimd = AIMD(trajectory = traj,
                keywords = job_keywords,
                qm = method,
                id = None,
                dir = None,
                bond_index=bond_index,
                bond_limit=bond_limit,
                )
    return aimd

def save_pickle(traj_data: dict, step_keys: list, traj_keys: list, save_path: list, mode: str, coord = None, state: int = None, velo = None, grad = None, iteration: int = None):
    """
    Save data as pickle object to save_path. Use threading to save time.

    Parameters
    ----------
    traj_data : dict
        dictionary with trajectory data. Data saved in the dictionary is moved to data_save for saving.
    step_keys : list
        list of keys of traj_data to data related to each time step (e.g. coordinates, energies, forces).
    traj_keys : list
        list of keys of traj_data to data related to trajectories (e.g. termination reason).
    save_path : list
        list of paths to the saved data. ([current information data path, trajectroies data path])
    mode : str
        wb for creating the file or overwrite the existing file. ab for appending on the existing file.
    coord : numpy.ndarray
        coordinate of the current step (shape: (number of atoms, 3)) for restarting the trajectory.
    state : int
        state of the current step for restarting the trajectory.
    velo : numpy.ndarray
        velocity of the current step (shape: (number of atoms, 3)) for restarting the trajectory.
    grad: numpy.ndarray
        gradient of the current step (shape: (number of states, number of atoms, 3)) for restarting the trajectory.
    iteration : int
        current iteration (time step) for restarting the trajectory.

    Returns
    -------
    None.

    """
    #if type(traj_data) == dict:
    #    traj_data['iteration'] = iteration
    #    traj_data['velo'] = velo
    if not iteration is None and not velo is None:
        with open(save_path[0], "wb") as fh:
            pickle.dump([coord, state, velo, grad, iteration], fh)
    with open(save_path[1], mode) as fh:
        pickle.dump(traj_data, fh)
    del traj_data, coord, state, velo, grad, iteration
    gc.collect()