#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:08:14 2022

@author: Chen Zhou
"""
import pickle
import os
import time
import numpy as np
from mpi4py import MPI

class FCNN(object):
    def __init__(self, keywords = None, id = None):
        self.metadata_dir = "/home/chen/Documents/BlueOLED/NNsForMD/blueOLED/pyrai2md/metadata"
        self.natom = 12
        self.nstate = 2
        self.world_comm    = kwargs.get('world_comm', None)
        self.md_group       = kwargs.get('md_group', None)
        
    def train(self):
        ## fake	function does nothing
        pass
    
    def load(self):
        ## fake	function does nothing
        pass
    
    def	appendix(self,addons):
       	## fake	function does nothing
        pass
        
    
    def evaluate(self, traj, md_comm, world_comm, md_comm_rank):
        # stop propagation when MD calculation failed and have NaN in coordinates
        if np.isnan(traj.coord).any():
            traj.status = 0
            return traj
        # gather coordiantes from trajectories of all MD processes
        md_rank = md_comm.Get_rank()
        md_size = md_comm.Get_size()
        coord_send = traj.coord.flatten()
        coord_receiv = np.empty((md_comm.Get_size(), self.natom, 3), dtype=float).flatten() if md_rank == md_comm_rank else None
        md_comm.Gather([coord_send, MPI.DOUBLE], coord_receiv, root=md_comm_rank)
        print(f"DEBUG: coordinates gathered to MD rank {md_rank}")
        
        # send coordinates to PL process (rank 0)
        if md_rank == md_comm_rank:
            world_comm.Send([coord_receiv, MPI.DOUBLE], dest=0, tag=1)
            print(f"DEBUG: coordinates send to rank 0 (PL process) from MD process rank {md_rank}")
            
            # receive energy and gradient predictions from PL process (rank 0)
            eng_status = MPI.Status()
            grad_status = MPI.Status()
            world_comm.Probe(source=MPI.ANY_SOURCE, tag=3, status=eng_status)
            eng_pred = np.empty(eng_status.Get_count(datatype=MPI.DOUBLE), dtype=float)
            world_comm.Recv([eng_pred, MPI.DOUBLE], source=eng_status.source, tag=3)
            print(f"DEBUG: shape of energy prediction: {eng_pred.shape}")
            print(f"DEBUG: last energy received: {eng_pred[-2:]}")
            world_comm.Probe(source=MPI.ANY_SOURCE, tag=4, status=grad_status)
            grad_pred = np.empty(grad_status.Get_count(datatype=MPI.DOUBLE), dtype=float)
            world_comm.Recv([grad_pred, MPI.DOUBLE], source=grad_status.source, tag=4)
            print(f"DEBUG: shape of force prediction: {grad_pred.shape}")
            print(f"DEBUG: last gradient received: {grad_pred[-12:]}")
        else:
            eng_pred = None
            grad_pred = None
            
        # distribute energy and gradient predictions to each MD process
        eng_recv = np.empty((self.nstate*self.nmodels,), dtype=float)
        md_comm.Scatter([eng_pred, MPI.DOUBLE], [eng_recv, MPI.DOUBLE], root=md_comm_rank)
        eng_recv = eng_recv.reshape(self.nmodels, self.nstate)
        print(f"DEBUG: MD rank {md_rank} receives energy {eng_recv}")
        grad_recv = np.empty((self.nmodels, self.nstate, self.natom, 3), dtype=float).flatten()
        md_comm.Scatter([grad_pred, MPI.DOUBLE], [grad_recv, MPI.DOUBLE], root=md_comm_rank)
        grad_recv = grad_recv.reshape(self.nmodels, self.nstate, self.natom, 3)
        print(f"DEBUG: MD rank {md_rank} receives gradient with shape {grad_recv.shape}")
        
        # TODO: check standard deviation
        if (np.std(eng_recv, axis=0)>self.std_threshold).any() or (np.std(grad_recv, axis=0)>self.std_threshold).any():
            traj.status = 0
            world_comm.Send()
            pass
        else:
            eng_recv = np.mean(eng_recv, axis=0)
            grad_recv = np.mean(grad_recv, axis=0)
        
        nac = []
        soc = []
        err_e = 0
       	err_g =	0
       	err_n =	0
       	err_s =	0
        
        traj.energy = np.copy(eng_recv)
        traj.grad = np.copy(grad_recv)
        traj.nac = np.copy(nac)
        traj.soc = np.copy(soc)
        traj.err_energy = err_e
        traj.err_grad = err_g
        traj.err_nac = err_n
        traj.err_soc = err_s
        traj.status = 1
        
        return traj
