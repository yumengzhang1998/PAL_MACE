#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:15:25 2023

@author: chen
"""

import time, os, pickle, sys, json, gc, threading
import numpy as np
from mpi4py import MPI

from PyRAI2MD.Dynamics.Propagators.surface_hopping import SurfaceHopping
from PyRAI2MD.Dynamics.Ensembles.ensemble import Ensemble
from PyRAI2MD.Dynamics.verlet import VerletI, VerletII
from PyRAI2MD.Dynamics.reset_velocity import ResetVelo
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong
from PyRAI2MD.Utils.coordinates import PrintCoord


class AIMD:
    """ Ab initial molecular dynamics class

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary
            qm               class       QM method class
            id               int         trajectory id index
            dir              boolean     create a subdirectory

        Attributes:          Type:
            version          str         version information header
            title            str         calculation title
            maxerr_energy    float       maximum energy error threshold
            maxerr_grad      float       maximum gradient error threshold
            maxerr_nac       float       maximum nac error threshold
            maxerr_soc       float       maximum soc error threshold
            silent           int         silient mode for screen output
            verbose          int         verbose level of output information
            direct           int         number of steps to record output
            buffer           int         number of steps to skip output
            record           int         record the history of trajectory
            checkpoint       int         trajectory checkpoint frequency
            restart          int         restart calculation
            addstep          int         number of steps that will be added in restarted calculation
            stop             int         trajectory termination signal
            skipstep         int         number of steps being skipped to write output
            skiptraj         int         number of steps being skipped to save trajectory

        Functions:           Returns:
            run              class       run molecular dynamics simulation

    """

    def __init__(self, trajectory = None, keywords = None, qm = None, id = None, dir = None, **kwargs):
        self.timing = 0  ## I use this to test calculation time

        ## initialize variables
        #self.version       = keywords['version']
        self.title         = keywords['control']['title']
        self.maxerr_energy = keywords['control']['maxenergy']
        self.maxerr_grad   = keywords['control']['maxgrad']
        self.maxerr_nac    = keywords['control']['maxnac']
        self.maxerr_soc    = keywords['control']['maxsoc']
        self.silent        = keywords['md']['silent']
        self.verbose       = keywords['md']['verbose']
        self.direct        = keywords['md']['direct']
        self.buffer        = keywords['md']['buffer']
        self.record        = keywords['md']['record']
        self.checkpoint    = keywords['md']['checkpoint']
        self.restart       = keywords['md']['restart']
        self.addstep       = keywords['md']['addstep']
        self.stop          = 0
        self.skipstep      = 0
        self.skiptraj      = 0
        self.bond_index    = kwargs.get('bond_index', None)
        self.bond_limit    = kwargs.get('bond_limit', None)

        
        ## create a trajectory object
        self.traj = trajectory

        ## create an electronic method object
        self.QM = qm
        
        ## loop over molecular dynamics steps
        self.traj.step += self.addstep
        self.step_left = self.traj.step - self.traj.iter
        
    def propagate_step_one(self, **kwargs):
        ##
	##-----------------------------------------
        ## ---- Initial Kinetic Energy Scaling
        ##-----------------------------------------
        ##
        if self.traj.iter == 1:
            f = 1
            ## add excess kinetic energy in the first step if requested
            if self.traj.excess != 0:
                K0 = self._kinetic_energy(self.traj)
                f = ((K0 + self.traj.excess) / K0)**0.5

            ## scale kinetic energy in the first step if requested
            if self.traj.scale != 1:
                f = self.traj.scale**0.5

            ## scale kinetic energy to target value in the first step if requested
            if self.traj.target != 0:
                K0 = self._kinetic_energy(self.traj)
                f = (self.traj.target / K0)**0.5

            self.traj.velo *= f

	##-----------------------------------------
        ## ---- Trajectory Propagation
	##-----------------------------------------
        ##
        ## update previous-preivous and previous nuclear properties              
        self.traj.update_nu()

        ## update current kinetic energies, coordinates, and gradient
        self.traj = VerletI(self.traj)
        
        return self.traj.state - 1, np.copy(self.traj.coord)
    
    def post_propagate(self, eng_pred, force_pred):
        ## update trajectory energy and force with predictions
        nac = []
        soc = []
        err_e = 0
        err_g =	0
        err_n =	0
        err_s =	0
        
        self.traj.energy = np.copy(eng_pred)
        self.traj.grad = np.copy(force_pred)
        self.traj.nac = np.copy(nac)
        self.traj.soc = np.copy(soc)
        self.traj.err_energy = err_e
        self.traj.err_grad = err_g
        self.traj.err_nac = err_n
        self.traj.err_soc = err_s
        self.traj.status = 1

        self.traj = VerletII(self.traj)

        self.traj = self._kinetic_energy(self.traj)

        ##
	##-----------------------------------------
        ## ---- Velocity Adjustment
	##-----------------------------------------
        ##
        ## reset velocity to avoid flying ice cube
        ## end function early if velocity reset is not requested
        if self.traj.reset != 1:
            return None

       	## end function early if	velocity reset step is 0 but iteration is more than 1
        if self.traj.resetstep == 0 and self.traj.iter > 1:
            return None

        ## end function early if velocity reset step is not 0 but iteration is not the multiple of it 
        if self.traj.resetstep != 0:
            if self.traj.iter % self.traj.resetstep != 0:
                return None

        ## finally reset velocity here
        self.traj = ResetVelo(self.traj)
        
        # normal return
        return 0
        
    def _potential_energies(self, traj):
        if self.comm_md_mg is None:
            traj = self.QM.evaluate(traj)
        else:
            ## get energy/force through MPI interface
            # communicate through MPI to get energy/force predictions
            coords_send = np.concatenate((np.array([float(traj.state-1),]), traj.coord.flatten()), axis=0)
            coords_collect = None
            # send coordinates to PL process through MG process
            self.t1 = time.time()
            self.comm_md_mg.Gather([coords_send, MPI.DOUBLE], coords_collect, root=0)
            self.t2 = time.time()
            # receive energy and force predictions through MG process
            data_to_md = None
            data_recv = np.empty((traj.nstate+traj.nstate*traj.natom*3+1,), dtype=float)
            self.comm_md_mg.Scatter([data_to_md, MPI.DOUBLE], [data_recv, MPI.DOUBLE], root=0)
            self.t3 = time.time()
            save_data = int(data_recv[-1])
            data_recv = data_recv[:-1]
            eng_recv = data_recv[:traj.nstate].reshape(traj.nstate,) / 27.21138624598853  # eV to Hatree
            grad_recv = data_recv[traj.nstate:].reshape(traj.nstate, traj.natom, 3) / 27.21138624598853 * 0.52917721090380  # eV to Hatree
            
            if save_data == 1:
                self.traj_data['save_data'] = True
            #else:
            #    self.traj_data['save_data'] = False
            
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
            
            # check if PL predictions fail the std test
            if eng_recv[0] == 0.0:
                self.mpi_status = 1
            else:
                self.traj_data['energy'][-1].append(eng_recv.tolist())
                self.traj_data['force'][-1].append(grad_recv[traj.state-1].tolist())
                self.traj_data['coord'][-1].append(traj.coord.tolist())
                self.traj_data['state'][-1].append(int(traj.state-1))
                del eng_recv, grad_recv
                gc.collect()

        return traj
    
    def _kinetic_energy(self, traj):
        traj.kinetic = np.sum(0.5 * (traj.mass * traj.velo**2))

        return traj

    def _thermodynamic(self):
        self.traj = Ensemble(self.traj)

        return self.traj

    def _surfacehop(self):
        ## update previous population, energy matrix, and non-adiabatic coupling matrix
        self.traj.update_el()

        # update current population, energy matrix, and non-adiabatic coupling matrix
        self.traj = SurfaceHopping(self.traj)

        return self.traj

    def _reactor(self):
        ## TODO periodically adjust velocity to push reactants toward center

        return self

    def _heading(self):
        state_info = ''.join(['%4d' % (x+1) for x in range(len(self.traj.statemult))])
        mult_info = ''.join(['%4d' % (x) for x in self.traj.statemult])

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*


 State order:      %s
 Multiplicity:     %s

 QMMM key:         %s
 QMMM xyz          %s
 Active atoms:     %s
 Inactive atoms:   %s
 Link atoms:       %s
 Highlevel atoms:  %s
 Lowlevel atoms:   %s

""" % ( self.version,
        state_info,
        mult_info,
        self.traj.qmmm_key,
        self.traj.qmmm_xyz,
        self.traj.natom,
        self.traj.ninac,
        self.traj.nlink,
        self.traj.nhigh,
        self.traj.nlow)

        return headline

    def _chkerror(self):
        ## This function check the errors in energy, force, and NAC
        ## This function stop MD if the errors exceed the threshold
        ## This function stop MD if the qm calculation failed

        if self.traj.err_energy != None and\
            self.traj.err_grad != None and\
            self.traj.err_nac != None and\
            self.traj.err_soc != None:
            if  self.traj.err_energy > self.maxerr_energy or\
                self.traj.err_grad > self.maxerr_grad or\
                self.traj.err_nac > self.maxerr_nac or\
                self.traj.err_soc > self.maxerr_soc:
                self.stop = 1
        elif self.traj.status == 0:
            self.stop = 2
            
    def _step_counter(self, counter, step):
        counter += 1

        ## reset counter to 0 at printing step or end point or stopped point
        if counter == step or counter == self.traj.step or self.stop == 1:
            counter = 0

        return counter

    def _force_output(self, counter):
        if self.traj.iter == self.traj.step or self.stop != 0:
            counter = 0
        return counter
    
    def _calc_bond_length(self, coord):
        c1 = np.expand_dims(coord, axis=0)
        c2 = np.expand_dims(coord, axis=1)
        dist_matrix = np.sqrt(np.sum(np.square(c1-c2), axis=-1))
        bond_length = {}
        for b, idx in self.bond_index.items():
            bond_length[b] = [dist_matrix[i,j] for (i,j) in idx]
        return bond_length
    
    def _check_bond_length(self, traj):
        bond_lengths = self._calc_bond_length(traj.coord)
        for b, l in self.bond_limit.items():
            if (np.array(bond_lengths[b], dtype=float) > l).any():
                return False, b
        return True, None
    
    def propagate_step_two(self, eng_pred, force_pred):
        ## propagate nuclear positions (E,G,N,R,V,Ekin) with predicted energy and force
        self.post_propagate(eng_pred, force_pred)
        
        ## adjust kinetics (Ekin,V,thermostat)
        self._thermodynamic()

        ## detect surface hopping
        self._surfacehop()   # update A,H,D,V,state

        ## check errors and checkpointing
        self._chkerror()
        
        ## check if bond length has exceeded limit
        normal_bond_length, b = self._check_bond_length(self.traj)
        
        ## terminate trajectory
        if not normal_bond_length:
            warning = f'Trajectory terminated beacuse bond {b} has exceeded bond length limit.'
            return 1
        if   self.stop == 1:
            warning = 'Trajectory terminated because the NN prediction differences are larger than thresholds.'
            return 2
        elif self.stop == 2:
            warning = 'Trajectory terminated because the QM calculation failed.'
            return 3
        ## this trajectory step ends normally
        return 0
        
    def run(self):
        ## loop over molecular dynamics steps
        self.traj.step += self.addstep
        step_left = self.traj.step - self.traj.iter
        #for iter in range(self.traj.step - self.traj.iter):
        while self.traj.iter < step_left:
            self.t0 = time.time()
            self.traj.iter += 1

            ## propagate nuclear positions (E,G,N,R,V,Ekin)
            self._propagate()

            if self.traj_data['save_data'] and (self.traj_data['md_traj_req'] is None or self.traj_data['md_traj_req'].Test()) and (self.traj_data['md_time_req'] is None or self.traj_data['md_time_req'].Test()):
                traj_save = np.array([], dtype=float)
                for k in ['coord', 'energy', 'force', 'state']:
                    traj_save = np.append(traj_save, np.array(self.traj_data[k][:-1], dtype=float).flatten(), axis=0)
                    del self.traj_data[k][:-1]
                    gc.collect()
                    traj_save = np.append(traj_save, [-1,], axis=0)
                    self.traj_data[k] = [self.traj_data[k][-1],]

                self.traj_data['md_traj_req'] = MPI.Grequest.Start(query_fn, free_fn, cancel_fn)
                #traj_thread = threading.Thread(target=save_data, name=f'save_traj_{self.rank}', args=(self.traj_data['md_traj_req'],), kwargs={'fname': self.fname_traj, 'comm': self.comm_md, 'amode': self.amode_traj, 'data': np.copy(traj_save),}, daemon=True)
                traj_thread = threading.Thread(target=save_by_numpy, name=f'save_traj_{self.rank}', args=(self.traj_data['md_traj_req'],), kwargs={'fname': self.fname_traj, 'rank': self.rank, 'data': np.copy(traj_save),}, daemon=True)
                traj_thread.start()
                del traj_save
                gc.collect()
                self.traj_data['save_data'] = False

            ## check if PL predictions passed std test (for mpi interface only)
            if self.mpi_status == 1:
                self.t4 = time.time()
                if self.time_md != None:
                    self.time_md['gather'].append(self.t2 - self.t1)
                    self.time_md['scatter'].append(self.t3 - self.t2)
                    self.time_md['prop'].append(self.t4 -self.t0 - self.t3 + self.t1)
                warning = 'Trajectory terminated beacuse std of PL predictions is too high.'
                break

            ## adjust kinetics (Ekin,V,thermostat)
            self._thermodynamic()

            ## detect surface hopping
            self._surfacehop()   # update A,H,D,V,state

            ## check errors and checkpointing
            self._chkerror()
            
            ## check if bond length has exceeded limit
            normal_bond_length, b = self._check_bond_length(self.traj)

            ## terminate trajectory
            if not normal_bond_length:
                warning = f'Trajectory terminated beacuse bond {b} has exceeded bond length limit.'
                break
            if   self.stop == 1:
                warning = 'Trajectory terminated because the NN prediction differences are larger than thresholds.'
                break
            elif self.stop == 2:
                warning = 'Trajectory terminated because the QM calculation failed.'
                break

        if len(self.traj_data['state'][-1]) == 0:
            for k, v in self.traj_data.items():
                if type(v) is list:
                    del self.traj_data[k][-1]
        elif len(self.traj_data['state'][-1]) < self.traj.step:
            tmp_eng = np.array(self.traj_data['energy'][-1], dtype=float)
            del self.traj_data['energy'][-1]
            gc.collect()
            tmp = np.zeros((self.traj.step, self.traj.nstate), dtype=float)
            tmp[:tmp_eng.shape[0]] = tmp_eng
            self.traj_data['energy'].append(tmp.tolist())
            del tmp, tmp_eng
            gc.collect()
            
            tmp_force = np.array(self.traj_data['force'][-1], dtype=float)
            del self.traj_data['force'][-1]
            gc.collect()
            tmp = np.zeros((self.traj.step, self.traj.natom, 3), dtype=float)
            tmp[:tmp_force.shape[0]] = tmp_force
            self.traj_data['force'].append(tmp.tolist())
            del tmp, tmp_force
            gc.collect()

            tmp_coord = np.array(self.traj_data['coord'][-1], dtype=float)
            del self.traj_data['coord'][-1]
            gc.collect()
            tmp = np.zeros((self.traj.step, self.traj.natom, 3), dtype=float)
            tmp[:tmp_coord.shape[0]] = tmp_coord
            self.traj_data['coord'].append(tmp.tolist())
            del tmp, tmp_coord
            gc.collect()

            print(f"Number of steps: {self.traj.step}")
            print(f"Number of states: {len(self.traj_data['state'][-1])}")
            tmp_state = np.zeros((self.traj.step,), dtype=int)
            tmp_state[:len(self.traj_data['state'][-1])] = self.traj_data['state'][-1]
            del self.traj_data['state'][-1]
            self.traj_data['state'].append(tmp_state)
            del tmp_state
            gc.collect()

        return self.traj
