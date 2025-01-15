#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:27:15 2023

@author: chen
"""

import gc

import numpy as np
import os, pickle


class UserGene(object):
    """
    User defined Generator. Receive prediction from Prediction kernel and generate new data points.
    """
    def __init__(self, rank, result_dir):
        """
        initilize the generator.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
        """
        self.rank = rank
        self.result_dir = result_dir
        ##### User Part ######
        self.counter = 0
        #self.limit = float("inf")
        self.limit = 300000 + self.rank
        self.state = np.random.randn(4)
        self.history = [[],]
        self.save_path = os.path.join(self.result_dir, f"generator_data_{rank}")
        
    def generate_new_data(self, data_to_gene):
        """
        Generate new data point based on data_to_gene (prediction from Prediction kernel).
        
        Args:
            data_to_gene (1-D numpy.ndarray or None): data from prediction kernel through EXCHANGE process.
                                                      Initialized as None for the first time step.
                                                      Source: element of data_to_gene_list from UserModel.predict()
            
        Returns:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
            data_to_pred (1-D numpy.ndarray): data to prediction kernel through EXCHANGE process.
                                              Destination: element of input_list at UserModel.predict()
        """
        stop_run = False
        data_to_pred = None
        
        # please notice that data_to_gene is intinilized to be None for the first iteration.
        ##### User Part #####
        # in this simple example, generator processes return random numbers
        if self.counter > self.limit:
            stop_run = True
            data_to_pred = np.random.randn(4)
            print(f"Generator rank{self.rank}: stop signal sent.")
        elif data_to_gene is None:
            data_to_pred = np.random.randn(4)
            self.history.append([data_to_pred,])
        elif (data_to_gene == 0).any():
            data_to_pred = np.random.randn(4)
            self.history.append([data_to_pred,])
        else:
            data_to_pred = self.state * data_to_gene
            self.history[-1].append(data_to_pred)  

        if self.counter % 10000 == 0:
            print(f"Generator rank{self.rank}: iteration {self.counter} finished.")    #TODO remove debug later
        
        self.counter += 1
        
        # stop_run should be returned as a bool value
        # data_to_pred should be returned as an 1-D numpy array
        return stop_run, data_to_pred
    
    def save_progress(self):
        """
        Save the current state and progress. Called everytime after the interval defined by progress_save_interval in al_setting.
        """
        ##### User Part #####
        m = 'ab' if os.path.exists(self.save_path) else 'wb'
        with open(self.save_path, m) as fh:
            pickle.dump(self.history[:-1], fh)
        self.history = self.history[-1:]

    def stop_run(self):
        """
        Called before the Generator process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        self.save_progress()
