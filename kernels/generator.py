#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:27:15 2023

@author: chen
"""

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
        
        # stop_run should be returned as a bool value
        # data_to_pred should be returned as an 1-D numpy array
        return stop_run, data_to_pred
    
    def save_progress(self, stop_run):
        """
        Save the current state and progress. Called everytime after the interval defined by progress_save_interval in al_setting, and when the active learning workflow is shutdown (stop_run is True).

        Args:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        ##### User Part #####
        pass

    def stop_run(self):
        """
        Called before the Generator process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        pass
