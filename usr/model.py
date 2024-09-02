#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:53:03 2023

@author: chen
"""

class UserModel(object):
    """
    User defined model for both Prediction and Machine learning.
    Prediction:
        Receive inputs from Generator and make predictions.
        Receive model parameters from ML and update the model.
    Machine Learning:
        Receive inputs from Oracle and retrain the model.
        Output model parameters sent to Prediction models.
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
        self.rank = rank
        self.result_dir = result_dir
        self.mode = mode
        self.i_gpu = i_gpu
        
        ##### User Part #####
            
    ##########################################
    #          Prediction Part          #
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
        data_to_gene_list = None
        
        ##### User Part #####
        
        # data_to_gene_list should be a list containing 1-D numpy arrays
        return data_to_gene_list
    
    def update(self, weight_array):
        """
        Update model/scalar with new weights in weight_array.
        
        Args:
            weight_array (1-D numpy.ndarray): 1-D numpy array containing model/scalar weights.
                                              Source: weight_array from UserModel.get_weight().
        """
        ##### User Part #####
        pass
            
    def get_weight_size(self):
        """
        Return the size of model weight when unpacked as an 1-D numpy array.
        Used to send/receive weights through MPI.
        
        Returns:
            weight_size (int): size of model weight when unpacked as an 1-D numpy array.
        """
        weight_size = None
        
        ##### User Part #####
        
        # weight_size should be returned as an integer
        return weight_size

    ###########################################
    #          Machine Learning Part          #
    ###########################################         

    def get_weight(self):
        """
        Return model/scalar weights as an 1-D numpy array.
        
        Returns:
            weight_array (1-D numpy.ndarray): 1-D numpy array containing model/scalar weights.
                                              Destination: weight_array at UserModel.update().
        """
        weight_array = None
        
        ##### User Part #####
        
        # weight_array should be returned as an 1-D numpy array
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
        pass
        
    def retrain(self, req_data):
        """
        Retrain the model with current training set.
        Retraining should stop before or when receiving new data points.
        
        Args:
            req_data (MPI.Request): MPI request object indicating status of receiving new data points.

        Returns:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        stop_run = False
        ##### User Part #####
        
        # stop_run should be returned as a bool value.
        return stop_run
            
    def save_progress(self, stop_run):
        """
        Save the current progress/data/state.
        Called everytime after retraining and receiving new data points.
        """
        ##### User Part #####
        pass

    def stop_run(self):
        """
        Called before the Training/Prediction process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        pass
