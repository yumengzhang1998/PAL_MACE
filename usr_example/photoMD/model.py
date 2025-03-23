#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:53:03 2023

@author: chen
"""

from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import MaskedEnergyGradientStandardScaler
from pyNNsMD.utils.loss import get_lr_metric, ZeroEmptyLoss, MaskedScaledMeanAbsoluteError, masked_r2_metric, mask_MeanSquaredError

import tensorflow as tf
import tensorflow.keras as ks
from mpi4py import MPI
import numpy as np
import os, gc, pickle, json, sys

from photoMD.settings import model_setting

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
    def __init__(self, rank, result_dir, i_device, mode):
        """
        Initilize the model.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
            i_device (int): Index for assigning computation to device (GPU or CPU).
            mode (str): 'predict' for Passive Learner and 'train' for Machine Learning.
        """
        self.rank = rank
        self.result_dir = result_dir
        self.mode = mode
        self.i_device = i_device
        
        ##### User Part #####
        # seperate output logs of prediction and training kernels
        output_log = os.path.join(self.result_dir, f"log_{self.mode}_{self.rank}.output")
        mode_log = 'a' if os.path.exists(output_log) else 'w'
        self.fstdout = open(output_log, mode_log)
        sys.stdout = self.fstdout

        # set up GPU
        set_gpu([self.i_device,])

        self.hyper = model_setting.ml_hyper    # model and retraining hyperparameters
        self.model_dir = os.path.join(self.result_dir, model_setting.model_path)    # path to the directory (in the result directory) with model files
        self.mode = mode    # flag of prediction or training kernel
        self.natoms = self.hyper['model']['atoms']
        self.nstates = self.hyper['model']['states']

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        if self.mode == 'train':
            weight_path = os.path.join(self.model_dir, f'weights_v{self.i_device}')
            if os.path.exists(weight_path):
                # weights saved as pickle file
                with open(weight_path, 'rb') as fh:
                    self.model_weight = pickle.load(fh)
            else:
                # weights saved in h5 format
                with tf.device('/CPU:0'):
                    model = EnergyGradientModel(**self.hyper['model'])    # create models
                    if os.path.exists(os.path.join(self.model_dir, f'weights_v{self.i_device}.h5')):
                        model.load_weights(os.path.join(self.model_dir, f'weights_v{self.i_device}.h5'))
                    self.model_weight = model.get_weights()
                    del model
                    gc.collect()
            self._scaler = MaskedEnergyGradientStandardScaler()    # create scalar
            if os.path.exists(os.path.join(self.model_dir, f"scaler_v{self.i_device}.json")):
                self._scaler.load(os.path.join(self.model_dir, f"scaler_v{self.i_device}.json"))
            if os.path.exists(os.path.join(self.model_dir, f'new_training_set{self.i_device}.npy')):
                # load saved training set
                with open(os.path.join(self.model_dir, f'new_training_set{self.i_device}.npy'), 'rb') as fh:
                    self.coord = np.load(fh)
                    self.energy = np.load(fh)
                    self.force = np.load(fh)
                    self.i_train = np.load(fh)
                    self.i_val = np.load(fh)
            elif os.path.exists(os.path.join(self.model_dir, 'data_x')):
                # load initial training set for each model
                self.training_set_path = [os.path.join(self.model_dir, 'data_x'), os.path.join(self.model_dir, 'data_y')]
                with open(self.training_set_path[0], 'rb') as fh:
                    self.coord = pickle.load(fh)
                with open(self.training_set_path[1], 'rb') as fh:
                    self.energy, self.force = pickle.load(fh)
                # x and y data for all models are stored together to save space
                # need to load index and rebuild data for each model
                self.i_train = np.load(os.path.join(self.model_dir, 'index', f'train_val_idx_v{self.i_device}.npy'))
                self.i_val = np.array([i for i in range(0, self.coord.shape[0]) if i not in self.i_train], dtype=int)
            else:
                self.coord, self.energy, self.force = None, None, None
                self.i_train, self.i_val = [], []

            self.hist_path = os.path.join(self.model_dir, f'retrain_history{self.i_device}.json')
            try:
                with open(self.hist_path, 'r') as fh:
                    self.hist = json.load(fh)
            except:
                self.hist = {
                    'energy_mean_absolute_error': [],
                    'energy_r2': [],
                    'val_energy_mean_absolute_error': [],
                    'val_energy_r2': [],
                    'force_mean_absolute_error': [],
                    'force_r2': [],
                    'val_force_mean_absolute_error': [],
                    'val_force_r2': [],
                    'num_epoch': []
                    }
        else:
            #set_gpu([gpu_index,])
            self._model = EnergyGradientModel(**self.hyper['model'])    # create models
            self._scaler = MaskedEnergyGradientStandardScaler()    # create scalar
            self._model_setup()    # load weights for model and scaler
            
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
        if self.mode == "predict":
            # unpack the input data generated by the Generater kernel
            x = np.array(list_data_to_pred, dtype=float)
            occupied_states = x[:,0]
            x = x[:,1:].reshape(x.shape[0], self.natoms, 3)
            batch_size = self.hyper['general']['batch_size_predict']
            x_scaled = self._scaler.transform(x=x)[0]
            res = self._model.predict(x_scaled, batch_size=batch_size)

        else:
            x = np.array(list_data_to_pred, dtype=float)
            occupied_states = x[:,0]
            x = x[:,1:].reshape(x.shape[0], self.natoms, 3)
            batch_size = self.hyper['general']['batch_size_predict']
            x_scaled = self._scaler.transform(x=x)[0]
            tf.keras.backend.clear_session()
            model = EnergyGradientModel(**self.hyper['model'])    # create models
            model.precomputed_features = False
            model.output_as_dict = False
            model.energy_only = False
            model.set_weights(self.model_weight)
            res = model.predict(x_scaled, batch_size=batch_size)
            del model, x, x_scaled
            gc.collect()

        with tf.device('/CPU:0'):
            eng_pred, force_pred = self._scaler.inverse_transform(y=res)[1]

        data_to_gene_list = [np.append(eng_pred[i, :self.nstates].flatten(), force_pred[i, :self.nstates].flatten(), axis=0) for i in range(0, eng_pred.shape[0])]

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
        del self._model
        gc.collect()
        tf.keras.backend.clear_session()
        self._model = EnergyGradientModel(**self.hyper['model'])

        weight_list = []
        for s in self._weight_shape:
            n_weights = s[0] if len(s) == 1 else s[0]*s[1]
            weight_list.append(weight_array[:n_weights].reshape(s))
            weight_array = weight_array[n_weights:]
        self._model.set_weights(weight_list)
        del weight_list
        gc.collect()

        self._model.precomputed_features = False
        self._model.output_as_dict = False
        self._model.energy_only = self.hyper['retraining']['energy_only']

        # unpack the scalar weights
        n_state = self.hyper['model']['states']
        self._scaler.x_mean = np.array([float(weight_array[0])])
        self._scaler.x_std = np.array([float(weight_array[1])])
        self._scaler.energy_mean = weight_array[2:2+n_state].reshape(1, n_state)
        self._scaler.energy_std = weight_array[2+n_state:2+2*n_state].reshape(1, n_state)
        self._scaler.gradient_std = weight_array[2+2*n_state:].reshape(1, n_state, 1, 1)

        del weight_array
        gc.collect()
            
    def get_weight_size(self):
        """
        Return the size of model weight when unpacked as an 1-D numpy array.
        Used to send/receive weights through MPI.
        
        Returns:
            weight_size (int): size of model weight when unpacked as an 1-D numpy array.
        """
        weight_size = None
        
        ##### User Part #####
        model_weight = self._model.get_weights()
        # count size of weights of model
        weight_size = 0
        for w in model_weight:
            s = w.shape
            weight_size += s[0] if len(s)==1 else s[0]*s[1]
        # add the number of weight of scalar
        weight_size += 2    # energy mean and energy std
        weight_size += self._scaler.energy_mean.flatten().shape[0]
        weight_size += self._scaler.energy_std.flatten().shape[0]
        weight_size += self._scaler.gradient_std.flatten().shape[0]
        del model_weight
        gc.collect()

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
        # format of weight array that is sent to Prediction process: [number of layers, number of weights of layer 0, number of weights of layer 1, ..., weights of layer 0, weights of layer 1, ..., scalar weights...]
        weight_array = np.array([], dtype=float)
        for i in range(0, len(self.model_weight)):
            weight_array = np.concatenate((weight_array, self.model_weight[i].flatten()), axis=0)
        # add scalar parameters to the weight array
        weight_array = np.concatenate((weight_array, self._scaler.x_mean.flatten(), self._scaler.x_std.flatten(), \
                                       self._scaler.energy_mean.flatten(), self._scaler.energy_std.flatten(), self._scaler.gradient_std.flatten()), axis=0)
        
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
        # organize input data
        new_coord = np.empty((len(datapoints), self.natoms, 3), dtype=float)
        new_energy = np.empty((len(datapoints), self.nstates), dtype=float)
        new_force = np.zeros((len(datapoints), self.nstates, self.natoms, 3), dtype=float)
        new_size = int((1-self.hyper['retraining']['val_split'])*new_coord.shape[0])
        for i in range(0, len(datapoints)):
            current_state = int(datapoints[i][0][0])
            new_coord[i] = datapoints[i][0][1:].reshape(self.natoms, 3)
            new_energy[i] = datapoints[i][1][:self.nstates].reshape(self.nstates,)
            new_force[i][current_state] = datapoints[i][1][self.nstates:].reshape(self.natoms, 3)

        if self.coord is None:
            self.coord = new_coord.copy()
            self.energy = new_energy.copy()
            self.force = new_force.copy()
            self.i_train = np.random.choice(np.arange(self.coord.shape[0]), size=new_size, replace=False)
            self.i_val = np.array([i for i in np.arange(self.coord.shape[0]) if i not in self.i_train], dtype=int)
        else:
            idx = np.array(range(self.coord.shape[0], self.coord.shape[0]+new_coord.shape[0]), dtype=int)
            np.random.shuffle(idx)
            new_train = np.random.choice(idx, size=new_size, replace=False)
            new_val = np.array([i for i in idx if i not in new_train], dtype=int)
            self.i_train = np.concatenate((self.i_train, new_train), axis=0)
            self.i_val = np.concatenate((self.i_val, new_val), axis=0)
            self.coord = np.concatenate((self.coord, new_coord), axis=0)
            self.energy = np.concatenate((self.energy, new_energy), axis=0)
            self.force = np.concatenate((self.force, new_force), axis=0)
        assert self.coord.shape[0] == self.energy.shape[0] and self.coord.shape[0] == self.force.shape[0], "Check training increment at _add_trainingset"
        del new_coord, new_energy, new_force, datapoints
        gc.collect() 
        
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
        tf.keras.backend.clear_session()

        fit_hyper = self.hyper['retraining']
        batch_size = fit_hyper['batch_size']
        learning_rate = fit_hyper['learning_rate']
        energy_only = fit_hyper['energy_only']
        loss_weights = fit_hyper['loss_weights']
        epo = fit_hyper['epo']
        epostep = fit_hyper['epostep']

        model = EnergyGradientModel(**self.hyper['model'])    # create models
        model.precomputed_features = True
        model.output_as_dict = True
        model.energy_only = energy_only
        model.set_weights(self.model_weight)
        
        cbks = [MPICallback(req_data),]

        # scale x, y
        self._scaler.fit(self.coord, [self.energy, self.force])
        x_rescale, y_rescale = self._scaler.transform(self.coord, [self.energy, self.force])
        y1, y2 = y_rescale

        # Model + Model precompute layer +feat
        feat_x, feat_grad = model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)
        # Finding Normalization
        feat_x_mean, feat_x_std = model.set_const_normalization_from_features(feat_x,normalization_mode=1)

        # Train Test split
        xtrain = [feat_x[self.i_train], feat_grad[self.i_train]]
        ytrain = [y1[self.i_train], y2[self.i_train]]
        xval = [feat_x[self.i_val], feat_grad[self.i_val]]
        yval = [y1[self.i_val], y2[self.i_val]]
        
        print(f"Info: shape of training input feature is {feat_x[self.i_train].shape}")
        
        # Setting constant feature normalization
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        lr_metric = get_lr_metric(optimizer)

        mae_energy = MaskedScaledMeanAbsoluteError(scaling_shape=self._scaler.energy_std.shape)
        mae_force = MaskedScaledMeanAbsoluteError(scaling_shape=self._scaler.gradient_std.shape)
        mae_energy.set_scale(self._scaler.energy_std)
        mae_force.set_scale(self._scaler.gradient_std)
        train_metrics = {'energy': [mae_energy, lr_metric, masked_r2_metric],
                        'force': [mae_force, lr_metric, masked_r2_metric]}
        if energy_only:
            train_loss = {'energy': mask_MeanSquaredError, 'force' : ZeroEmptyLoss()}
        else:
            train_loss = {'energy': mask_MeanSquaredError, 'force': mask_MeanSquaredError}

        model.compile(optimizer=optimizer,
                      loss=train_loss, loss_weights=loss_weights,
                      metrics=train_metrics)
        self._scaler.print_params_info()
        print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
        print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)
        print("Info: Feature data type: ",  feat_x.dtype, feat_grad.dtype)
        print("Info: Target data type: ", ytrain[0].dtype, ytrain[1].dtype)

        print("")
        print("Start fit.")
        model.summary()
        hist = model.fit(x=xtrain, y={'energy': ytrain[0], 'force': ytrain[1]}, epochs=epo, batch_size=batch_size,
                         callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, {'energy': yval[0], 'force': yval[1]}), verbose=2)
        print("End fit.")
        print("")
        
        # remove tensors to free gpu memory
        del xtrain, ytrain, xval, yval, feat_x, feat_grad, y1, y2, y_rescale, x_rescale, self.model_weight
        gc.collect()
        
        # update model weights
        with tf.device('/CPU:0'):
            self.model_weight = model.get_weights()
        del model
        gc.collect()
        
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        self._add_hist(outhist)
        
        # stop_run should be returned as a bool value.
        return stop_run
            
    def save_progress(self, stop_run):
        """
        Save the current progress/data/state.
        Called everytime after retraining and receiving new data points.
        """
        ##### User Part #####
        # save model and scalar weights
        self._scaler.save(os.path.join(self.model_dir, f'scaler_v{self.i_device}.json'))
        weight_path = os.path.join(self.model_dir, f'weights_v{self.i_device}')
        
        if self.mode == 'train':
            with open(weight_path, 'wb') as fh:
                pickle.dump(self.model_weight, fh)
            # save retrain history
            with open(self.hist_path, 'w') as fh:
                json.dump(self.hist, fh)
                
            # save training/validation set
            with open(os.path.join(self.model_dir, f'new_training_set{self.i_device}.npy'), 'wb') as fh:
                np.save(fh, self.coord)
                np.save(fh, self.energy)
                np.save(fh, self.force)
                np.save(fh, self.i_train)
                np.save(fh, self.i_val)
        else:
            with open(weight_path, 'wb') as fh:
                pickle.dump(self._model.get_weights(), fh)

    def stop_run(self):
        """
        Called before the Training/Prediction process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        # save progress before exit
        self.save_progress(stop_run=True)
        # close the log
        self.fstdout.close()

    def _model_setup(self):
        weight_path = os.path.join(self.model_dir, f'weights_v{self.i_device}')
        if os.path.exists(weight_path):
            with open(weight_path, 'rb') as fh:
                self._model.set_weights(pickle.load(fh))
        elif os.path.exists(os.path.join(self.model_dir, f'weights_v{self.i_device}.h5')):
            self._model.load_weights(os.path.join(self.model_dir, f'weights_v{self.i_device}.h5'))
        self._model.precomputed_features = False
        self._model.output_as_dict = False
        self._model.energy_only = self.hyper['retraining']['energy_only']
        if os.path.exists(os.path.join(self.model_dir, f"scaler_v{self.i_device}.json")):
            self._scaler.load(os.path.join(self.model_dir, f"scaler_v{self.i_device}.json"))
        self._weight_shape = []
        weight_list = self._model.get_weights()
        for i in range(0, len(weight_list)):
            self._weight_shape.append(weight_list[i].shape)
        del weight_list
        gc.collect()

    def _add_hist(self, hist_new):
        self.hist['energy_mean_absolute_error'] += hist_new['energy_mean_absolute_error']
        self.hist['energy_r2'] += hist_new['energy_masked_r2_metric']
        self.hist['val_energy_mean_absolute_error'] += hist_new['val_energy_mean_absolute_error']
        self.hist['val_energy_r2'] += hist_new['val_energy_masked_r2_metric']
        self.hist['force_mean_absolute_error'] += hist_new['force_mean_absolute_error']
        self.hist['force_r2'] += hist_new['force_masked_r2_metric']
        self.hist['val_force_mean_absolute_error'] += hist_new['val_force_mean_absolute_error']
        self.hist['val_force_r2'] += hist_new['val_force_masked_r2_metric']
        self.hist['num_epoch'].append(len(hist_new['energy_mean_absolute_error']))

# Callback defined for photodynamics simulation project
class MPICallback(ks.callbacks.Callback):
    def __init__(self, req):
        super(MPICallback, self).__init__()
        self.req = req
        self.status = MPI.Status()
        self.patience = 20
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.inf
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss", None)
        if self.req.Test(self.status):
            self.model.stop_training = True
            print()
            print('New data arrived. Retraining restarts...')
            print()
        elif val_loss != None:
            if np.less(val_loss, self.best):
                self.wait = 0
                self.best = val_loss
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
                    print("Early stopping: validation loss is getting higher!")
