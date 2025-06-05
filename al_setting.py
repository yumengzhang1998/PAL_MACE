#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:49:49 2023

@author: chen
"""

AL_SETTING = {
    "result_dir": './results/bi4-2',    # directory to save all metadata and results
    "orcl_buffer_path": './results/bi4-2/orcl_buffer',    # path to save data ready to send to ML. Set to None to skip buffer backup.
    "ml_buffer_path": './results/bi4-2/ml_buffer',    # path to save data ready to send to Oracle. Set to None to skip buffer backup.

    # Number of process in total = 2 MPI communication processes (Manager and Exchange)
    #                              + pred_process + orcl_process + gene_process + ml_process
    "pred_process": 2,                     # number of prediction processes
    "orcl_process": 50,                     # number of oracle processes
    "gene_process": 4,                    # number of generator processes
    "ml_process": 2,                       # number of machine learning processes
    "designate_task_number": False,         # set to True if need to specify the number of tasks running on each node (e.g. number of model per computation node)
                                           # if False, tasks are arranged randomly
    "fixed_size_data": False,              # set to True if data communicated among kernels have fixed sizes.
                                           # if false, additional communications are necessary for each iteration to exchange data size info thus lower efficiency.
    "task_per_node":{                      # designate the number of tasks per node, used only if designate_task_number is True
        "prediction": None,              # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "generator": None,                 # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "oracle": None,                    # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "learning": None,                # list for the number of tasks per node (length must matches the number of nodes), None for no limit
    },
    "orcl_time": 10,                       # Oracle calculation time in seconds
    "progress_save_interval": 60,          # time interval (in seconds) to save the progress
    "retrain_size": 50,                    # batch size of increment retraining set
    "dynamic_orcale_list": True,           # adjust data points for orcale calculation based on ML predictions everytime when retrainings finish
    "gpu_pred": [],                        # gpu index list for prediction processes
    "gpu_ml": [],                          # gpu index list for machine learning
    "usr_pkg": {                           # dictionary of paths to user implemented modules (generator, model, oracle and utils)
        "generator": "./usr/generator.py",
        "model": "./usr/model.py",
        "oracle": "./usr/oracle.py",
        "utils": "./usr/utils.py",
    },
    }
