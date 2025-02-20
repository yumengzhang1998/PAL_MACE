#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:49:49 2023

@author: chen
"""

AL_SETTING = {
    "result_dir": '../results/TestRun',    # directory to save all metadata and results
    "orcl_buffer_path": '../results/TestRun/ml_buffer',    # path to save data ready to send to ML. Set to None to skip buffer backup.
    "ml_buffer_path": '../results/TestRun/orcl_buffer',    # path to save data ready to send to Oracle. Set to None to skip buffer backup.

    # Number of process in total = 2 MPI communication processes (Manager and Exchange)
    #                              + pred_process + orcl_process + gene_process + ml_process
    "pred_process": 2,                     # number of prediction processes
    "orcl_process": 20,                     # number of oracle processes
    "gene_process": 38,                    # number of generator processes
    "ml_process": 2,                       # number of machine learning processes
    "designate_task_number": True,         # set to True if need to specify the number of tasks running on each node (e.g. number of model per computation node)
                                           # if False, tasks are arranged randomly
    "fixed_size_data": True,              # set to True if data communicated among kernels have fixed sizes.
                                           # if false, additional communications are necessary for each iteration to exchange data size info thus lower efficiency.
    "task_per_node":{                      # designate the number of tasks per node, used only if designate_task_number is True
        "prediction": [2,],              # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "generator": None,                 # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "oracle": None,                    # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "learning": [2,],                # list for the number of tasks per node (length must matches the number of nodes), None for no limit
    },
    "orcl_time": 10,                       # Oracle calculation time in seconds
    "progress_save_interval": 60,          # time interval (in seconds) to save the progress
    "retrain_size": 20,                    # batch size of increment retraining set
    "dynamic_orcale_list": True,           # adjust data points for orcale calculation based on ML predictions everytime when retrainings finish
    "gpu_pred": [0, 1],                        # gpu index list for prediction processes
    "gpu_ml": [2, 3],                          # gpu index list for machine learning
    "usr_pkg": {                           # dictionary of paths to user implemented modules (generator, model, oracle and utils)
        "generator": "./usr_example/photoMD/generator.py",
        "model": "./usr_example/photoMD/model.py",
        "oracle": "./usr_example/photoMD/oracle.py",
        "utils": "./usr_example/photoMD/utils.py",
    },
    }
