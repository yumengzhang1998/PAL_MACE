"""
Settings for the ML models of the Prediction and the Training kernel
"""

# path to the directory (in the result directory) with model files
model_path = "mlp_models"

# model hyperparameters
ml_hyper = {
            'general':{
                'model_type' : 'mlp_eg',
                'batch_size_predict': 265,
                },
            'model':{
                'use_dropout': False,
                "dropout": 0.0,
                'atoms': 38,
                'states': 7,
                'depth' : 6,
                'nn_size' : 5000,   # number of neurons per layer
                'use_reg_weight' : {'class_name': 'L2', 'config': {'l2': 1e-4}},
                "activ": {"class_name": "sharp_softplus", "config": {"n": 10.0}},
                'invd_index' : True,
                #'activ': 'relu',
                },
            'retraining':{
                "energy_only": False,
                "masked_loss": True,
                "auto_scaling": {"x_mean": True, "x_std": True, "energy_std": True, "energy_mean": True},
                "loss_weights": {"energy": 1, "force": 1},
                'learning_rate': 1e-07,
                "initialize_weights": False,
                "val_disjoint": True,
                'normalization_mode' : 1,
                'epo': 10000,
                'val_split' : 0.25,
                'batch_size' : 32,
                "epostep": 10,
                "exp_callback": {"use": False, "factor_lr": 0.9, "epomin": 100, "learning_rate_start": 1e-06, "epostep": 20},
                }
            }