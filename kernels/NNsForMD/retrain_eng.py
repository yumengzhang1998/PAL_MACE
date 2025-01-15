import numpy as np
import json
import pickle

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([-1]) #No GPU for prediciton or this main class

from pyNNsMD.nn_pes import NeuralNetPes
import tensorflow as tf

with open('blueOLED/data/data_44826_new.json', 'r') as indata:
    data = json.load(indata)

with open('blueOLED/retrain_data_idx/retrain_data_index_44826new.json', 'r') as f:
    idx_lis = json.load(f)

coords, engs, grads, engindex, gradindex = data
#coords, engs, grads = data
#Target Properties y
x = np.array(coords)
x = np.array(x[:,:,1:],dtype=float)  
#grads = np.array(grad) * 27.21138624598853/0.52917721090380  #Hatree to eV and
Energy = np.array(engs, dtype=float) * 27.21138624598853  #Hatree to eV
Energy = np.array(Energy[:,:4])
x_select = x[idx_lis]
Energy_select = Energy[idx_lis]
#nacs= np.array(nac)/0.52917721090380 #Bohr to A

nn = NeuralNetPes("blueOLED/res/adaptive_learning_models", mult_nn=3)

hyper_energy =  {  
                'retraining':{
                    'learning_rate': 10 ** -6.0,
                    'normalization_mode' : 1,
                    'epo': 2,
                    'val_split' : 0.3,
                    'batch_size' : 64,
                    'initialize_weights': False,
                    'linear_callback' : {'use' : True, 'learning_rate_start' : 1e-6,'learning_rate_stop' : 1e-7, 'epomin' : 50, 'epo': 200},
                    #'log_callback': {'use': True, 'learning_rate_start': 1e-6, 'learning_rate_stop': 1e-7, 'epo': 20, 'epomin': 20, 'epomax': 50},
                    #'step_callback' : {'use': True , 'epoch_step_reduction' : [2000,2000,500,500],'learning_rate_step' :[10 ** -6.0,10 ** -7.0,10 ** -8.0,10 ** -9.0]},
                }
                }

nn.load(model_name='e')

nn.update(
    hyp_dict=hyper_energy
)

y = {'e': Energy_select}

fitres = nn.fit(x_select,
                y,
                gpu_dist={},
                #gpu_dist= {'e': [0, 1, 2]},
                proc_async=True,
                fitmode='retraining',
                random_shuffle=True)