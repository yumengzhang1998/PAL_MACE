# PAL for excited states molecular dynamics (MD) simulations

Employ PAL to enable the simulation of multiple excited-state potential energy surfaces of a small molecule organic semiconductor, where fully connected neural networks implemented with NNsForMD (https://github.com/aimat-lab/NNsForMD) are utilized in Prediction and Training kernels to predict the ground-state energy and excited-state energy levels. The processes in the Generator kernel propagate MD trajectories and generate new atomic coordinates with PyRAI2MD developed by Jingbai Li et al (https://github.com/lopez-lab/PyRAI2MD). In the oracle kernel, accurate energy and force labels are computed using time-dependent density functional theory (TDDFT) at the B3LYP/6-31G* level of theory with TURBOMOLE (https://www.turbomole.org/).

Install packages. Note this will also install NNsForMD and PyRAI2MD packages modified specifily for this project.
```
bash install_tools/install_pyNNsMD.sh
bash install_tools/install_pyrai2MD.sh
bash install_tools/install_photoMD.sh
```
Set the path to the photoMD example kernels in ``al_setting``:
```
"usr_pkg": {                           
    "generator": "./usr_example/photoMD/generator.py", # path to the Generator kernel
    "model": "./usr_example/photoMD/model.py",         # path to the Prediction/Training kernel
    "oracle": "./usr_example/photoMD/oracle.py",       # path to the Oracle kernel
    "utils": "./usr_example/photoMD/utils.py",         # path to utility functions
},

"pred_process": 4,                                     # number of prediction processes
"orcl_process": 28,                                    # number of oracle processes
"gene_process": 90,                                    # number of generator processes
"ml_process": 4,                                       # number of machine learning processes
```

Initialize 128 processes on 2 GPU nodes for 1 hour on a computational cluster with Slurm system
```
#!/bin/sh

#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10240mb
#SBATCH --job-name=test_toy
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4

module load chem/turbomole/7.7.1
module load mpi/openmpi/4.1
module load devel/cuda/12.4

export OMPI_MCA_coll_hcoll_enable=1
export UCX_TLS=dc,self,posix,sysv,cma

mpirun --bind-to core --map-by core --rank-by slot -report-bindings python3 main.py
```
