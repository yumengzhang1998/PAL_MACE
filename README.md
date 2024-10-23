# <img src="https://github.com/user-attachments/assets/548da441-e718-4647-912b-ef3ddd34ba61" width="250"> Parallel Active Learning 
Parallel active learning (PAL) workflow with data and task parallelism through Message Passing Interface (MPI) and mpi4py.

## Features
* The automatic workflow reduces human intervention in active learning.
* The machine learning training (ML) and inference processes are decoupled, enabling data and task parallelism for data generation, labeling, and training tasks.
* PAL is designed in a modular and highly adaptive fashion that can be extended to different tasks with various combinations of resources, data, and ML model types.
* Implemented with MPI and its Python package (mpi4py), PAL is scalable and can be deployed flexibly on shared- (e.g., laptop) and distributed-memory systems (e.g., computer cluster).

## Prerequisite
* Python >= 3.9
* mpi4py >= 3.1 with openmpi
* Matplotlib/Numpy
* openmpi == 4.1

## Run PAL
Initialize 20 processes locally
  ```
  mpirun -n 20 python main.py
  ```
Initialize 20 processes on 2 nodes for 1 hour on a computational cluster with Slurm system
```
#!/bin/sh

#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1

export OMPI_MCA_coll_hcoll_enable=1
export UCX_TLS=dc,self,posix,sysv,cma

mpirun --bind-to core --map-by core --rank-by slot -report-bindings python main.py
```
