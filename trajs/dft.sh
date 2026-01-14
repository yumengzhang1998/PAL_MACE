#! /bin/bash
#SBATCH --job-name=syn
#SBATCH --partition=cpuonly
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=syn.txt     # Output file
#SBATCH --error=syn_ERR.txt       # Error file        
#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

python dft_ml_energy_traj_fixed_sample.py --element bi --charge -3 --num_atom 11 --model_number 56 --steps 100000 --synthesis True --n_jobs 16 --threads_per_job 8 --dft_interval 5000 --project "retrained_298k"