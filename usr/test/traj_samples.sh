#! /bin/bash
#SBATCH --job-name=bi11-3_samples_ps100traj
#SBATCH --partition=accelerated
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi11-3_samples_ps10_output.txt     # Output file
#SBATCH --error=bi11-3_samples_ps10_error.txt       # Error file
#SBATCH --gres=gpu:1         

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

python batch_traj_samples.py --element bi --charge -3 --num_atom 11 --model_number 25 --steps 10000
