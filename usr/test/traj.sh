#! /bin/bash
#SBATCH --job-name=bisyn_10pstraj
#SBATCH --partition=accelerated
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bisyn_10pstraj_output.txt     # Output file
#SBATCH --error=bisyn_10pstraj_error.txt       # Error file
#SBATCH --gres=gpu:1         

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

python batch_traj_full.py --element bi --charge -3 --num_atom 11 --model_number 55 --steps 10000 --synthesis True
