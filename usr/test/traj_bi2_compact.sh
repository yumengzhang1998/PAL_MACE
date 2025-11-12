#! /bin/bash
#SBATCH --job-name=bi2syn_200pstraj
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi2syn_200pstraj_output.txt     # Output file
#SBATCH --error=bi2syn_200pstraj_error.txt       # Error file
#SBATCH --gres=gpu:1         

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

python batch_traj_full.py --element bi --charge -3 --num_atom 11 --model_number 56 --steps 100000 --synthesis True --compact_type="bi2"
