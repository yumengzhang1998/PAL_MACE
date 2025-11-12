#! /bin/bash
#SBATCH --job-name=syn_charge_embed
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi11-3_samples.txt     # Output file
#SBATCH --error=bi11-3_samples_error.txt       # Error file
#SBATCH --gres=gpu:1     
#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
python train.py --atom bi --num_atom 11 --charge="-3_samples" --config charge_embedding.yaml --results_dir results/
