#! /bin/bash
#SBATCH --job-name=bi11-3_charge_embed
#SBATCH --partition=normal
#SBATCH --time=08:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi11-3.txt     # Output file
#SBATCH --error=bi11-3error.txt       # Error file
#SBATCH --gres=gpu:1     
#SBATCH --exclude=haicn1707  
#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

python train.py --atom bi --num_atom 11 --charge -3 --config charge_embedding.yaml --results_dir results/