#! /bin/bash
#SBATCH --job-name=bi7-3_charge_embed
#SBATCH --partition=accelerated
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi7-3.txt     # Output file
#SBATCH --error=bi7-3error.txt       # Error file
#SBATCH --gres=gpu:1     

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
cd ..
python synthetic_boot_train.py --atom bi --num_atom 7 --charge -3 --num_samples 5 --config charge_embedding.yaml --results_dir results/charge_embedding