#! /bin/bash
#SBATCH --job-name=bi2_syn_charge_embed
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi2_syn.txt     # Output file
#SBATCH --error=bi2_syn_error.txt       # Error file
#SBATCH --gres=gpu:1     
#SBATCH --exclude=haicn1707  
#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
cd ../..
python boot_strap_with_fixed_samples.py --atom bi --num_atom 11 --charge='-3_samples_bi2' --num_samples 5 --config charge_embedding.yaml --results_dir results/charge_embedding