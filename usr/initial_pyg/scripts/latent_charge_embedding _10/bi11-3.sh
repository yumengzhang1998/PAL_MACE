#! /bin/bash
#SBATCH --job-name=11_L_E
#SBATCH --partition=accelerated
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi11-3.txt     # Output file
#SBATCH --error=bi11-3error.txt       # Error file
#SBATCH --gres=gpu:1     

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
cd ..
python boot_strap_with_fixed_samples.py --atom bi --num_atom 11 --charge -3 --num_samples 5 --config latentcharge_embed.yaml --results_dir results/latent_charge_embed_10 --penalty 10 --latent True