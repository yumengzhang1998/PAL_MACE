#! /bin/bash
#SBATCH --job-name=4_head
#SBATCH --partition=accelerated
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi4-2.txt     # Output file
#SBATCH --error=bi4-2error.txt       # Error file
#SBATCH --gres=gpu:1     

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
cd ..
python boot_strap_with_fixed_samples.py --atom bi --num_atom 4 --charge -2 --num_samples 5 --config charge_head.yaml --results_dir results/charge_head --latent True --penalty 1