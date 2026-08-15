#! /bin/bash
#SBATCH --job-name=bi14-6_samples_charge_embed
#SBATCH --partition=accelerated
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi14-6_samples.txt     # Output file
#SBATCH --error=bi14-6_samples_error.txt       # Error file
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,FAIL,END     # Send email when the job starts, fails, or ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
cd ../..
python boot_strap_with_fixed_samples.py --atom bi --num_atom 14 --charge="-6_samples" --num_samples 5 --config charge_embedding.yaml --results_dir results/charge_embedding
