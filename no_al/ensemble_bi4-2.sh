#! /bin/bash
#SBATCH --job-name=bi4-2MACE
#SBATCH --partition=accelerated 
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi4-2.txt          # Output file
#SBATCH --error=bi4-2_err.txt      # Error file

#SBATCH --gres=gpu:1       

#SBATCH --mail-type=BEGIN,END,FAIL           # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

python run_ensemble_uncertainty.py --atom bi --num_atom 4 --charge -2 --config charge_embedding.yaml 