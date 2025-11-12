#! /bin/bash
#SBATCH --job-name=bi4-2_charge_embed
#SBATCH --partition=normal
#SBATCH --time=08:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bi4-2.txt     # Output file
#SBATCH --error=bi4-2error.txt       # Error file
#SBATCH --gres=gpu:1     
#SBATCH --exclude=haicn1707    
#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
#SBATCH --exclude=haicn1707   
srun python train.py \
  --atom bi --num_atom 4 --charge -2  \
  --config charge_embedding.yaml \
  --results_dir results/