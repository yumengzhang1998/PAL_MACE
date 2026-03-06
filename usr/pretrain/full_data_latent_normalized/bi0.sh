#! /bin/bash
#SBATCH --job-name=full_bi_init
#SBATCH --partition=accelerated
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=output.txt     # Output file
#SBATCH --error=error.txt       # Error file
#SBATCH --gres=gpu:1     

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications
python full_charge_embde.py 