#! /bin/bash
#SBATCH --job-name=bi4-6MACE
#SBATCH --partition=normal 
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=60
#SBATCH --cpus-per-task=1
#SBATCH --output=bi4-6.txt          # Output file
#SBATCH --error=bi4-6_err.txt      # Error file

#SBATCH --gres=gpu:2         

#SBATCH --mail-type=END               # Send an email when the job ends
#SBATCH --mail-user=noname19980927@gmail.com  # Email address to send notifications

cd ..
python generate_config_yaml.py --prefix bi4-6 --full_dataset True
python generate_al_setting.py
mpirun -n 60 python main.py
