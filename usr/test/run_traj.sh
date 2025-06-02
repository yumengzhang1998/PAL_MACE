#!/bin/bash

# List of your prefixes
prefixes=("bi4-6" "bi4-2" "bi7-3" "bi11-3")
prefixes=("bi4-2" "bi7-3")
# Loop and submit a separate SLURM job for each prefix
for prefix in "${prefixes[@]}"; do

    # Extract num_atom and charge from the prefix
    num_atom=$(echo "$prefix" | grep -oP '(?<=bi)\d+')
    charge=$(echo "$prefix" | grep -oP '(?<=-)\d+' | awk '{print -$1}')

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${prefix}_traj
#SBATCH --partition=accelerated
#SBATCH --time=03:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=${prefix}.out
#SBATCH --error=${prefix}.err
#SBATCH --mail-type=END
#SBATCH --mail-user=noname19980927@gmail.com

# Command to run
python batch_traj_full.py --element bi --charge ${charge} --num_atom ${num_atom} --model_number 58 --steps 10000

EOF

    echo "Submitted job for $prefix (num_atom=$num_atom, charge=$charge)"

done
