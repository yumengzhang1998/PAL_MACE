#!/bin/bash
# submit_all.sh
# Usage: bash submit_all.sh

# Loop over all job scripts matching bi*.sh
for job in bi*.sh; do
    if [[ -f "$job" ]]; then
        echo "Submitting $job ..."
        sbatch "$job"
        sleep 1   # small delay to avoid overloading scheduler
    fi
done
