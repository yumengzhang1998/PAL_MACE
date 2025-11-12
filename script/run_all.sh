#!/bin/bash

jobs=(
    "bi11-3_samples.sh"
    "bi4-2.sh"
    "bi4-6.sh"
    "bi7-3.sh"
    "bi11-3.sh"
    "bi11-3_samples_bi2.sh"
    "bi2-2.sh"
)
jobs=(
    "bi4-2.sh"
    "bi7-3.sh"
    "bi11-3.sh"
    "bi2-2.sh"
)
# Submit all jobs directly
for job_script in "${jobs[@]}"; do
    echo "Submitting $job_script..."
    sbatch "$job_script"
done

echo "All jobs have been submitted!"
