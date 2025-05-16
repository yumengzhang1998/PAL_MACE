#!/bin/bash

# List of your job scripts
jobs=(
    "bi4-2.sh"
    "bi4-6.sh"
    "bi7-3.sh"
    "bi11-3.sh"
    "bi11-3_samples.sh"
)

# Loop over each job
for job_script in "${jobs[@]}"; do
    # Submit the job
    echo "Submitting $job_script..."
    job_output=$(sbatch "$job_script")
    job_id=$(echo "$job_output" | awk '{print $4}')
    echo "Job submitted with JobID: $job_id"

    # Wait until the job starts running
    echo "Waiting for JobID $job_id to start running..."
    while true; do
        state=$(squeue -j "$job_id" -h -o "%T")
        if [[ "$state" == "RUNNING" ]]; then
            echo "JobID $job_id is now RUNNING."
            break
        elif [[ "$state" == "" ]]; then
            echo "Warning: JobID $job_id disappeared from queue (maybe completed instantly or failed)."
            break
        fi
        sleep 5
    done

    # Rest for 20 seconds before submitting next job
    echo "Waiting 20 seconds before submitting next job..."
    sleep 20
done

echo "All jobs have been submitted and started running!"
