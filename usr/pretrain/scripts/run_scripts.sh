#!/bin/bash

for dir in */; do
    for script in "$dir"*.sh; do
        if [[ -f "$script" ]]; then
            echo "📤 Submitting $script..."
            base=$(basename "$script" .sh)
            sbatch --output="$dir/${base}_%j.out" --error="$dir/${base}_%j.err" "$script"
        fi
    done
done
