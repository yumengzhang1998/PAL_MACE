#!/usr/bin/env bash
set -euo pipefail

# Target directory names (sorted longest → shortest to avoid partial matches)
targets=("bi11-3_samples_bi2" "bi11-3_samples" "bi11-3" "bi7-3" "bi4-6" "bi4-2" "bi2-2")

# Output directory
outdir="./collected_data"

# Create output subdirs
for t in "${targets[@]}"; do
    mkdir -p "$outdir/$t"
done

# Find all relevant CSV files at any depth
find . -type f \( -name "56_added_data.csv" -o -name "57_added_data.csv" -o -name "added_data.csv" \) | while read -r file; do
    filepath=$(realpath "$file")

    # Split the path into components
    IFS='/' read -ra parts <<< "$filepath"

    matched_dir=""
    for t in "${targets[@]}"; do
        for p in "${parts[@]}"; do
            if [[ "$p" == "$t" ]]; then
                matched_dir="$t"
                break 2  # stop at first exact match
            fi
        done
    done

    # Skip if no matching directory
    [[ -z "$matched_dir" ]] && continue

    base=$(basename "$file" .csv)
    dest_dir="$outdir/$matched_dir"
    dest_base="${dest_dir}/${base}.csv"

    # Avoid overwriting by appending _1, _2, etc.
    if [[ -e "$dest_base" ]]; then
        i=1
        while [[ -e "${dest_dir}/${base}_${i}.csv" ]]; do
            ((i++))
        done
        dest="${dest_dir}/${base}_${i}.csv"
    else
        dest="$dest_base"
    fi

    cp "$file" "$dest"
    echo "Copied $file → $dest"
done
