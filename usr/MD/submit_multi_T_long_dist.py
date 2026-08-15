#!/usr/bin/env python3
import subprocess
from pathlib import Path

# ==============================
# 1. Temperature list
# ==============================
temperatures = [300, 400, 500, 600, 700]
job_type = "long_dist"
log_dir = Path("slurm_logs") / job_type
log_dir.mkdir(parents=True, exist_ok=True)
slurm_dir = Path("slurm_jobs") / job_type
slurm_dir.mkdir(parents=True, exist_ok=True)
# ==============================
# 2. Base SLURM template
# ==============================
slurm_template = """#!/bin/bash
#SBATCH --job-name=1ns_{T}K
#SBATCH --partition=accelerated
#SBATCH --constraint=LSDF
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/long_dist/long_dist_{T}K_%j.out
#SBATCH --error=slurm_logs/long_dist/long_dist_{T}K_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=noname19980927@gmail.com
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=LSDF
set -euo pipefail

TMPROOT=${{TMPDIR:-/tmp}}
JOB_TMP=$(mktemp -d "${{TMPROOT}}/pal_mace_traj_{T}K_XXXXXX")
echo "JOB_TMP=$JOB_TMP"

export PAL_MACE_JOB_TMP="$JOB_TMP"

save_results() {{
  echo ">>> save_results triggered"

  SRC="$JOB_TMP/results"
  DEST="/lsdf/kit/int/hv3694/results/PAL_MACE/trajs/long_dist/run_${{SLURM_JOB_ID}}_{T}K"

  if [ -d "$SRC" ]; then
    if [ -d /lsdf ]; then
      mkdir -p "$DEST"
      rsync -av "$SRC/" "$DEST/"
      echo ">>> Results copied to $DEST"
    else
      echo "!!! LSDF not mounted — results remain in $SRC"
    fi
  else
    echo "!!! No results directory found"
  fi

  rm -rf "$JOB_TMP"
}}


trap save_results EXIT TERM INT

python long_distance_md.py \
  --npz ./building_blocks/bi_mixed_dataset.npz \
  --charge -3 \
  --model_number 56 \
  --steps 1000000 \
  --T {T}.0

echo "Job finished cleanly."
"""

# ==============================
# 3. Generate and submit
# ==============================

for T in temperatures:
    slurm_path = slurm_dir / f"run_long_dist_{T}K.slurm"

    with open(slurm_path, "w") as f:
        f.write(slurm_template.format(T=T))

    print(f"Submitting job for {T}K using {slurm_path}")
    subprocess.run(["sbatch", str(slurm_path)], check=True)
