#!/usr/bin/env python3
import subprocess
from pathlib import Path

# ==============================
# 1. Temperature list
# ==============================
temperatures = [300, 400, 500, 600, 700]
temperatures = [500,525,575]
job_type = "batch_traj"
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
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/batch_traj/batch_traj_{T}K_%j.out
#SBATCH --error=slurm_logs/batch_traj/batch_traj_{T}K_%j.err
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
  DEST="/lsdf/kit/int/hv3694/results/PAL_MACE/trajs/run_${{SLURM_JOB_ID}}_{T}K"

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

python batch_traj_full_h5.py \
  --element bi \
  --charge -3 \
  --num_atom 11 \
  --model_number 56 \
  --steps 1000000 \
  --synthesis True \
  --T {T}.0

echo "Job finished cleanly."
"""

# ==============================
# 3. Generate and submit
# ==============================

for T in temperatures:
    slurm_path = slurm_dir / f"run_batch_traj_{T}K.slurm"

    with open(slurm_path, "w") as f:
        f.write(slurm_template.format(T=T))

    print(f"Submitting job for {T}K using {slurm_path}")
    subprocess.run(["sbatch", str(slurm_path)], check=True)
