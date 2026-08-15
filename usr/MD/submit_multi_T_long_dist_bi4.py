#!/usr/bin/env python3
import subprocess
from pathlib import Path

# ==============================
# 1. Temperature list
# ==============================
temperatures = [300, 400, 500, 600, 700]

# ==============================
# 2. Global settings for generation + MD
# ==============================
target_dist = 4.5
number = 100
model_number = 56
steps = 1000000

script_name = "bi4_md.py"
log_dir = Path("slurm_logs/bi4")
log_dir.mkdir(parents=True, exist_ok=True)
slurm_dir = Path("slurm_jobs/bi4")
slurm_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# 3. Base SLURM template
# ==============================
slurm_template = """#!/bin/bash
#SBATCH --job-name=bi4_{T}K
#SBATCH --partition=accelerated
#SBATCH --constraint=LSDF
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/bi4/bi4_{T}K_%j.out
#SBATCH --error=slurm_logs/bi4/bi4_{T}K_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=noname19980927@gmail.com
#SBATCH --signal=B:TERM@120

set -euo pipefail

TMPROOT=${{TMPDIR:-/tmp}}
JOB_TMP=$(mktemp -d "${{TMPROOT}}/pal_mace_traj_bi4_{T}K_XXXXXX")
echo "JOB_TMP=$JOB_TMP"

export PAL_MACE_JOB_TMP="$JOB_TMP"

save_results() {{
  echo ">>> save_results triggered"

  SRC="$JOB_TMP/results"
  DEST="/lsdf/kit/int/hv3694/results/PAL_MACE/trajs/bi4/run_${{SLURM_JOB_ID}}_{T}K"

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

python {script_name} \\
  --target_dist {target_dist} \\
  --number {number} \\
  --charge -3 \\
  --model_number {model_number} \\
  --steps {steps} \\
  --T {T}.0 \\
  --out_dir bi4/${{SLURM_JOB_ID}}_{T}K

echo "Job finished cleanly."
"""

# ==============================
# 4. Generate and submit
# ==============================
for T in temperatures:
    slurm_path = slurm_dir / f"run_bi4_{T}K.slurm"

    with open(slurm_path, "w") as f:
        f.write(
            slurm_template.format(
                T=T,
                target_dist=target_dist,
                number=number,
                model_number=model_number,
                steps=steps,
                script_name=script_name,
            )
        )

    print(f"Submitting job for {T}K using {slurm_path}")
    subprocess.run(["sbatch", str(slurm_path)], check=True)
