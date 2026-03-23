#!/bin/bash
#SBATCH --job-name=1ns_550K
#SBATCH --partition=accelerated
#SBATCH --constraint=LSDF
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=bisyn_2nstraj_550K_%j.out
#SBATCH --error=bisyn_2nstraj_550K_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=noname19980927@gmail.com
#SBATCH --signal=B:TERM@120
#SBATCH --constraint=LSDF
set -euo pipefail

TMPROOT=${TMPDIR:-/tmp}
JOB_TMP=$(mktemp -d "${TMPROOT}/pal_mace_traj_550K_XXXXXX")
echo "JOB_TMP=$JOB_TMP"

export PAL_MACE_JOB_TMP="$JOB_TMP"

save_results() {
  echo ">>> save_results triggered"

  SRC="$JOB_TMP/results"
  DEST="/lsdf/kit/int/hv3694/results/PAL_MACE/trajs/run_${SLURM_JOB_ID}_550K"

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
}


trap save_results EXIT TERM INT

python batch_traj_full_h5.py   --element bi   --charge -3   --num_atom 11   --model_number 56   --steps 1000000   --synthesis True   --T 550.0

echo "Job finished cleanly."
