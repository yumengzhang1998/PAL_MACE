#!/usr/bin/env bash
#SBATCH --job-name=bi14_fragmix
#SBATCH --partition=accelerated
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/hk-project-aimat2/hv3694/PAL_MACE/usr/pretrain/scripts/charge_embed_scripts/bi14-6_samples_fragmix_%j.out
#SBATCH --error=/home/hk-project-aimat2/hv3694/PAL_MACE/usr/pretrain/scripts/charge_embed_scripts/bi14-6_samples_fragmix_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=noname19980927@gmail.com

set -eo pipefail

readonly PROJECT_ROOT="/home/hk-project-aimat2/hv3694/PAL_MACE"
readonly PRETRAIN_DIR="${PROJECT_ROOT}/usr/pretrain"
readonly PREFIX="bi14-6_samples_fragmix"

source "/home/hk-project-aimat2/hv3694/miniconda3/etc/profile.d/conda.sh"
conda activate mace_al
set -u

export PYTHONUNBUFFERED=1

cd "${PRETRAIN_DIR}"

if [[ ! -f "raw/${PREFIX}_parsed.csv" ]]; then
    printf 'Missing parsed training dataset: %s\n' \
        "${PRETRAIN_DIR}/raw/${PREFIX}_parsed.csv" >&2
    exit 1
fi

printf 'Training five models for %s\n' "${PREFIX}"
python -u boot_strap_with_fixed_samples.py \
    --atom bi \
    --num_atom 14 \
    --charge="-6_samples_fragmix" \
    --num_samples 5 \
    --config charge_embedding.yaml \
    --results_dir results/charge_embedding

# The uncertainty calculation uses the first three trained ensemble members.
for sample_index in 0 1 2; do
    model_path="results/charge_embedding/${PREFIX}_logs/sample_${sample_index}/${PREFIX}.model"
    validation_path="samples/${PREFIX}/sample_${sample_index}/val.csv"

    if [[ ! -f "${model_path}" ]]; then
        printf 'Expected trained model is missing: %s\n' "${model_path}" >&2
        exit 1
    fi
    if [[ ! -f "${validation_path}" ]]; then
        printf 'Expected validation dataset is missing: %s\n' \
            "${validation_path}" >&2
        exit 1
    fi
done

printf 'Calculating three-model validation uncertainty for %s\n' "${PREFIX}"
cd "${PRETRAIN_DIR}/results"
python -u make_val_std_distribution.py \
    --prefix "${PREFIX}" \
    --logs_root charge_embedding \
    --samples_root ../samples \
    --k 3 \
    --device cuda \
    --default_dtype float64

printf 'Finished pretraining and uncertainty calculation for %s\n' "${PREFIX}"
