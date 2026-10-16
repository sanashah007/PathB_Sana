#!/bin/bash
#SBATCH --job-name=cache_ref_baselines
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-51
#SBATCH --output=/home/sanashah/PATHB_sample/logs/cache_ref_%A_%a.out
#SBATCH --error=/home/sanashah/PATHB_sample/logs/cache_ref_%A_%a.err

# 51,702 training-set genomes / 1000 per chunk = 52 chunks (indices 0-51)

cd /home/sanashah/PATHB_sample

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export PYTHONUNBUFFERED=1

mkdir -p logs "metagenome_tests/Adding disagreements Tami/reference_baseline_cache/chunks"

python "metagenome_tests/Adding disagreements Tami/cache_reference_baselines.py" \
    --chunk-idx   "$SLURM_ARRAY_TASK_ID" \
    --chunk-size  1000 \
    --train-split population_perturbation/population_perturbation_results/train_val_test_split.tsv \
    --ref-dir     all_embeded_genomes/
