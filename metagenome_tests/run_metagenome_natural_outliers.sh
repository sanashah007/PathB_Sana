#!/bin/bash
#SBATCH --job-name=meta_outliers
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/sanashah/PATHB_sample/logs/meta_outliers_%j.out
#SBATCH --error=/home/sanashah/PATHB_sample/logs/meta_outliers_%j.err

cd /home/sanashah/PATHB_sample

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export PYTHONUNBUFFERED=1

mkdir -p logs metagenome_tests/metagenome_natural_outliers

python metagenome_tests/metagenome_natural_outliers.py \
    --vectors-dir   metagenome_tests/metagenome_vectors/ \
    --assignments   metagenome_tests/assignments.tsv \
    --ref-dir       all_embeded_genomes/ \
    --dor           population_perturbation/population_perturbation_results/per_gene_dor.tsv \
    --out-dir       metagenome_tests/metagenome_natural_outliers/ \
    --extra-quantiles 0.999,0.9999
