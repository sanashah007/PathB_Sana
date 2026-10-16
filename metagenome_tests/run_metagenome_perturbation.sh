#!/bin/bash
#SBATCH --job-name=meta_perturb
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/sanashah/PATHB_sample/logs/meta_perturb_%j.out
#SBATCH --error=/home/sanashah/PATHB_sample/logs/meta_perturb_%j.err

cd /home/sanashah/PATHB_sample

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export PYTHONUNBUFFERED=1

mkdir -p logs metagenome_tests/metagenome_perturbation_results

python metagenome_tests/metagenome_perturbation.py \
    --vectors-dir metagenome_tests/metagenome_vectors/ \
    --cutoffs     population_perturbation/population_perturbation_results/per_gene_cutoffs.tsv \
    --out-dir     metagenome_tests/metagenome_perturbation_results/
