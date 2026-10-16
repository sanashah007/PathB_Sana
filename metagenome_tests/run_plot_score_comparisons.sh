#!/bin/bash
#SBATCH --job-name=plot_score_comparisons
#SBATCH --partition=pi_tami
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH -n 1
#SBATCH --output=logs/plot_score_comparisons_%j.out
#SBATCH --error=logs/plot_score_comparisons_%j.err

mkdir -p logs
cd /home/sanashah/PATHB_sample
conda run -n PathB python metagenome_tests/plot_score_comparisons.py
