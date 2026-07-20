#!/bin/bash
#SBATCH --job-name=perturbation
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/perturbation_%j.out
#SBATCH --error=logs/perturbation_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
python pertubation_experiment.py
