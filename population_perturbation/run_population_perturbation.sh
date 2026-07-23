#!/bin/bash
#SBATCH --job-name=pop_perturbation
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/pop_perturbation_%j.out
#SBATCH --error=logs/pop_perturbation_%j.err

cd /home/sanashah/PATHB_sample

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PathB
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
python /home/sanashah/PATHB_sample/population_perturbation/population_perturbation.py
