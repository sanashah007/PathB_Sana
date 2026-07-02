#!/bin/bash
#SBATCH --job-name=genome_processing
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=output_trial.out
#SBATCH --error=error_trial.err

python train_model.py

