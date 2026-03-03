#!/bin/bash
#SBATCH --job-name=defamation_train
#SBATCH --output=defamation_train_%j.out
#SBATCH --error=defamation_train_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=day

# Part 2: Fine-tune defamation detector on Yale Bouchet HPC
# Usage: sbatch run_defamation_train.sh

# Load modules (Yale Bouchet: module load miniconda)
module load miniconda

# Activate environment (conda recommended on Yale HPC)
conda activate law-llms
# Or if using venv: source venv/bin/activate

# Run from project directory (adjust path if submitting from elsewhere)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Train the model
python defamation_detector.py train \
  --dataset defamation_dataset.csv \
  --output defamation_model \
  --epochs 4 \
  --batch-size 4

echo "Training complete. Model saved to defamation_model/"
