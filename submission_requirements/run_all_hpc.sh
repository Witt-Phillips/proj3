#!/bin/bash
#SBATCH --job-name=defamation_full
#SBATCH --output=defamation_full_%j.out
#SBATCH --error=defamation_full_%j.err
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=day

# Run full Part 2 pipeline: train + predict (Yale Bouchet HPC)
# Usage: sbatch run_all_hpc.sh

module load miniconda
conda activate law-llms

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Step 1: Training ==="
python defamation_detector.py train \
  --dataset defamation_dataset.csv \
  --output defamation_model \
  --epochs 4 \
  --batch-size 4

echo "=== Step 2: Prediction ==="
python defamation_detector.py predict \
  --model-dir defamation_model \
  --transcript "Depp v. Heard_transcription.txt" \
  -o defamation_results.json

echo "=== Done ==="
