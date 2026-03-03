#!/bin/bash
#SBATCH --job-name=defamation_validate
#SBATCH --output=validate_defamation_%j.out
#SBATCH --error=validate_defamation_%j.err
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=devel

# Short validation test for Bouchet (~2-3 min)
# Verifies: env, deps, train, predict
# Usage: sbatch run_validate_bouchet.sh

set -e

module load miniconda
conda activate law-llms

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Bouchet validation: defamation detector ==="
echo "Start: $(date)"

echo ""
echo "--- Step 1: Train (4 samples, 1 epoch) ---"
python defamation_detector.py train \
  --dataset test_data/defamation_mini.csv \
  --output test_data/defamation_model_mini \
  --epochs 1 \
  --batch-size 2

echo ""
echo "--- Step 2: Predict on minimal transcript ---"
python defamation_detector.py predict \
  --model-dir test_data/defamation_model_mini \
  --transcript test_data/transcript_mini.txt \
  -o test_data/validate_results.json

echo ""
echo "--- Results ---"
cat test_data/validate_results.json

echo ""
echo "End: $(date)"
echo "=== Validation PASSED ==="
