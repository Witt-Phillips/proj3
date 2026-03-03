#!/bin/bash
#SBATCH --job-name=defamation_v2
#SBATCH --output=cluster_results/run1/results/defamation_v2_%j.out
#SBATCH --error=cluster_results/run1/results/defamation_v2_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=day

# Run1: Improved defamation detector
# - Expanded dataset (62 examples vs 20)
# - BERT-base instead of DistilBERT
# - 12 epochs, learning rate warmup
# Usage: from proj3 root: sbatch cluster_results/run1/run_all_hpc_v2.sh

module load miniconda
conda activate law-llms

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RUN1="cluster_results/run1"
mkdir -p "${RUN1}/results"

echo "=== Run1: Improved defamation detector ==="
echo "Dataset: ${RUN1}/defamation_dataset_expanded.csv (62 examples)"
echo "Model: bert-base-uncased"
echo "Epochs: 12, warmup: 0.1"
echo ""

echo "=== Step 1: Training ==="
python defamation_detector.py train \
  --dataset "${RUN1}/defamation_dataset_expanded.csv" \
  --output "${RUN1}/defamation_model" \
  --model bert-base-uncased \
  --epochs 12 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.1

echo ""
echo "=== Step 2: Prediction ==="
python defamation_detector.py predict \
  --model-dir "${RUN1}/defamation_model" \
  --transcript "Depp v. Heard_transcription.txt" \
  -o "${RUN1}/defamation_results.json"

echo ""
echo "=== Done ==="
