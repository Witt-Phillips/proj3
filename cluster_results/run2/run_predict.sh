#!/bin/bash
#SBATCH --job-name=defamation_run2
#SBATCH --output=cluster_results/run2/results/defamation_run2_%j.out
#SBATCH --error=cluster_results/run2/results/defamation_run2_%j.err
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=day

# Run2: Apply run1 model to a cleaner document (no Whisper errors)
# Prediction only — uses trained model from run1
# Usage: from proj3 root: sbatch cluster_results/run2/run_predict.sh

module load miniconda
conda activate law-llms

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RUN1="cluster_results/run1"
RUN2="cluster_results/run2"

mkdir -p "${RUN2}/results"

echo "=== Run2: Defamation check on clean document ==="
echo "Model: ${RUN1}/defamation_model (from run1)"
echo "Input: ${RUN2}/transcript_sample.txt"
echo ""

python defamation_detector.py predict \
  --model-dir "${RUN1}/defamation_model" \
  --transcript "${RUN2}/transcript_sample.txt" \
  -o "${RUN2}/results/defamation_results.json"

echo ""
echo "=== Done ==="
