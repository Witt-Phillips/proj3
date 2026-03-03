#!/bin/bash
#SBATCH --job-name=defamation_v3
#SBATCH --output=cluster_results/run3/results/defamation_v3_%j.out
#SBATCH --error=cluster_results/run3/results/defamation_v3_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=day

# Run3: Legal-BERT, ~250 examples, validation split, early stopping
# Usage: from proj3 root: sbatch cluster_results/run3/run_all_hpc_v3.sh

module load miniconda
conda activate law-llms

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RUN3="cluster_results/run3"
mkdir -p "${RUN3}/results"

echo "=== Run3: Legal-BERT + expanded dataset + early stopping ==="
echo "Dataset: ${RUN3}/defamation_dataset_v3.csv (~250 examples)"
echo "Model: nlpaueb/legal-bert-base-uncased"
echo "Epochs: 12, val_ratio: 0.15, warmup: 0.1"
echo ""

echo "=== Step 1: Training ==="
python defamation_detector.py train \
  --dataset "${RUN3}/defamation_dataset_v3.csv" \
  --output "${RUN3}/defamation_model" \
  --model nlpaueb/legal-bert-base-uncased \
  --epochs 12 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.1 \
  --val-ratio 0.15 \
  --seed 42

echo ""
echo "=== Step 2: Prediction on run2 transcript ==="
python defamation_detector.py predict \
  --model-dir "${RUN3}/defamation_model" \
  --transcript "${RUN3}/../run2/transcript_sample.txt" \
  -o "${RUN3}/results/defamation_results.json"

echo ""
echo "=== Step 3: Prediction on Depp v. Heard transcript ==="
python defamation_detector.py predict \
  --model-dir "${RUN3}/defamation_model" \
  --transcript "Depp v. Heard_transcription.txt" \
  -o "${RUN3}/results/defamation_results_depp_heard.json"

echo ""
echo "=== Done ==="
