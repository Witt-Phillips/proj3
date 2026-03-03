#!/bin/bash
#SBATCH --job-name=defamation_predict
#SBATCH --output=defamation_predict_%j.out
#SBATCH --error=defamation_predict_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=day

# Part 2: Run defamation detection on transcript (Yale Bouchet HPC)
# Usage: sbatch run_defamation_predict.sh
# Prerequisite: Run run_defamation_train.sh first to create defamation_model/

module load miniconda
conda activate law-llms

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python defamation_detector.py predict \
  --model-dir defamation_model \
  --transcript "Depp v. Heard_transcription.txt" \
  -o defamation_results.json

echo "Prediction complete. Results in defamation_results.json"
