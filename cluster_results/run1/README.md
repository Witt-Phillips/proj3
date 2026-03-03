# Run1 — Improved Defamation Detector

Experimental run to improve model performance over run0.

## Changes from run0

| Setting | run0 | run1 |
|---------|------|------|
| Dataset | 20 examples | **62 examples** (expanded) |
| Model | DistilBERT | **BERT-base-uncased** |
| Epochs | 4 | **12** |
| Warmup | none | **10%** |
| Memory | 16 GB | **24 GB** |
| Time limit | 2.5 hr | **4 hr** |

## Files

- `defamation_dataset_expanded.csv` — 62 labeled examples (30 defamatory, 32 non-defamatory)
- `run_all_hpc_v2.sh` — SLURM script for Bouchet

## Usage

From the project root (`proj3/`):

```bash
sbatch cluster_results/run1/run_all_hpc_v2.sh
```

## Results

Outputs live in `results/`:

- `defamation_model/` — fine-tuned BERT (in run1 root)
- `defamation_results.json` — per-sentence predictions on Depp v. Heard transcript
- `defamation_v2_<jobid>.out` / `.err` — SLURM logs

### Run summary (job 5473107)

- **Training:** Loss 0.59 → 0.0005 over 12 epochs (~30 sec)
- **Predictions:** 28 sentences; mix of defamatory/non-defamatory (no longer all 1s)
- **Improvements over run0:** "Please tell the jury why you're here today" and "Where does it end?" now correctly non-defamatory; "Look at me, this kid!" correctly non-defamatory
- **Remaining issues:** Garbled Whisper transcript ("It's… Kidney!", "People are alive between people") still confuses the model; rhetorical questions sometimes misclassified

## Next: Run2

Apply the run1 model to a **cleaner document** (no Whisper errors) to see how it performs on well-formed text. See `../run2/`.

## Fallback (if BERT OOMs)

Edit the script and change `--model bert-base-uncased` to `--model distilbert-base-uncased`, and `--mem=24G` to `--mem=16G`.
