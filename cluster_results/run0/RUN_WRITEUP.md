# Bouchet HPC Run Writeup — Defamation Detector

## Overview

We ran the Part 2 defamation detection pipeline on Yale's Bouchet cluster in two stages: a quick validation test, then the full train-and-predict pipeline.

---

## 1. Validation Run

**Purpose:** Smoke test to verify environment, dependencies, and pipeline before the full run.

**Script:** `run_validate_bouchet.sh`  
**Partition:** devel | **Memory:** 8 GB | **Time:** ~18 seconds  
**Run:** Tue Mar 3, 13:47:31–13:47:49 EST 2026

### What we did

- **Step 1 (Train):** Fine-tuned DistilBERT on 4 samples from `defamation_mini.csv` for 1 epoch. Model saved to `test_data/defamation_model_mini`.
- **Step 2 (Predict):** Ran inference on 2 sentences from `transcript_mini.txt`, wrote results to `test_data/validate_results.json`.

### Results

| Sentence                 | Prediction      |
|--------------------------|-----------------|
| Billy is a liar.         | non-defamatory  |
| The weather is nice today. | non-defamatory |

### Issue

"Billy is a liar" is defamatory (it damages reputation), but the model predicted it as non-defamatory. With only 4 training samples and 1 epoch, the model did not learn the task well. The validation "passed" in the sense that the pipeline ran without errors, but model accuracy was poor.

---

## 2. Full Pipeline Run

**Purpose:** Train on the full 20-statement dataset and classify the Depp v. Heard transcript.

**Script:** `run_all_hpc.sh`  
**Job ID:** 5471614  
**Partition:** day | **Memory:** 16 GB | **Cores:** 4 | **Time:** ~2.5 hr limit

### What we did

- **Step 1 (Train):** Fine-tuned DistilBERT on `defamation_dataset.csv` (20 statements: 10 defamatory, 10 non-defamatory) for 4 epochs. Loss decreased from ~0.70 to ~0.33. Model saved to `defamation_model`.
- **Step 2 (Predict):** Classified 28 sentences from `Depp v. Heard_transcription.txt`, wrote results to `defamation_results.json`.

### Results summary

- **21 sentences** predicted defamatory (75%)
- **7 sentences** predicted non-defamatory (25%)

### Why the results are unreliable

1. **Over-prediction of defamation:** Many neutral or procedural statements were labeled defamatory, e.g.:
   - "Please tell the jury why you're here today."
   - "She did."
   - "Where does it end?"
   - "You" (single word)

2. **Small training set:** 20 examples is too few for robust generalization. The model did not see enough variation to distinguish defamatory from non-defamatory in real trial dialogue.

3. **Domain mismatch:** Training used simple, clear examples (e.g., "Billy is a liar" vs. "Billy wears a baseball cap"). The transcript has legal language, dialogue, and Whisper transcription errors (e.g., "It's… Kidney!"), which the model was not trained on.

4. **Transcription noise:** Whisper errors and garbled phrases (e.g., "My livelihood to Clark have gained disguise at the definition case") make the input noisy and harder to classify.

### Stderr warnings (non-fatal)

- `conda: command not found` — transient before `module load miniconda`
- `Some weights... newly initialized` — expected when fine-tuning
- `Can't initialize NVML` — no GPU; job ran on CPU
- `pin_memory... no accelerator` — harmless on CPU

---

## Conclusion

The Bouchet runs completed successfully and the pipeline is functioning. However, the model’s predictions are not trustworthy for real defamation detection due to the small training set and domain mismatch. For a class project, this illustrates the limits of fine-tuning on minimal data and the impact of domain shift between training and deployment.
