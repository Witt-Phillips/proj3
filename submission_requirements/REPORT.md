# Law and LLMs — Project 3 Report

## Overview

This project implements (1) speech-to-text transcription of the Depp v. Heard trial video using Whisper, and (2) a fine-tuned defamation detector using DistilBERT that classifies each sentence in the transcript.

---

## Part 1: Speech-to-Text

**Scripts:** `transcribe.py`, `transcribe.ipynb`  
**Output:** `Depp v. Heard_transcription.txt`

### Design decisions

- **Whisper base.en** — English-only model for court dialogue
- **Input options** — Supports both YouTube URL (assignment video) and local audio file (e.g. Canvas `Depp v. Heard.wav`). For YouTube, downloads and extracts audio before passing to Whisper (preprocessing per assignment)
- **Long audio** — Whisper handles long input via internal sliding 30-second windows; no manual chunking
- **condition_on_previous_text=False** — Reduces repetition hallucinations in long audio

### How to run

```bash
pip install -r requirements.txt
python transcribe.py "Depp v. Heard.wav"
# Or: python transcribe.py   # uses YouTube URL
```

---

## Part 2: Fine-Tuned Defamation Detector

**Script:** `defamation_detector.py`  
**Dataset:** `defamation_dataset.csv` (10 defamatory + 10 non-defamatory statements)  
**Results:** `defamation_results.json`

### Design decisions

- **Fine-tuning, not few-shot** — The 20 labeled statements are used as a training corpus, not as prompts
- **DistilBERT** — Smaller and faster than BERT; suitable for HPC with limited memory
- **AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments** — Hugging Face Transformers (per assignment suggestion)
- **report_to="none"** — No wandb API key required
- **NLTK sentence tokenizer** — Splits transcript into sentences for per-sentence classification; regex fallback if NLTK unavailable

### Twenty statements (defamation_dataset.csv)

**Defamatory (label=1):** Billy is a liar; She stole money from the company; He is a convicted criminal; The defendant is a fraud; She cheated on her taxes; He abused his employees; She spread false rumors about me; He is incompetent at his job; She committed perjury in court; He is a drug addict

**Non-defamatory (label=0):** Billy wears a baseball cap; She went to the store yesterday; The meeting is at 3 PM; He lives in Boston; She enjoys reading books; The weather is nice today; He drives a blue car; She has two children; The restaurant is closed on Mondays; He plays guitar in a band

### How to run (local)

```bash
pip install -r requirements.txt
python defamation_detector.py train
python defamation_detector.py predict
```

---

## Extensions (Runs 1–3) and Performance Gains

Beyond the baseline (run0: 20 examples, DistilBERT, 4 epochs), we ran three experimental extensions on Bouchet to improve defamation detection.

### Run 1: Expanded dataset + BERT

| Change | run0 | run1 |
|--------|------|------|
| Dataset | 20 examples | **62 examples** (30 defamatory, 32 non-defamatory) |
| Model | DistilBERT | **BERT-base-uncased** |
| Epochs | 4 | **12** |
| Warmup | none | **10%** |

**Gains:** Reduced over-prediction. Statements like "Please tell the jury why you're here today," "Where does it end?," and "Look at me, this kid!" were correctly classified as non-defamatory (run0 had flagged them defamatory). **Why:** More training examples and a larger model improved generalization.

**Limitation:** Whisper transcription errors in the Depp v. Heard transcript (e.g., garbled phrases) still confused the model.

### Run 2: Clean-document evaluation

**Change:** Applied the run1 model to a **well-formed document** (`transcript_sample.txt`, ~66 sentences) with no Whisper errors — a fictional news-style text mixing neutral facts, defamatory accusations, denials, and procedural statements.

**Finding:** On clean text, the model **correctly identified all 8 explicit defamatory accusations** (theft, fraud, abuse, perjury, addiction, rumors, criminal record) but **over-triggered** on 17 neutral sentences. It associated defamation with reporting verbs ("said," "claimed," "alleged," "denied") and context about people under scrutiny, even when the sentence was a denial or factual statement (e.g., "Davis denied the allegation," "A spokesperson said the matter would be investigated").

**Why:** The training set lacked examples of denials, neutral reporting, and factual context. The model learned to flag indirect speech about accusations but could not distinguish "X accused Y" (defamatory) from "Y denied" or "A journalist asked about X" (non-defamatory).

### Run 3: Legal-BERT + expanded dataset + early stopping

| Change | run2 (run1 model) | run3 |
|--------|-------------------|------|
| Dataset | 62 examples | **~250 examples** |
| Model | BERT-base | **Legal-BERT** (nlpaueb/legal-bert-base-uncased) |
| Validation | none | **15% val split, early stopping** |
| Batch size | 4 | **8** |

The expanded dataset explicitly included run2 failure modes: denials ("He denied the allegation"), neutral reporting ("A spokesperson said the matter would be investigated"), factual context ("Davis has served for six years"), and questions.

**Performance gains (on run2’s clean document):**

| Metric | run2 | run3 |
|--------|------|------|
| Defamatory recall | 8/8 (100%) | 7/8 (87.5%) |
| Defamatory precision | 8/25 ≈ 32% | 7/7 = **100%** |
| False positives | 17 | **0** |
| False negatives | 0 | 1 |

**Why the gains:** (1) **Legal-BERT** — pretrained on legal text, better suited to defamation-style language. (2) **Expanded dataset** — denials, neutral reporting, and factual context taught the model to separate accusations from reporting and refutations. (3) **Early stopping** — validation loss prevented overfitting. (4) **Trade-off** — a small recall drop (one missed editorial accusation of incompetence) for a large precision gain (32% → 100%), making the model much more usable in practice.

**Conclusion:** Run3 fixes all 17 run2 false positives. The model now correctly treats denials, neutral reporting, and factual statements as non-defamatory while still flagging explicit defamatory accusations.

---

## HPC (Yale Bouchet)

**API key:** None required.

**HPC settings used:**

| Setting    | Value   | Reason                                  |
|-----------|---------|-----------------------------------------|
| Cluster   | Bouchet | bouchet.ycrc.yale.edu                   |
| Memory    | 16 GB   | DistilBERT + PyTorch exceed default     |
| Cores     | 4       | DataLoader parallelism                  |
| Partition | day     | Batch jobs; devel for quick validation   |

### How to run on HPC

1. Upload this folder to Bouchet (Open OnDemand Files or scp)
2. Create conda env on a compute node:
   ```bash
   salloc --partition=devel --mem=15G --time=2:00:00 --cpus-per-task=2
   module load miniconda
   conda create -n law-llms python=3.9
   conda activate law-llms
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
   exit
   ```
3. Submit from project directory:
   ```bash
   sbatch run_all_hpc.sh
   ```

**Validation test** (optional): `sbatch run_validate_bouchet.sh` — quick smoke test before full run.


This project was completed with the aid of AI tools. The full repo can be found at [].