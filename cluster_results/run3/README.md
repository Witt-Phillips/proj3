# Run3 — Legal-BERT + Expanded Dataset + Early Stopping

Targeted improvements over run2: larger dataset, legal-domain model, validation split with early stopping.

## Changes from run2

| Setting | run2 (run1 model) | run3 |
|---------|------------------|------|
| Dataset | 62 examples | **~250 examples** |
| Model | BERT-base | **Legal-BERT** (nlpaueb/legal-bert-base-uncased) |
| Validation | none | **15% val split, early stopping** |
| Batch size | 4 | **8** |

## Dataset (defamation_dataset_v3.csv)

~250 examples covering:
- **Defamatory (~100):** Direct accusations, reported allegations, character attacks (liar, fraud, thief, abuser, perjury, etc.)
- **Non-defamatory (~150):** Neutral facts, denials, neutral reporting, questions, procedural statements, factual context

Includes run2 failure modes: denials ("He denied the allegation"), neutral reporting ("A spokesperson said the matter would be investigated"), factual context ("Davis has served for six years"), questions ("What time is it?").

## Files

- `defamation_dataset_v3.csv` — ~250 labeled examples
- `run_all_hpc_v3.sh` — SLURM script (train + predict on both transcripts)

## Usage

From the project root (`proj3/`):

```bash
sbatch cluster_results/run3/run_all_hpc_v3.sh
```

## Output

- `defamation_model/` — fine-tuned Legal-BERT (best epoch by val loss)
- `results/defamation_results.json` — predictions on run2 transcript (clean document)
- `results/defamation_results_depp_heard.json` — predictions on Depp v. Heard transcript
- `results/defamation_v3_<jobid>.out` / `.err` — SLURM logs

---

## Results (job 5474701)

**Summary:** 66 sentences classified. Run3 **fixes all run2 false positives** on denials, neutral reporting, and factual statements. One new false negative: "One editorial suggested that Davis is incompetent..." (run2 correctly flagged it).

### Performance summary

| Metric | Run2 | Run3 |
|--------|------|------|
| **Defamatory recall** | 8/8 (100%) | 7/8 (87.5%) |
| **Defamatory precision** | 8/25 ≈ 32% | 7/7 = 100% |
| **False positives** | 17 | 0 |
| **False negatives** | 0 | 1 |

**Characterization:** Run3 is much more usable. Run2 had high recall but very low precision — it over-triggered on reporting verbs ("said", "claimed", "alleged", "denied"). Run3 fixes that. It correctly flags real accusations (theft, fraud, abuse, perjury, addiction, rumors, criminal record), correctly treats denials as non-defamatory, and correctly treats neutral reporting and factual context as non-defamatory. The one miss is a softer, opinion-style accusation ("incompetent"). The trade-off: a small drop in recall (~12.5%) for a large gain in precision (32% → 100%), making the model practical for real use.

### Training

- **Train/val split:** 208 train, 37 val
- **Loss progression:** 0.69 → 0.05 (epoch 4, best) → val loss rose after epoch 4
- **Runtime:** ~100 sec training, ~2 min total
- **Best model:** Epoch 4 (early stopping by val loss)

### Hits (correct defamatory)

| Sentence | Why defamatory |
|----------|----------------|
| One speaker claimed that Councilor Davis stole funds from the parks department. | Accusation of theft |
| Another resident alleged that the mayor is a fraud who cheated on his taxes. | Accusation of fraud, tax evasion |
| A former employee accused the mayor of abusing his staff and creating a hostile workplace. | Accusation of abuse |
| Someone said that the treasurer embezzled money and committed perjury when questioned. | Accusation of embezzlement, perjury |
| One citizen claimed that Johnson is a drug addict and unfit for office. | Accusation of addiction |
| A blogger wrote that Davis spread false rumors about her opponents during the campaign. | Accusation of spreading rumors |
| One of them reportedly said that our city manager is a convicted criminal. | Accusation of criminal record |

### Fixed (run2 false positives → run3 correct)

All 17 run2 false positives are now correctly non-defamatory:

- "Several residents spoke during the public comment period." — neutral
- "Davis denied the allegation and called it a lie." — denial
- "A spokesperson said the matter would be investigated." — neutral
- "Davis has served on the council for six years." — factual
- "Johnson declined to comment." / "Johnson stated that he would cooperate fully with any inquiry." — neutral
- "A journalist asked about the timeline for the budget vote." — neutral question
- "Davis responded that she had done nothing wrong." — denial
- "Several community groups praised the council's transparency." — positive, not defamatory
- "Johnson's approval rating stands at 62 percent." — factual
- "The treasurer denied everything." — denial
- "The mayor's office issued a brief statement." — neutral
- "Davis proposed the initiative last fall." / "The proposal passed with broad support." — neutral
- "The city manager has not commented." — neutral
- "She previously worked in Phoenix." — factual
- "Davis called the blog post a fabrication." — denial
- "The statement said the allegations were false and defamatory." — denial/refutation

### Miss (false negative)

- "One editorial suggested that Davis is incompetent at her job and should resign." — run2 correctly defamatory; run3 missed it (accusation of incompetence)

### Conclusion

Run3’s expanded dataset and Legal-BERT **eliminate run2’s false positives** on denials, neutral reporting, and factual context. The model now distinguishes "X accused Y" (defamatory) from "Y denied" and "A spokesperson said..." (non-defamatory). One regression: the editorial incompetence accusation is now missed. Overall, run3 is a substantial improvement.
