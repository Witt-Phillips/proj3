# Run2 — Clean Document Test

Apply the run1 defamation model to a **well-formed document** with no Whisper transcription errors.

## Rationale

Run0 and run1 both used the Depp v. Heard transcript, which has:
- Whisper errors ("It's… Kidney!" for "Depp", "People are alive between people")
- Garbled sentences that confuse the model

Run2 uses `transcript_sample.txt` — a fictional news-style document (~66 sentences) with clear, grammatical text. This tests whether the model performs better when the input is clean.

## Input document

`transcript_sample.txt` — ~66 sentences mixing:
- **Neutral/factual:** "The city council met on Tuesday.", "She enjoys hiking."
- **Defamatory allegations:** "One speaker claimed that Councilor Davis stole funds.", "Another resident alleged that the mayor is a fraud who cheated on his taxes."

## Usage

From the project root (`proj3/`):

```bash
sbatch cluster_results/run2/run_predict.sh
```

**Prediction only** — no training. Uses the run1 model. Runs in ~1 min.

## Output

- `results/defamation_results.json` — per-sentence predictions
- `results/defamation_run2_<jobid>.out` / `.err` — SLURM logs

---

## Results (job 5473774)

**Summary:** 66 sentences classified. Model correctly flags explicit defamatory accusations but over-predicts defamation on neutral reporting and factual statements.

### Hits (correct defamatory)

| Sentence | Why defamatory |
|----------|----------------|
| One speaker claimed that Councilor Davis stole funds from the parks department. | Accusation of theft |
| Another resident alleged that the mayor is a fraud who cheated on his taxes. | Accusation of fraud, tax evasion |
| One editorial suggested that Davis is incompetent at her job and should resign. | Accusation of incompetence |
| A former employee accused the mayor of abusing his staff and creating a hostile workplace. | Accusation of abuse |
| Someone said that the treasurer embezzled money and committed perjury when questioned. | Accusation of embezzlement, perjury |
| One citizen claimed that Johnson is a drug addict and unfit for office. | Accusation of addiction |
| A blogger wrote that Davis spread false rumors about her opponents during the campaign. | Accusation of spreading rumors |
| One of them reportedly said that our city manager is a convicted criminal. | Accusation of criminal record |

### Misses (false positives — neutral flagged as defamatory)

The model over-triggers on **reporting verbs** and **context**:

- "Several residents spoke during the public comment period." — neutral
- "Davis denied the allegation and called it a lie." — denial, not accusation
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
- "She previously worked in Phoenix." — factual (odd trigger)

### Pattern

The model appears to associate defamation with:
1. **Indirect speech** — "said", "claimed", "alleged", "stated", "responded", "denied"
2. **Sentences about people under scrutiny** — even factual ones get flagged
3. **Reporting context** — coverage of allegations gets treated like the allegation itself

### Conclusion

On clean text, the model **correctly identifies explicit defamatory content** (theft, fraud, abuse, perjury, etc.) but **over-applies** to neutral reporting, denials, and factual statements. It struggles to distinguish "X accused Y of Z" (defamatory if false) from "Y denied the accusation" (not defamatory) and "A journalist asked about X" (neutral). Better performance would require more training examples that include reported speech, denials, and factual context.
