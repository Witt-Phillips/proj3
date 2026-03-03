#!/usr/bin/env python3
"""
Part 2: Fine-Tuned Defamation Detector
Law and LLMs - Project 3

Fine-tunes a BERT-style model on a labeled dataset of defamatory/non-defamatory
statements, then applies it to classify each sentence in the video transcript.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _ensure_nltk():
    """Ensure NLTK punkt tokenizer is available."""
    import nltk
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK (or regex fallback)."""
    try:
        _ensure_nltk()
        import nltk
        sents = nltk.sent_tokenize(text.strip())
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        import re
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sents if s.strip()]


def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load defamation dataset from CSV. Returns (texts, labels)."""
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(int(row["label"]))
    return texts, labels


def train(
    dataset_path: Path,
    model_name: str = "distilbert-base-uncased",
    output_dir: Path = Path("defamation_model"),
    epochs: int = 4,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> None:
    """Fine-tune model on defamation dataset."""
    print(f"Loading dataset from {dataset_path}")
    texts, labels = load_dataset(dataset_path)

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    class DefamationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {
                k: v[idx] for k, v in self.encodings.items()
            }
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    eval_dataset = None
    if val_ratio > 0 and len(texts) >= 20:
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_ratio, stratify=labels, random_state=seed
        )
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        train_dataset = DefamationDataset(train_encodings, train_labels)
        eval_dataset = DefamationDataset(val_encodings, val_labels)
        print(f"Train/val split: {len(train_texts)} train, {len(val_texts)} val")
    else:
        train_dataset = DefamationDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        save_strategy="epoch" if eval_dataset else "no",
        eval_strategy="epoch" if eval_dataset else "no",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        report_to="none",  # No wandb; no API key needed
        logging_steps=2,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


def predict(
    model_dir: Path,
    transcript_path: Path,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """Load model, split transcript into sentences, predict defamation for each."""
    print(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    transcript = transcript_path.read_text(encoding="utf-8")
    sentences = split_sentences(transcript)
    print(f"Found {len(sentences)} sentences")

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for sent in sentences:
            if not sent:
                continue
            inputs = tokenizer(
                sent,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            pred = int(logits.argmax(dim=-1).item())
            results.append({
                "sentence": sent,
                "label": pred,
                "prediction": "defamatory" if pred == 1 else "non-defamatory",
            })

    if output_path:
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Part 2: Fine-tuned defamation detector"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune on dataset")
    train_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("defamation_dataset.csv"),
        help="Path to CSV with text,label columns",
    )
    train_parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="HuggingFace model name (default: distilbert-base-uncased)",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=Path("defamation_model"),
        help="Directory to save fine-tuned model",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    train_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    train_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation split ratio (default: 0). Use 0.15 for 15%% val, enables early stopping.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    predict_parser = subparsers.add_parser("predict", help="Run on transcript")
    predict_parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("defamation_model"),
        help="Path to fine-tuned model",
    )
    predict_parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("Depp v. Heard_transcription.txt"),
        help="Path to transcript file",
    )
    predict_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("defamation_results.json"),
        help="Output path for results JSON",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            dataset_path=args.dataset,
            model_name=args.model,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            val_ratio=getattr(args, "val_ratio", 0.0),
            seed=getattr(args, "seed", 42),
        )
    else:
        if not args.model_dir.exists():
            print(f"Error: Model not found at {args.model_dir}. Run 'train' first.", file=sys.stderr)
            sys.exit(1)
        if not args.transcript.exists():
            print(f"Error: Transcript not found at {args.transcript}", file=sys.stderr)
            sys.exit(1)
        results = predict(
            model_dir=args.model_dir,
            transcript_path=args.transcript,
            output_path=args.output,
        )
        print("\n--- Sample results ---")
        for r in results[:5]:
            s = r["sentence"]
            print(f"  [{r['label']}] {s[:60]}{'...' if len(s) > 60 else ''}")
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")


if __name__ == "__main__":
    main()
