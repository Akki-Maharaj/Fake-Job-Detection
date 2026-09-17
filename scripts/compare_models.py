#!/usr/bin/env python
"""Compare the TF-IDF baseline against DistilBERT on the same test set.

This is the script the original project never had: it puts both models'
metrics side by side so the transformer's cost is justified by a measured
gain, not assumed.

Usage
-----
    python scripts/compare_models.py --config configs/default.yaml

Assumes both `train-baseline` and `train-bert` have already been run (so
models/baseline/ and models/bert/ both exist). Evaluates both on the same
held-out test split.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from fake_job_detection.config import load_config
from fake_job_detection.data import stratified_splits
from fake_job_detection.evaluate import compute_metrics
from fake_job_detection.models.baseline import load_baseline


def eval_baseline(test_df: pd.DataFrame, model_dir: str) -> dict:
    vectorizer, clf = load_baseline(model_dir)
    X = vectorizer.transform(test_df["combined_text"])
    y = test_df["label"].values
    probs = clf.predict_proba(X)[:, 1]
    preds = clf.predict(X)
    return compute_metrics(y, preds, probs)


def eval_bert(test_df: pd.DataFrame, model_dir: str, max_len: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model.eval()

    all_preds, all_probs = [], []
    batch_size = 32
    texts = test_df["combined_text"].astype(str).tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return compute_metrics(test_df["label"].values, all_preds, all_probs)


def print_table(baseline_metrics: dict, bert_metrics: dict) -> None:
    keys = [
        "accuracy", "f1_macro", "precision_macro", "recall_macro",
        "fake_precision", "fake_recall", "fake_f1", "auc_roc", "avg_precision",
    ]
    header = f"{'Metric':<18}{'Baseline (TF-IDF)':<20}{'DistilBERT':<15}{'Delta':<10}"
    print(header)
    print("-" * len(header))
    for k in keys:
        b = baseline_metrics.get(k)
        t = bert_metrics.get(k)
        if b is None or t is None:
            continue
        delta = t - b
        sign = "+" if delta >= 0 else ""
        print(f"{k:<18}{b:<20.4f}{t:<15.4f}{sign}{delta:.4f}")

    print()
    print("Confusion matrix (false negatives = fake jobs missed, most costly error):")
    for name, m in [("Baseline", baseline_metrics), ("DistilBERT", bert_metrics)]:
        print(
            f"  {name:<12} FN={m['false_negatives']:<4} FP={m['false_positives']:<4} "
            f"TP={m['true_positives']:<4} TN={m['true_negatives']:<4}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="Optional path to write comparison JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg.data.processed_csv)
    _, _, test_df = stratified_splits(df, cfg.data)

    print(f"Evaluating on {len(test_df)} held-out test examples "
          f"({test_df['label'].mean()*100:.1f}% fake)\n")

    baseline_metrics = eval_baseline(test_df, cfg.baseline.output_dir)
    bert_metrics = eval_bert(test_df, cfg.bert.output_dir, cfg.bert.max_len)

    print_table(baseline_metrics, bert_metrics)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"baseline": baseline_metrics, "bert": bert_metrics}, f, indent=2)
        print(f"\nWrote comparison to {args.out}")


if __name__ == "__main__":
    main()
