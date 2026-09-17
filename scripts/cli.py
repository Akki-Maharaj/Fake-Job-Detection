#!/usr/bin/env python
"""Command-line entry point: prepare data, train baseline/bert, predict.

Examples
--------
python scripts/cli.py prepare-data --config configs/default.yaml
python scripts/cli.py train-baseline --config configs/default.yaml
python scripts/cli.py train-bert --config configs/default.yaml
python scripts/cli.py predict --model-type baseline --model-dir models/baseline --text "..."
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from fake_job_detection.config import load_config
from fake_job_detection.data import build_processed, load_raw, stratified_splits, check_missingness_leakage
from fake_job_detection.models.baseline import train_baseline
from fake_job_detection.models.bert import train_bert
from fake_job_detection.predict import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_prepare_data(args):
    cfg = load_config(args.config)
    raw = load_raw(cfg.data.raw_csv)

    leakage = check_missingness_leakage(raw, cfg.data.label_column)
    logger.info("Top missingness/label correlations:\n%s", leakage.head(5).to_string())

    processed = build_processed(raw, cfg.data)
    processed.to_csv(cfg.data.processed_csv, index=False)
    logger.info("Wrote processed data to %s", cfg.data.processed_csv)


def cmd_train_baseline(args):
    cfg = load_config(args.config)
    df = pd.read_csv(cfg.data.processed_csv)
    train_df, val_df, _ = stratified_splits(df, cfg.data)
    metrics = train_baseline(train_df, val_df, cfg.baseline)
    logger.info("Baseline metrics: %s", metrics)


def cmd_train_bert(args):
    cfg = load_config(args.config)
    df = pd.read_csv(cfg.data.processed_csv)
    train_df, val_df, _ = stratified_splits(df, cfg.data)
    metrics = train_bert(train_df, val_df, cfg.bert, seed=cfg.seed)
    logger.info("BERT metrics: %s", metrics)


def cmd_predict(args):
    result = predict(args.text, args.model_type, args.model_dir)
    print(f"Prediction: {result['label'].upper()} (fake probability: {result['fake_probability']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Fake Job Detection CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare-data")
    p_prep.add_argument("--config", default="configs/default.yaml")
    p_prep.set_defaults(func=cmd_prepare_data)

    p_base = sub.add_parser("train-baseline")
    p_base.add_argument("--config", default="configs/default.yaml")
    p_base.set_defaults(func=cmd_train_baseline)

    p_bert = sub.add_parser("train-bert")
    p_bert.add_argument("--config", default="configs/default.yaml")
    p_bert.set_defaults(func=cmd_train_bert)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--text", required=True)
    p_pred.add_argument("--model-type", choices=["baseline", "bert"], default="baseline")
    p_pred.add_argument("--model-dir", default="models/baseline")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
