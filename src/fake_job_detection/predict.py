"""Run inference on a single raw job posting.

The original repo trains and evaluates but has no script that takes a new
job posting and returns a verdict -- this is that script.
"""
from __future__ import annotations

import argparse

from fake_job_detection.models.baseline import predict_baseline
from fake_job_detection.models.bert import predict_bert


def predict(text: str, model_type: str, model_dir: str) -> dict:
    if model_type == "baseline":
        return predict_baseline(text, model_dir)
    elif model_type == "bert":
        return predict_bert(text, model_dir)
    raise ValueError(f"Unknown model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a job posting as real or fake.")
    parser.add_argument("--text", required=True, help="Raw job posting text (combine title/description/etc. yourself, or pass description alone).")
    parser.add_argument("--model-type", choices=["baseline", "bert"], default="baseline")
    parser.add_argument("--model-dir", default="models/baseline")
    args = parser.parse_args()

    result = predict(args.text, args.model_type, args.model_dir)
    print(f"Prediction: {result['label'].upper()}  (fake probability: {result['fake_probability']:.3f})")


if __name__ == "__main__":
    main()
