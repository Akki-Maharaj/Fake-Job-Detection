# Fake Job Posting Detection

A fraudulent job posting classifier: a fast TF-IDF + Logistic Regression
baseline, and a fine-tuned DistilBERT model, evaluated identically so the
gain from the transformer (if any) is measurable rather than assumed.

This is a rebuild of the [original notebook-based project](https://github.com/Akki-Maharaj/Fake-Job-Detection),
restructured as a testable Python package with:

- A **classic ML baseline** run before any transformer training, so a
  99%-accuracy headline number has something to be compared against.
- A **leakage check** on field missingness vs. label — the original never
  verified the model wasn't just learning "this field is blank" as a proxy
  for fraud.
- **Unit tests** for the data pipeline and metrics (there were none before).
- **Config-driven** runs (`configs/default.yaml`) instead of hyperparameters
  hardcoded in notebook cells.
- An actual **inference script/CLI** — the original trains and evaluates but
  has no code path that takes a new job posting and returns a verdict.

## Project layout

```
fjd/
├── src/fake_job_detection/
│   ├── config.py          # typed config, loaded from YAML
│   ├── data.py             # loading, text combination, stratified splits, leakage check
│   ├── evaluate.py         # shared metrics (used by both models)
│   ├── predict.py          # inference entry point
│   └── models/
│       ├── baseline.py     # TF-IDF + LogReg/XGBoost
│       └── bert.py         # DistilBERT dataset, training loop, inference
├── configs/default.yaml
├── scripts/cli.py          # `fjd prepare-data|train-baseline|train-bert|predict`
├── tests/                  # pytest
├── notebooks/              # exploration only — not the pipeline
└── pyproject.toml
```

## Setup

```bash
pip install -e ".[dev]"          # baseline + tests
pip install -e ".[dev,xgboost]"  # + xgboost baseline option
```

DistilBERT fine-tuning needs `torch` + `transformers`; install those
separately if you don't already have a CUDA-matched torch build:

```bash
pip install torch transformers
```

## Usage

```bash
# 1. Download fake_job_postings.csv from Kaggle into data/, then:
python scripts/cli.py prepare-data --config configs/default.yaml

# 2. Baseline first — this is the number the transformer has to beat
python scripts/cli.py train-baseline --config configs/default.yaml

# 3. DistilBERT
python scripts/cli.py train-bert --config configs/default.yaml

# 4. Predict on a new posting
python scripts/cli.py predict --model-type baseline --model-dir models/baseline \
  --text "Earn $5000/week from home, no experience needed, wire transfer registration fee required."
```

## Tests

```bash
pytest
```

## Design notes

- **Class weighting**: both models handle the ~95/5 imbalance — the baseline
  via `class_weight="balanced"`, BERT via weighted cross-entropy (same
  `total / (2 * class_count)` formula as the original).
- **Splits**: stratified train/val/test (80/10/10) preserves the fake-job
  ratio in every split — important with only a few hundred positive examples.
- **Model selection**: checkpoints are saved on validation macro F1, not
  accuracy, since accuracy is dominated by the majority class here.
- **Leakage check**: `check_missingness_leakage` flags any column whose
  missingness rate differs sharply between real and fake postings, so you
  can decide deliberately whether to keep, drop, or explicitly encode that
  signal rather than have the model exploit it implicitly.

## What to compare, and why it matters

Report baseline vs. DistilBERT side by side (accuracy, macro F1, fake
precision/recall, AUC-ROC) rather than only reporting the transformer's
numbers. On formulaic fraud language, a linear model over n-grams can land
surprisingly close to a fine-tuned transformer — if that holds here too,
it's a genuinely useful finding about the problem, not a downgrade.
