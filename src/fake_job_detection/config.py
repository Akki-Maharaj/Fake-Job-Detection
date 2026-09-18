"""Typed configuration loaded from YAML files.

Keeping hyperparameters here (instead of buried in notebook cells) means every
run is reproducible from a single file, and diffs in a PR actually show what
changed about an experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    raw_csv: str = "data/fake_job_postings.csv"
    processed_csv: str = "data/processed.csv"
    text_columns: list[str] = field(
        default_factory=lambda: [
            "title",
            "company_profile",
            "description",
            "requirements",
            "benefits",
        ]
    )
    label_column: str = "fraudulent"
    test_size: float = 0.2
    val_size: float = 0.5  # fraction of the held-out split used for validation
    seed: int = 42


@dataclass
class BaselineConfig:
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    model: str = "logreg"  # "logreg" or "xgboost"
    class_weight: str = "balanced"
    output_dir: str = "models/baseline"


@dataclass
class BertConfig:
    pretrained_name: str = "distilbert-base-uncased"
    max_len: int = 256
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "models/bert"
    tensorboard_dir: str = "tensorboard_logs"
    checkpoint_dir: str = "checkpoints/bert"
    checkpoint_every_n_steps: int = 100


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    bert: BertConfig = field(default_factory=BertConfig)
    seed: int = 42


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file, falling back to defaults for anything unset."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    if "baseline" in raw:
        cfg.baseline = BaselineConfig(**raw["baseline"])
    if "bert" in raw:
        cfg.bert = BertConfig(**raw["bert"])
    if "seed" in raw:
        cfg.seed = raw["seed"]
    return cfg
