"""Data loading, text combination, and stratified splitting.

This mirrors the logic from the original notebooks (01_EDA, 03_Training) but
as testable functions instead of notebook cells, and adds an explicit
leakage check that the original project never ran.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from fake_job_detection.config import DataConfig

logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def combine_text(row: pd.Series, text_columns: list[str]) -> str:
    """Concatenate text fields into a single string, separated by '.'.

    Empty/NaN fields are dropped rather than contributing empty segments,
    so we don't train the model to treat "missing field" as a token pattern
    by itself (that would be a cheap shortcut, not fraud-language signal).
    """
    parts = [str(row[c]) for c in text_columns if pd.notna(row[c]) and str(row[c]).strip()]
    return " . ".join(parts)


def build_processed(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    df = df.copy()
    for col in cfg.text_columns:
        if col not in df.columns:
            raise KeyError(f"Expected text column '{col}' not found in dataframe")
        df[col] = df[col].fillna("")

    df["combined_text"] = df.apply(lambda r: combine_text(r, cfg.text_columns), axis=1)
    return df[["combined_text", cfg.label_column]].rename(columns={cfg.label_column: "label"})


def stratified_splits(
    df: pd.DataFrame, cfg: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test split preserving the class ratio in every split.

    With only a few hundred fake examples, a naive random split risks a
    validation or test set with almost no positive examples at all.
    """
    train_df, temp_df = train_test_split(
        df, test_size=cfg.test_size, stratify=df["label"], random_state=cfg.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=cfg.val_size, stratify=temp_df["label"], random_state=cfg.seed
    )
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(
            "%s: %d rows, %.1f%% fake", name, len(split), split["label"].mean() * 100
        )
    return train_df, val_df, test_df


def class_weights(train_df: pd.DataFrame) -> tuple[float, float]:
    """Inverse-frequency class weights: total / (n_classes * class_count)."""
    n_total = len(train_df)
    n_real = int((train_df["label"] == 0).sum())
    n_fake = int((train_df["label"] == 1).sum())
    weight_real = n_total / (2 * n_real)
    weight_fake = n_total / (2 * n_fake)
    return weight_real, weight_fake


def check_missingness_leakage(raw_df: pd.DataFrame, label_col: str = "fraudulent") -> pd.DataFrame:
    """Check whether missing values in any column correlate strongly with the label.

    The original project never checked this. If, say, `salary_range` is missing
    far more often in fake postings, a model can learn "field is absent" as a
    shortcut instead of learning actual fraud language -- inflating offline
    metrics while teaching nothing that generalizes to fakes which do fill in
    that field.
    """
    rows = []
    for col in raw_df.columns:
        if col == label_col:
            continue
        is_missing = raw_df[col].isna()
        if is_missing.sum() == 0:
            continue
        miss_rate_fake = is_missing[raw_df[label_col] == 1].mean()
        miss_rate_real = is_missing[raw_df[label_col] == 0].mean()
        rows.append(
            {
                "column": col,
                "missing_rate_fake": miss_rate_fake,
                "missing_rate_real": miss_rate_real,
                "gap": miss_rate_fake - miss_rate_real,
            }
        )
    return pd.DataFrame(rows).sort_values("gap", key=abs, ascending=False)
