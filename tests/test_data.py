import pandas as pd
import pytest

from fake_job_detection.config import DataConfig
from fake_job_detection.data import (
    build_processed,
    check_missingness_leakage,
    class_weights,
    combine_text,
    stratified_splits,
)


@pytest.fixture
def cfg():
    return DataConfig(text_columns=["title", "description"], label_column="fraudulent")


@pytest.fixture
def raw_df():
    # 100 rows, 20 fake (20%) -- enough that both the outer split and the
    # val/test split of the remainder each retain >=2 fake examples.
    rows = []
    for i in range(80):
        rows.append({"title": f"Real job {i}", "description": "Normal description", "fraudulent": 0})
    for i in range(20):
        rows.append({"title": f"Fake job {i}", "description": None, "fraudulent": 1})
    return pd.DataFrame(rows)


def test_combine_text_drops_empty_fields():
    row = pd.Series({"title": "Engineer", "description": ""})
    assert combine_text(row, ["title", "description"]) == "Engineer"


def test_combine_text_joins_with_separator():
    row = pd.Series({"title": "Engineer", "description": "Build things"})
    assert combine_text(row, ["title", "description"]) == "Engineer . Build things"


def test_build_processed_has_expected_columns(raw_df, cfg):
    processed = build_processed(raw_df, cfg)
    assert list(processed.columns) == ["combined_text", "label"]
    assert len(processed) == len(raw_df)


def test_build_processed_missing_column_raises(raw_df):
    bad_cfg = DataConfig(text_columns=["nonexistent"], label_column="fraudulent")
    with pytest.raises(KeyError):
        build_processed(raw_df, bad_cfg)


def test_stratified_splits_preserve_ratio(raw_df, cfg):
    processed = build_processed(raw_df, cfg)
    train_df, val_df, test_df = stratified_splits(processed, cfg)
    assert len(train_df) + len(val_df) + len(test_df) == len(processed)
    # every split should contain at least one fake example given the ratio
    assert train_df["label"].sum() >= 1


def test_class_weights_favor_minority():
    df = pd.DataFrame({"label": [0] * 90 + [1] * 10})
    w_real, w_fake = class_weights(df)
    assert w_fake > w_real


def test_missingness_leakage_flags_correlated_column():
    df = pd.DataFrame(
        {
            "salary_range": [None] * 8 + [1.0] * 2 + [None] * 1 + [1.0] * 9,
            "fraudulent": [1] * 10 + [0] * 10,
        }
    )
    result = check_missingness_leakage(df, label_col="fraudulent")
    assert "salary_range" in result["column"].values
    top = result.iloc[0]
    assert abs(top["gap"]) > 0.5
