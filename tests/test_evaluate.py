import numpy as np

from fake_job_detection.evaluate import compute_metrics


def test_perfect_predictions_give_perfect_scores():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.9, 0.8]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["false_negatives"] == 0
    assert metrics["false_positives"] == 0


def test_confusion_counts_correct():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]  # one FP, one FN
    y_prob = [0.1, 0.6, 0.4, 0.9]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["true_negatives"] == 1


def test_handles_single_class_gracefully():
    # AUC/AP undefined with one class; should not raise, just omit them
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    y_prob = [0.1, 0.2, 0.3]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "auc_roc" not in metrics
    assert metrics["accuracy"] == 1.0
