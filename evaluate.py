import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += np.abs(acc - conf) * np.mean(mask)

    return float(ece)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else np.nan,
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else np.nan,
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(expected_calibration_error(y_true, y_prob)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["Sensitivity"] = metrics["Recall"]
    metrics["Threshold"] = float(threshold)
    return metrics


def find_best_threshold(y_true, y_prob, optimize_for="F1"):
    best_threshold = 0.5
    best_score = -1

    for threshold in np.linspace(0.1, 0.9, 17):
        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        score = metrics.get(optimize_for, metrics["F1"])
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)


def compute_degradation(within_score, transfer_score):
    if within_score == 0:
        return np.nan
    return float((within_score - transfer_score) / within_score * 100.0)


def paired_test(scores_a, scores_b):
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return np.nan
    _, p_value = ttest_rel(scores_a, scores_b)
    return float(p_value)