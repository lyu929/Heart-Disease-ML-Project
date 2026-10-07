"""Discrimination, threshold and calibration metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Equal-width ECE: sum_b |B|/n * |mean(y_b) - mean(p_b)|."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _newton_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 50) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta + offset, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        grad = X.T @ (y - mu)
        hess = X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope.

    Perfect calibration: intercept 0, slope 1. Slope < 1 means predictions are too extreme
    (over-fitting); intercept > 0 means risks are systematically under-estimated.
    """
    y = np.asarray(y, float)
    lp = _logit(p)
    ones = np.ones((len(y), 1))
    intercept = float(_newton_logistic(ones, y, lp)[0])
    slope = float(_newton_logistic(np.column_stack([ones, lp]), y, np.zeros(len(y)))[1])
    return intercept, slope


def confusion_counts(y, p, threshold: float) -> tuple[int, int, int, int]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def threshold_metrics(y, p, threshold: float) -> dict[str, float]:
    tp, fp, tn, fn = confusion_counts(y, p, threshold)
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    return {
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": prec,
        "npv": _div(tn, tn + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
    }


def compute_metrics(y, p, threshold: float = 0.5) -> dict[str, float]:
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 0, 1)
    out: dict[str, float] = {}
    both = len(np.unique(y)) == 2
    out["roc_auc"] = float(roc_auc_score(y, p)) if both else float("nan")
    out["pr_auc"] = float(average_precision_score(y, p)) if both else float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))
    out["ece"] = expected_calibration_error(y, p)
    if both:
        out["cal_intercept"], out["cal_slope"] = calibration_intercept_slope(y, p)
    else:
        out["cal_intercept"] = out["cal_slope"] = float("nan")
    out["threshold"] = float(threshold)
    out.update(threshold_metrics(y, p, threshold))
    return out


# Metrics where larger is better (used for ranking / reporting direction)
HIGHER_IS_BETTER = {
    "roc_auc": True,
    "pr_auc": True,
    "brier": False,
    "log_loss": False,
    "ece": False,
    "accuracy": True,
    "balanced_accuracy": True,
    "sensitivity": True,
    "specificity": True,
    "ppv": True,
    "npv": True,
    "f1": True,
}
