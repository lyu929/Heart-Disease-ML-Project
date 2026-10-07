"""Decision-threshold selection.

The legacy code picked the threshold that maximised F1 *on the test fold* and then
reported test-fold metrics at that threshold, which is optimistic (the test labels are
used for fitting). Here the threshold is always chosen on out-of-fold predictions of
the training data and then frozen before the test fold is scored.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_predict

STRATEGIES = ("f1", "youden", "sensitivity", "fixed")


def select_threshold(
    y, p, strategy: str = "f1", target_sensitivity: float = 0.9, fixed: float = 0.5
) -> float:
    """Return t such that ``p >= t`` is called positive."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if strategy == "fixed":
        return float(fixed)
    if len(np.unique(y)) < 2:
        return float(fixed)
    if strategy == "f1":
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)[:-1]
        return float(thr[int(np.argmax(f1))])
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    thr, fpr, tpr = thr[1:], fpr[1:], tpr[1:]  # drop the +inf sentinel
    if strategy == "youden":
        return float(thr[int(np.argmax(tpr - fpr))])
    if strategy == "sensitivity":
        ok = np.where(tpr >= target_sensitivity)[0]
        return float(thr[ok[0]]) if len(ok) else float(thr[-1])
    raise ValueError(f"unknown threshold strategy {strategy!r}; choose from {STRATEGIES}")


def oof_threshold(estimator, X, y, cv, **kwargs) -> tuple[float, np.ndarray]:
    """Threshold from out-of-fold training predictions (no test data involved)."""
    oof = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")[:, 1]
    return select_threshold(y, oof, **kwargs), oof
