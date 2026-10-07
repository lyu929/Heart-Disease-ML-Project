"""Probability calibration: reliability tables and recalibration wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

METHODS = ("none", "sigmoid", "isotonic")


def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Observed event rate vs mean predicted risk per bin (with Wilson 95% intervals)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    z = 1.959964
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        obs = y[m].mean()
        denom = 1 + z**2 / n
        centre = (obs + z**2 / (2 * n)) / denom
        half = z * np.sqrt(obs * (1 - obs) / n + z**2 / (4 * n**2)) / denom
        rows.append(
            {
                "bin": b,
                "n": n,
                "mean_predicted": p[m].mean(),
                "observed": obs,
                "lo": min(obs, max(0.0, centre - half)),
                "hi": max(obs, min(1.0, centre + half)),
            }
        )
    return pd.DataFrame(rows)


def calibrate(estimator, method: str = "sigmoid", n_splits: int = 5, seed: int = 0):
    """Wrap an (unfitted) estimator so calibration is learned with internal CV."""
    if method in (None, "none"):
        return estimator
    if method not in METHODS:
        raise ValueError(f"unknown calibration method {method!r}; choose from {METHODS}")
    return CalibratedClassifierCV(
        estimator, method=method, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    )
