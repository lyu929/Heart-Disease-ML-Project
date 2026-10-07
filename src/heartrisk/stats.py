"""Statistics for comparing models evaluated with repeated cross-validation.

Fold scores from (repeated) K-fold CV are not independent because training sets overlap,
so a plain paired t-test is anti-conservative. We use the corrected resampled t-test of
Nadeau & Bengio (2003), variance inflated by (1/J + n_test/n_train), as recommended by
Bouckaert & Frank (2004) for repeated K-fold CV.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def corrected_ttest(diffs, n_train: int, n_test: int) -> tuple[float, float]:
    """Two-sided corrected resampled t-test on paired per-fold differences. Returns (t, p)."""
    d = np.asarray(diffs, float)
    j = len(d)
    var = d.var(ddof=1)
    if j < 2 or var == 0:
        return (float("inf") if d.mean() else 0.0), (0.0 if d.mean() else 1.0)
    t = d.mean() / np.sqrt((1.0 / j + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return float(t), float(p)


def corrected_ci(values, n_train: int, n_test: int, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the mean CV score using the corrected variance."""
    v = np.asarray(values, float)
    j = len(v)
    if j < 2:
        return float("nan"), float("nan")
    se = np.sqrt((1.0 / j + n_test / n_train) * v.var(ddof=1))
    h = stats.t.ppf(0.5 + level / 2, df=j - 1) * se
    return float(v.mean() - h), float(v.mean() + h)


def holm(pvalues) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(
    y,
    p,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    stratified: bool = True,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``metric(y, p)``; stratified resampling keeps both classes."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    point = float(metric(y, p))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        else:
            idx = rng.integers(0, len(y), len(y))
        vals[b] = metric(y[idx], p[idx])
    a = (1 - level) / 2
    lo, hi = np.nanquantile(vals, [a, 1 - a])
    return point, float(lo), float(hi)
