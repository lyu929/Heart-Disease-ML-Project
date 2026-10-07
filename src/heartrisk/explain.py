"""Model explanations on the *raw* clinical features (not one-hot columns).

* :func:`shapley_values` – exact interventional Shapley values for one patient. With
  d <= 12 features all 2^d coalitions are enumerated (heart.csv: 11 features, 2048
  coalitions), so the values satisfy efficiency exactly:
  ``sum(phi) == f(x) - mean_b f(b)`` over the reference sample b.
* :func:`permutation_importance_cv` – global importance as the drop in held-out ROC-AUC
  when a raw feature is permuted, averaged over CV folds.
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def _evaluate(predict_fn, x: np.ndarray, background: pd.DataFrame, masks: np.ndarray) -> np.ndarray:
    features = list(background.columns)
    bg = background.to_numpy(dtype=object)
    big = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :]).reshape(-1, len(features))
    frame = pd.DataFrame(big, columns=features)
    for col in features:
        frame[col] = frame[col].astype(background[col].dtype)
    return np.asarray(predict_fn(frame), float).reshape(len(masks), len(bg)).mean(axis=1)


def shapley_values(
    predict_fn,
    row: pd.DataFrame,
    background: pd.DataFrame,
    max_exact: int = 12,
    n_permutations: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Return ``(phi, base_value, prediction)`` for the single-row frame ``row``."""
    features = list(background.columns)
    d = len(features)
    x = row[features].to_numpy(dtype=object)[0]
    if d <= max_exact:
        masks = ((np.arange(2**d)[:, None] >> np.arange(d)) & 1).astype(bool)
        v = _evaluate(predict_fn, x, background, masks)
        size = masks.sum(axis=1)
        w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
        phi = np.zeros(d)
        for j in range(d):
            without = np.where(~masks[:, j])[0]
            phi[j] = np.sum(w[size[without]] * (v[without + (1 << j)] - v[without]))
        return phi, float(v[0]), float(v[-1])
    # Monte-Carlo permutation estimator for wide inputs, with antithetic (reversed) orders:
    # every sampled order is paired with its reverse, which halves the variance and makes
    # pairwise interactions exact.
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(max(1, n_permutations // 2)):
        o = rng.permutation(d)
        orders += [o, o[::-1]]
    n_permutations = len(orders)
    masks = []
    for order in orders:
        m = np.zeros(d, bool)
        masks.append(m.copy())
        for j in order:
            m[j] = True
            masks.append(m.copy())
    masks = np.asarray(masks)
    v = _evaluate(predict_fn, x, background, masks).reshape(n_permutations, d + 1)
    phi = np.zeros(d)
    for k in range(n_permutations):
        m = masks[k * (d + 1) : (k + 1) * (d + 1)]
        added = np.argmax(m[1:] & ~m[:-1], axis=1)
        phi[added] += np.diff(v[k])
    full = _evaluate(predict_fn, x, background, np.ones((1, d), bool))[0]
    base = _evaluate(predict_fn, x, background, np.zeros((1, d), bool))[0]
    return phi / n_permutations, float(base), float(full)


def explain_patient(bundle, row: pd.DataFrame) -> pd.DataFrame:
    """Shapley table for one prepared row, sorted by absolute contribution."""
    phi, base, pred = shapley_values(
        bundle.predict_proba, row[bundle.features], bundle.background[bundle.features]
    )
    out = pd.DataFrame(
        {"feature": bundle.features, "value": row[bundle.features].iloc[0].tolist(), "contribution": phi}
    )
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    out.attrs.update(base_value=base, probability=pred)
    return out


def permutation_importance_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Held-out ROC-AUC drop per raw feature, mean and SD over folds."""
    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)):
        est = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(
            est,
            X.iloc[te],
            y.iloc[te],
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=seed + fold,
            n_jobs=n_jobs,
        )
        rows += [{"fold": fold, "feature": f, "importance": m} for f, m in zip(X.columns, r.importances_mean)]
    df = pd.DataFrame(rows)
    return (
        df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
