"""Decision-curve analysis (Vickers & Elkin, 2006).

Net benefit at threshold probability p_t weighs false positives by the odds p_t/(1-p_t):
    NB(p_t) = TP/n - FP/n * p_t / (1 - p_t)
A model is clinically useful at p_t when its net benefit exceeds both "treat all" and
"treat none" (NB = 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_benefit(y, p, thresholds) -> np.ndarray:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.asarray(out, float)


def treat_all(y, thresholds) -> np.ndarray:
    prev = float(np.mean(y))
    t = np.asarray(thresholds, float)
    return prev - (1 - prev) * t / (1 - t)


def decision_curve(y, probs: dict[str, np.ndarray], thresholds=None) -> pd.DataFrame:
    """Long table of net benefit for each model plus the treat-all / treat-none strategies."""
    thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2) if thresholds is None else np.asarray(thresholds)
    frames = [
        pd.DataFrame(
            {"threshold": thresholds, "strategy": "treat all", "net_benefit": treat_all(y, thresholds)}
        ),
        pd.DataFrame({"threshold": thresholds, "strategy": "treat none", "net_benefit": 0.0}),
    ]
    for name, p in probs.items():
        frames.append(
            pd.DataFrame(
                {"threshold": thresholds, "strategy": name, "net_benefit": net_benefit(y, p, thresholds)}
            )
        )
    return pd.concat(frames, ignore_index=True)
