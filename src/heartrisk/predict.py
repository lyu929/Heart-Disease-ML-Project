"""Batch / single-record prediction with input checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .data import clean, get_spec


def risk_band(p: float, bands: list) -> str:
    for upper, name in bands:
        if p < float(upper):
            return str(name)
    return str(bands[-1][1])


def prepare(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the training-time cleaning (0 -> missing for BP / cholesterol) and column order."""
    spec = get_spec(bundle.dataset)
    missing = [c for c in bundle.features if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing columns {missing}")
    zero_cols = (bundle.metadata.get("cleaning") or {}).get("zero_as_missing", [])
    tmp = df[bundle.features].copy()
    tmp[spec.target] = 0
    return clean(tmp, spec, zero_cols)[bundle.features]


def input_warnings(bundle: ModelBundle, row: pd.Series) -> list[str]:
    msgs = []
    for col, (lo, hi) in bundle.ranges.items():
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            msgs.append(f"{col} missing: imputed with the training median")
        elif not lo <= v <= hi:
            msgs.append(f"{col}={v:g} outside the training range [{lo:g}, {hi:g}] (extrapolation)")
    for col, levels in bundle.levels.items():
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            msgs.append(f"{col} missing: imputed with the most frequent level")
        elif str(v) not in levels:
            msgs.append(f"{col}={v!r} was never seen in training")
    return msgs


def predict_frame(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(bundle, df)
    p = bundle.predict_proba(X)
    out = pd.DataFrame(index=df.index)
    out["probability"] = p
    out["prediction"] = (p >= bundle.threshold).astype(int)
    out["threshold"] = bundle.threshold
    out["risk_band"] = [risk_band(v, bundle.risk_bands) for v in p]
    out["warnings"] = [input_warnings(bundle, X.loc[i]) for i in X.index]
    return out
