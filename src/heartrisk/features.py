"""Preprocessing. Every statistic (medians, scales, category levels) is learned inside the
pipeline, i.e. only from the training fold, and the same pipeline object is deployed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DatasetSpec


class ZeroAsMissing(TransformerMixin, BaseEstimator):
    """Replace physiologically impossible zeros (e.g. cholesterol = 0) with NaN. Stateless."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype(float).mask(X[col] == 0)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def preprocessing_steps(
    spec: DatasetSpec, cfg: dict | None = None, scale: bool = True
) -> list[tuple[str, object]]:
    """``[("zeros", ZeroAsMissing), ("columns", ColumnTransformer)]`` – flat so that the steps can
    also be spliced into an imbalanced-learn pipeline (which forbids nested pipelines)."""
    cleaning = (cfg or {}).get("cleaning", {}) or {}
    missing_indicator = bool(cleaning.get("missing_indicator", True))
    zero_cols = [c for c in cleaning.get("zero_as_missing", []) or [] if c in spec.features]

    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=missing_indicator))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(numeric_steps), list(spec.numeric))]
    if spec.binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), list(spec.binary)))
    if spec.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(spec.categorical),
            )
        )
    columns = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return [("zeros", ZeroAsMissing(tuple(zero_cols))), ("columns", columns)]


def build_preprocessor(spec: DatasetSpec, cfg: dict | None = None, scale: bool = True) -> Pipeline:
    return Pipeline(preprocessing_steps(spec, cfg, scale))
