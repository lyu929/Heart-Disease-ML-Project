import numpy as np
import pandas as pd
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df, target_col="HeartDisease"):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()
    return X, y


def build_preprocessor(X):
    """
    Build a preprocessing pipeline for mixed-type data.
    Numeric columns -> median imputation + scaling
    Categorical columns -> most_frequent imputation + one-hot encoding
    """

    # 明确按 pandas dtype 分列
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 双重保险：如果某列虽然没被识别成 object，但里面其实是字符串，也归到 categorical
    for col in X.columns:
        if col not in categorical_cols and col not in numeric_cols:
            categorical_cols.append(col)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop"
    )

    return preprocessor, numeric_cols, categorical_cols


def compute_class_weight_scale(y):
    counts = Counter(y)
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)
    if pos == 0:
        return 1.0
    return neg / pos


def safe_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)