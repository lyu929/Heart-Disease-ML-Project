import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from evaluate import compute_metrics
from preprocess import build_preprocessor, compute_class_weight_scale, split_features_target


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "heart": ("data/heart.csv", "HeartDisease"),
    "kaggle_processed": ("data/kaggle_processed.csv", "target"),
    "cleveland_processed": ("data/cleveland_processed.csv", "target"),
    "framingham_processed": ("data/framingham_processed.csv", "target"),
}


def build_models(y):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=compute_class_weight_scale(y),
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    return models


def evaluate_model_cv(X, y, estimator, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all = []
    y_prob_all = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        preprocessor, _, _ = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(estimator)),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test.to_numpy())
        y_prob_all.extend(y_prob)

    return compute_metrics(np.array(y_true_all), np.array(y_prob_all), threshold=0.5)


def run_dataset_comparison():
    rows = []

    for dataset_name, (relative_path, target_col) in DATASETS.items():
        df = pd.read_csv(PROJECT_DIR / relative_path)
        X, y = split_features_target(df, target_col)

        for model_name, estimator in build_models(y).items():
            print(f"Dataset comparison: {dataset_name} - {model_name}")
            metrics = evaluate_model_cv(X, y, estimator)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Rows": len(df),
                    "Features": X.shape[1],
                    "PositiveCases": int(y.sum()),
                    "NegativeCases": int((y == 0).sum()),
                    "Model": model_name,
                    **metrics,
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUTS_DIR / "dataset_comparison_results.csv", index=False)
    return result_df


def run_random_forest_parameter_comparison():
    df = pd.read_csv(PROJECT_DIR / "data/heart.csv")
    X, y = split_features_target(df, "HeartDisease")
    rows = []

    n_estimators_values = [50, 100, 200]
    max_depth_values = [5, 10, None]

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            print(
                "RF parameter comparison: "
                f"n_estimators={n_estimators}, max_depth={max_depth}"
            )
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
            metrics = evaluate_model_cv(X, y, estimator)
            rows.append(
                {
                    "Dataset": "heart",
                    "Model": "Random Forest",
                    "n_estimators": n_estimators,
                    "max_depth": "None" if max_depth is None else max_depth,
                    **metrics,
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUTS_DIR / "rf_parameter_comparison.csv", index=False)
    return result_df


def main():
    dataset_results = run_dataset_comparison()
    parameter_results = run_random_forest_parameter_comparison()

    print("\nSaved: outputs/dataset_comparison_results.csv")
    print("Saved: outputs/rf_parameter_comparison.csv")

    print("\nDataset comparison summary:")
    print(
        dataset_results[
            ["Dataset", "Model", "Accuracy", "Recall", "F1", "ROC_AUC", "MSE"]
        ].round(4).to_string(index=False)
    )

    print("\nRandom Forest parameter comparison:")
    print(
        parameter_results[
            ["n_estimators", "max_depth", "Accuracy", "Recall", "F1", "ROC_AUC", "MSE"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
