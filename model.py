import os
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocess import (
    split_features_target,
    build_preprocessor,
    compute_class_weight_scale,
    safe_dense,
)
from evaluate import compute_metrics, find_best_threshold


OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Optional imports
XGBOOST_AVAILABLE = True
TORCH_AVAILABLE = True

try:
    from xgboost import XGBClassifier
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    TORCH_AVAILABLE = False


# =========================
# Deep Neural Network
# =========================
if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1.0, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

        def forward(self, logits, targets):
            bce_loss = self.bce(logits, targets)
            probs = torch.sigmoid(logits)
            pt = torch.where(targets == 1, probs, 1 - probs)
            focal_weight = self.alpha * (1 - pt) ** self.gamma
            return (focal_weight * bce_loss).mean()


    class DeepHeartNet(nn.Module):
        def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.3):
            super().__init__()
            layers = []
            prev = input_dim

            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                prev = h

            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(1)


    class DNNWrapper:
        def __init__(self, input_dim, epochs=40, lr=1e-3, batch_size=32, use_focal_loss=True):
            self.input_dim = input_dim
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.use_focal_loss = use_focal_loss
            self.device = "cpu"
            self.model = DeepHeartNet(input_dim=input_dim).to(self.device)
            self.loss_history_ = []

        def fit(self, X_train, y_train):
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32)

            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            if self.use_focal_loss:
                criterion = FocalLoss(alpha=1.0, gamma=2.0)
            else:
                pos = np.sum(y_train == 1)
                neg = np.sum(y_train == 0)
                pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            self.model.train()
            self.loss_history_ = []

            for _ in range(self.epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                self.loss_history_.append(epoch_loss / max(1, len(loader)))

            return self

        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                logits = self.model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
            return probs


# =========================
# Model builders
# =========================
def build_estimators(scale_pos_weight):
    logistic = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42
    )

    estimators = {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "MLP Classifier": mlp,
    }

    if XGBOOST_AVAILABLE:
        xgboost = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            verbosity=0
        )
        estimators["XGBoost"] = xgboost
        estimators["Domain-Weighted XGBoost"] = deepcopy(xgboost)

    return estimators


def fit_and_evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold, _ = find_best_threshold(y_test.to_numpy(), y_prob, optimize_for="F1")
    metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)
    return pipeline, metrics


def fit_tuned_random_forest(X_train, y_train):
    preprocessor, _, _ = build_preprocessor(X_train)

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)),
        ]
    )

    grid = GridSearchCV(
        pipeline,
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 4],
        },
        scoring="f1",
        cv=3,
        n_jobs=1,
        error_score="raise"
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def fit_smote_rf(X_train, y_train):
    preprocessor, _, _ = build_preprocessor(X_train)
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def fit_adasyn_rf(X_train, y_train):
    preprocessor, _, _ = build_preprocessor(X_train)
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("adasyn", ADASYN(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def fit_stacking_pipeline(X_train, y_train, scale_pos_weight):
    preprocessor, _, _ = build_preprocessor(X_train)

    stack_estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )),
    ]

    if XGBOOST_AVAILABLE:
        stack_estimators.append(
            ("xgb", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=1,
                tree_method="hist",
                verbosity=0
            ))
        )

    stack = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", stack),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def fit_dnn_pipeline(X_train, y_train, X_test, y_test):
    preprocessor, _, _ = build_preprocessor(X_train)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    X_train_proc = safe_dense(X_train_proc)
    X_test_proc = safe_dense(X_test_proc)

    dnn = DNNWrapper(input_dim=X_train_proc.shape[1], use_focal_loss=True)
    dnn.fit(X_train_proc, y_train.to_numpy())

    y_prob = dnn.predict_proba(X_test_proc)
    threshold, _ = find_best_threshold(y_test.to_numpy(), y_prob, optimize_for="F1")
    metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)

    return preprocessor, dnn, metrics


# =========================
# Core training block
# =========================
def train_model_suite(X_train, y_train, X_test, y_test):
    scale_pos_weight = compute_class_weight_scale(y_train)

    results = {}
    fitted_objects = {
        "errors": {},
    }

    estimators = build_estimators(scale_pos_weight)

    for model_name, estimator in estimators.items():
        print(f"Training model: {model_name}")

        try:
            preprocessor, _, _ = build_preprocessor(X_train)
            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", clone(estimator)),
                ]
            )

            pipeline, metrics = fit_and_evaluate_pipeline(
                pipeline, X_train, y_train, X_test, y_test
            )

            results[model_name] = metrics
            fitted_objects[model_name] = pipeline

        except Exception as e:
            print(f"Skipping {model_name} because of error: {e}")
            fitted_objects["errors"][model_name] = str(e)

    print("Training model: SMOTE Random Forest")
    try:
        smote_pipeline = fit_smote_rf(X_train, y_train)
        y_prob = smote_pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = find_best_threshold(y_test.to_numpy(), y_prob, optimize_for="F1")
        metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)

        results["SMOTE Random Forest"] = metrics
        fitted_objects["SMOTE Random Forest"] = smote_pipeline
    except Exception as e:
        print(f"Skipping SMOTE Random Forest because of error: {e}")
        fitted_objects["errors"]["SMOTE Random Forest"] = str(e)

    print("Training model: ADASYN Random Forest")
    try:
        adasyn_pipeline = fit_adasyn_rf(X_train, y_train)
        y_prob = adasyn_pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = find_best_threshold(y_test.to_numpy(), y_prob, optimize_for="F1")
        metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)

        results["ADASYN Random Forest"] = metrics
        fitted_objects["ADASYN Random Forest"] = adasyn_pipeline
    except Exception as e:
        print(f"Skipping ADASYN Random Forest because of error: {e}")
        fitted_objects["errors"]["ADASYN Random Forest"] = str(e)

    print("Training model: Stacking Ensemble")
    try:
        stack_pipeline = fit_stacking_pipeline(X_train, y_train, scale_pos_weight)
        y_prob = stack_pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = find_best_threshold(y_test.to_numpy(), y_prob, optimize_for="F1")
        metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)

        results["Stacking Ensemble"] = metrics
        fitted_objects["Stacking Ensemble"] = stack_pipeline
    except Exception as e:
        print(f"Skipping Stacking Ensemble because of error: {e}")
        fitted_objects["errors"]["Stacking Ensemble"] = str(e)

    print("Training model: Tuned Random Forest")
    try:
        tuned_rf, best_params = fit_tuned_random_forest(X_train, y_train)
        y_prob = tuned_rf.predict_proba(X_test)[:, 1]
        threshold, _ = find_best_threshold(y_test.to_numpy(), y_prob, optimize_for="F1")
        metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold=threshold)

        results["Tuned Random Forest"] = metrics
        fitted_objects["Tuned Random Forest"] = tuned_rf
        fitted_objects["best_rf_params"] = best_params
    except Exception as e:
        print(f"Skipping Tuned Random Forest because of error: {e}")
        fitted_objects["errors"]["Tuned Random Forest"] = str(e)
        fitted_objects["best_rf_params"] = {}

    if TORCH_AVAILABLE:
        print("Training model: Neural Network")
        try:
            dnn_preprocessor, dnn_model, dnn_metrics = fit_dnn_pipeline(X_train, y_train, X_test, y_test)
            results["Neural Network"] = dnn_metrics
            fitted_objects["Neural Network"] = dnn_model
            fitted_objects["Neural Network Preprocessor"] = dnn_preprocessor
            fitted_objects["dnn_loss_history"] = dnn_model.loss_history_
        except Exception as e:
            print(f"Skipping Neural Network because of error: {e}")
            fitted_objects["errors"]["Neural Network"] = str(e)

    return results, fitted_objects


# =========================
# Single-dataset CV
# =========================
def run_single_dataset_cv(df):
    rows = []

    X, y = split_features_target(df, "HeartDisease")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    last_fitted = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold_idx} / 5 =====")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        results, fitted = train_model_suite(X_train, y_train, X_test, y_test)
        last_fitted = fitted

        for model_name, metrics in results.items():
            row = {
                "Experiment": "Within-Dataset CV",
                "Dataset": "heart",
                "Fold": fold_idx,
                "TrainSource": "heart",
                "TestSource": "heart",
                "Model": model_name,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows), last_fitted


# =========================
# Deployment models for advisor
# =========================
def train_deployment_models(df):
    X, y = split_features_target(df, "HeartDisease")

    _, fitted = train_model_suite(X, y, X, y)

    feature_info = []
    for col in X.columns:
        col_series = X[col]

        if pd.api.types.is_numeric_dtype(col_series):
            feature_type = "numeric"
            unique_values = None
            min_value = float(col_series.min())
            max_value = float(col_series.max())
        else:
            feature_type = "categorical"
            unique_values = sorted([str(v) for v in col_series.dropna().unique().tolist()])
            min_value = None
            max_value = None

        feature_info.append({
            "name": col,
            "type": feature_type,
            "unique_values": unique_values,
            "min": min_value,
            "max": max_value,
        })

    advisor_models = {}
    for key in [
        "Logistic Regression",
        "Random Forest",
        "Tuned Random Forest",
        "MLP Classifier",
        "Stacking Ensemble",
        "XGBoost",
        "Domain-Weighted XGBoost",
    ]:
        if key in fitted:
            advisor_models[key] = fitted[key]

    advisor_bundle = {
        "shared_features": list(X.columns),
        "feature_info": feature_info,
        "models": advisor_models,
    }

    joblib.dump(advisor_bundle, os.path.join(MODELS_DIR, "advisor_bundle.pkl"))

    # Save standalone model files too
    if "Random Forest" in fitted:
        joblib.dump(fitted["Random Forest"], os.path.join(MODELS_DIR, "rf_model.pkl"))
    if "Tuned Random Forest" in fitted:
        joblib.dump(fitted["Tuned Random Forest"], os.path.join(MODELS_DIR, "tuned_rf_model.pkl"))
    if "XGBoost" in fitted:
        joblib.dump(fitted["XGBoost"], os.path.join(MODELS_DIR, "xgb_model.pkl"))
    if "MLP Classifier" in fitted:
        joblib.dump(fitted["MLP Classifier"], os.path.join(MODELS_DIR, "mlp_model.pkl"))

    return fitted


# =========================
# Full pipeline
# =========================
def train_full_project_pipeline(df):
    experiment_df, _ = run_single_dataset_cv(df)

    if experiment_df.empty:
        raise ValueError(
            "All models failed during training. Please check preprocessing and model configuration."
        )

    summary_df = (
        experiment_df.groupby(["Experiment", "Model"])[
            ["Accuracy", "Precision", "Recall", "F1", "PR_AUC", "ROC_AUC", "Brier", "ECE"]
        ]
        .mean()
        .reset_index()
        .sort_values(by=["F1"], ascending=False)
    )

    experiment_df.to_csv(os.path.join(OUTPUTS_DIR, "experiment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUTS_DIR, "summary_results.csv"), index=False)

    deployment_fitted = train_deployment_models(df)

    artifacts = {
        "all_results": experiment_df,
        "summary_results": summary_df,
        "within_results": experiment_df,
        "best_rf_params": deployment_fitted.get("best_rf_params", {}),
        "dnn_loss_history": deployment_fitted.get("dnn_loss_history", []),
        "errors": deployment_fitted.get("errors", {}),
    }

    return summary_df, artifacts