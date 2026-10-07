"""Model zoo. Every model is a full pipeline (preprocessing + classifier) so that CV folds,
the deployed bundle and the API all run exactly the same code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .data import DatasetSpec
from .features import preprocessing_steps


def _xgb(seed: int, **kw):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )
    params.update(kw)
    return XGBClassifier(**params)


def _rf(seed: int, **kw):
    params = dict(n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=1, random_state=seed)
    params.update(kw)
    return RandomForestClassifier(**params)


def _logreg(seed: int, **kw):
    return LogisticRegression(C=kw.pop("C", 1.0), max_iter=5000, random_state=seed, **kw)


@dataclass(frozen=True)
class ModelDef:
    key: str
    label: str
    build: Callable[[DatasetSpec, dict, int], object]
    grid: dict


def _pipe(spec: DatasetSpec, cfg: dict, clf, scale: bool = True) -> Pipeline:
    return Pipeline([*preprocessing_steps(spec, cfg, scale), ("clf", clf)])


def _build_stacking(spec: DatasetSpec, cfg: dict, seed: int):
    base = [
        ("lr", _logreg(seed)),
        ("rf", _rf(seed, n_estimators=200)),
        ("xgb", _xgb(seed, n_estimators=200)),
    ]
    clf = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=5000),
        stack_method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_jobs=1,
    )
    return _pipe(spec, cfg, clf)


def _build_smote_rf(spec: DatasetSpec, cfg: dict, seed: int):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline(
        [
            *preprocessing_steps(spec, cfg),
            ("smote", SMOTE(random_state=seed)),
            ("clf", _rf(seed)),
        ]
    )


def _build_dnn(spec: DatasetSpec, cfg: dict, seed: int):
    from .dnn import TorchMLPClassifier

    return _pipe(spec, cfg, TorchMLPClassifier(random_state=seed, **cfg.get("dnn", {})))


MODELS: dict[str, ModelDef] = {
    "logreg": ModelDef(
        "logreg",
        "Logistic regression",
        lambda spec, cfg, seed: _pipe(spec, cfg, _logreg(seed)),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "rf": ModelDef(
        "rf",
        "Random forest",
        lambda spec, cfg, seed: _pipe(spec, cfg, _rf(seed), scale=False),
        {"clf__min_samples_leaf": [1, 5, 10]},
    ),
    "xgb": ModelDef(
        "xgb",
        "XGBoost",
        lambda spec, cfg, seed: _pipe(spec, cfg, _xgb(seed), scale=False),
        {"clf__max_depth": [2, 3, 4]},
    ),
    "hgb": ModelDef(
        "hgb",
        "Hist. gradient boosting",
        lambda spec, cfg, seed: _pipe(
            spec,
            cfg,
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
            ),
            scale=False,
        ),
        {"clf__max_leaf_nodes": [7, 15]},
    ),
    "mlp": ModelDef(
        "mlp",
        "MLP (scikit-learn)",
        lambda spec, cfg, seed: _pipe(
            spec,
            cfg,
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-2,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=seed,
            ),
        ),
        {"clf__alpha": [1e-3, 1e-2, 1e-1]},
    ),
    "stacking": ModelDef("stacking", "Stacking (LR + RF + XGB)", _build_stacking, {}),
    "smote_rf": ModelDef("smote_rf", "SMOTE + random forest", _build_smote_rf, {}),
    "dnn": ModelDef("dnn", "DNN (PyTorch, focal loss)", _build_dnn, {}),
}


def get_model(key: str) -> ModelDef:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODELS)}") from exc


def build_model(key: str, spec: DatasetSpec, cfg: dict, seed: int | None = None):
    """Build an unfitted pipeline. ``cfg["wrap_calibration"]`` (sigmoid/isotonic) wraps it in
    CalibratedClassifierCV; tuning grids are then addressed through ``estimator__``."""
    seed = cfg.get("seed", 42) if seed is None else seed
    est = get_model(key).build(spec, cfg, seed)
    method = cfg.get("wrap_calibration")
    if method and method != "none":
        from .calibration import calibrate

        est = calibrate(est, method, int(cfg.get("evaluation", {}).get("inner_splits", 5)), seed)
    return est


def param_grid(key: str, cfg: dict) -> dict:
    grid = get_model(key).grid
    method = cfg.get("wrap_calibration")
    if method and method != "none":
        return {f"estimator__{k}": v for k, v in grid.items()}
    return dict(grid)


def label(key: str) -> str:
    """Display name; variants such as ``"logreg + sigmoid"`` keep their suffix."""
    if key in MODELS:
        return MODELS[key].label
    head, _, rest = str(key).partition(" ")
    return f"{MODELS[head].label} {rest}" if head in MODELS else str(key)
