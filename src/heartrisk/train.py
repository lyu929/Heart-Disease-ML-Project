"""Train the deployable model bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from .bundle import ModelBundle, git_commit
from .data import clean, get_spec, load_dataset, sha256
from .metrics import compute_metrics
from .models import build_model, label, param_grid
from .stats import bootstrap_ci
from .thresholds import select_threshold


def _reference_sample(X, y, n: int, seed: int):
    """Stratified reference rows for Shapley explanations (kept in the bundle)."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in (0, 1):
        members = np.where(y.to_numpy() == cls)[0]
        k = max(1, round(n * len(members) / len(y)))
        idx += rng.choice(members, size=min(k, len(members)), replace=False).tolist()
    return X.iloc[sorted(idx)].reset_index(drop=True)


def train_bundle(
    cfg: dict, model_key: str | None = None, calibration: str | None = None, log=print
) -> ModelBundle:
    """Fit the deployment model on all of heart.csv.

    * hyper-parameters: inner-CV grid search (as in evaluation);
    * calibration: CalibratedClassifierCV with internal CV;
    * threshold: chosen on 5-fold out-of-fold predictions of the *whole* recipe;
    * performance: the same out-of-fold predictions, with bootstrap CIs. Because the
      grid search saw all rows these are very slightly optimistic; the unbiased estimate
      is the nested-CV report (``heartrisk evaluate``).
    """
    dataset = cfg.get("dataset", "heart")
    dep = cfg["deploy"]
    model_key = model_key or dep["model"]
    calibration = calibration or dep.get("calibration", "none")
    seed = int(cfg.get("seed", 42))
    ev = cfg["evaluation"]
    X, y, spec = load_dataset(dataset, cfg.get("data_dir", "data"))
    cv = StratifiedKFold(int(ev["inner_splits"]), shuffle=True, random_state=seed)

    # The deployment recipe may override cleaning options (e.g. drop the missing-value
    # indicator, which mostly encodes the recording hospital; see the report).
    cleaning = {**(cfg.get("cleaning") or {})}
    if dep.get("missing_indicator") is not None:
        cleaning["missing_indicator"] = bool(dep["missing_indicator"])
    cfg = {**cfg, "cleaning": cleaning}
    base_cfg = {**cfg, "wrap_calibration": None}
    est = build_model(model_key, spec, base_cfg, seed)
    params = {}
    grid = param_grid(model_key, base_cfg)
    if ev.get("tune", True) and grid:
        search = GridSearchCV(est, grid, scoring=ev.get("tune_scoring", "neg_log_loss"), cv=cv, n_jobs=1)
        search.fit(X, y)
        params = search.best_params_
        est.set_params(**params)
        log(f"tuned {model_key}: {params}")

    full_cfg = {**cfg, "wrap_calibration": calibration}
    final = build_model(model_key, spec, full_cfg, seed)
    final_params = {
        (f"estimator__{k}" if calibration not in (None, "none") else k): v for k, v in params.items()
    }
    final.set_params(**final_params)

    th = ev["threshold"]
    oof = cross_val_predict(final, X, y, cv=cv, method="predict_proba")[:, 1]
    threshold = select_threshold(
        y, oof, th.get("strategy", "f1"), th.get("target_sensitivity", 0.9), th.get("fixed", 0.5)
    )
    final.fit(X, y)
    log(f"fitted {model_key} (calibration={calibration}), threshold={threshold:.3f}")

    m = compute_metrics(y, oof, threshold)
    auc = bootstrap_ci(y, oof, roc_auc_score, n_boot=1000, seed=seed)
    brier = bootstrap_ci(y, oof, brier_score_loss, n_boot=1000, seed=seed)
    performance = {
        "estimate": "5-fold out-of-fold predictions of the deployed recipe",
        "roc_auc": {"value": auc[0], "ci95": [auc[1], auc[2]]},
        "brier": {"value": brier[0], "ci95": [brier[1], brier[2]]},
        **{
            k: m[k]
            for k in (
                "pr_auc",
                "ece",
                "cal_intercept",
                "cal_slope",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "f1",
            )
        },
    }

    Xc = clean(X.assign(**{spec.target: y}), spec, (cfg.get("cleaning") or {}).get("zero_as_missing", []))
    ranges = {c: [float(Xc[c].min()), float(Xc[c].max())] for c in (*spec.numeric, *spec.binary)}
    levels = {c: sorted(Xc[c].dropna().astype(str).unique().tolist()) for c in spec.categorical}
    data_path = Path(cfg.get("data_dir", "data")) / get_spec(dataset).file
    return ModelBundle(
        pipeline=final,
        model_key=model_key,
        model_label=label(model_key),
        dataset=dataset,
        features=spec.features,
        threshold=float(threshold),
        threshold_strategy=th.get("strategy", "f1"),
        calibration=calibration,
        background=_reference_sample(X, y, int(dep.get("background_size", 32)), seed),
        ranges=ranges,
        levels=levels,
        risk_bands=[list(b) for b in dep["risk_bands"]],
        performance=performance,
        metadata={
            "params": params,
            "n_train": len(y),
            "prevalence": float(y.mean()),
            "data_file": data_path.name,
            "data_sha256": sha256(data_path),
            "cleaning": cfg.get("cleaning"),
            "git_commit": git_commit(),
            "seed": seed,
        },
    )
