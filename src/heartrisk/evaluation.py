"""Leakage-free model evaluation.

Protocol for every outer fold (repeated stratified K-fold, identical splits for all models
so comparisons are paired):

1. optional hyper-parameter search with an inner stratified K-fold on the training part;
2. decision threshold chosen on inner *out-of-fold* predictions of the training part;
3. refit on the whole training part, score the untouched test fold at that frozen threshold.

For the leakage audit each fold also records the metrics the legacy protocol would have
reported (threshold re-optimised on the test fold), prefixed ``leaky_``. No extra models
are fitted for that.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from .data import DatasetSpec
from .metrics import HIGHER_IS_BETTER, compute_metrics, threshold_metrics
from .models import build_model, param_grid
from .stats import bootstrap_ci, corrected_ci, corrected_ttest, holm
from .thresholds import oof_threshold, select_threshold

REPORT_METRICS = [
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "cal_slope",
    "f1",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "threshold",
]


def _threshold_kwargs(cfg: dict) -> dict:
    th = cfg["evaluation"]["threshold"]
    return {
        "strategy": th.get("strategy", "f1"),
        "target_sensitivity": th.get("target_sensitivity", 0.9),
        "fixed": th.get("fixed", 0.5),
    }


def fit_fold(
    model_key: str,
    spec: DatasetSpec,
    cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    repeat: int = 0,
    fold: int = 0,
    tag: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Tune, choose the threshold and score one outer split. Returns (metrics row, test predictions)."""
    t0 = time.perf_counter()
    ev = cfg["evaluation"]
    seed = int(cfg.get("seed", 42))
    est = build_model(model_key, spec, cfg, seed)
    Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
    Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
    inner = StratifiedKFold(int(ev["inner_splits"]), shuffle=True, random_state=seed + 1000 * repeat + fold)

    params: dict = {}
    grid = param_grid(model_key, cfg)
    if ev.get("tune", True) and grid:
        search = GridSearchCV(
            est,
            grid,
            scoring=ev.get("tune_scoring", "neg_log_loss"),
            cv=inner,
            refit=False,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(Xtr, ytr)
        params = search.best_params_
        est.set_params(**params)

    kw = _threshold_kwargs(cfg)
    threshold, _ = oof_threshold(est, Xtr, ytr, inner, **kw)
    est.fit(Xtr, ytr)
    prob = est.predict_proba(Xte)[:, 1]

    row = {
        "model": tag or model_key,
        "repeat": repeat,
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "params": json.dumps(params, default=str),
    }
    row.update(compute_metrics(yte, prob, threshold))
    leaky_t = select_threshold(yte, prob, **kw)  # what the legacy protocol reported
    row["leaky_threshold"] = leaky_t
    row.update({f"leaky_{k}": v for k, v in threshold_metrics(yte, prob, leaky_t).items()})
    row["seconds"] = time.perf_counter() - t0
    preds = pd.DataFrame(
        {
            "model": tag or model_key,
            "repeat": repeat,
            "row": np.asarray(test_idx),
            "y": yte.to_numpy(),
            "prob": prob,
        }
    )
    return row, preds


@dataclass
class CVResult:
    folds: pd.DataFrame
    oof: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys(self.folds["model"]))

    def _sizes(self) -> tuple[int, int]:
        return int(self.folds["n_train"].mean()), int(self.folds["n_test"].mean())

    def summary(self, metrics: list[str] | None = None) -> pd.DataFrame:
        """Mean, SD and corrected 95% CI of each fold metric, per model."""
        metrics = metrics or REPORT_METRICS
        n_tr, n_te = self._sizes()
        rows = []
        for model, g in self.folds.groupby("model", sort=False):
            row = {"model": model, "folds": len(g)}
            for m in metrics:
                v = g[m].dropna().to_numpy()
                lo, hi = corrected_ci(v, n_tr, n_te)
                row.update({m: v.mean(), f"{m}_sd": v.std(ddof=1), f"{m}_lo": lo, f"{m}_hi": hi})
            rows.append(row)
        return pd.DataFrame(rows)

    def compare(self, baseline: str, metric: str = "roc_auc") -> pd.DataFrame:
        """Paired corrected t-tests of every model against ``baseline`` (Holm-adjusted)."""
        n_tr, n_te = self._sizes()
        wide = self.folds.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
        rows = []
        for model in self.models:
            if model == baseline:
                continue
            d = (wide[model] - wide[baseline]).dropna().to_numpy()
            t, p = corrected_ttest(d, n_tr, n_te)
            lo, hi = corrected_ci(d, n_tr, n_te)
            rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_diff": d.mean(),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "t": t,
                    "p": p,
                }
            )
        out = pd.DataFrame(rows)
        if len(out):
            out["p_holm"] = holm(out["p"].to_numpy())
        return out

    def pooled(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-patient out-of-fold probability averaged over repeats."""
        g = self.oof[self.oof["model"] == model].groupby("row").agg(y=("y", "first"), prob=("prob", "mean"))
        return g["y"].to_numpy(), g["prob"].to_numpy()

    def pooled_ci(self, model: str, metric_fn, n_boot: int = 2000, seed: int = 0):
        y, p = self.pooled(model)
        return bootstrap_ci(y, p, metric_fn, n_boot=n_boot, seed=seed)

    def best(self, metric: str = "roc_auc") -> str:
        s = self.summary([metric]).set_index("model")[metric]
        return str(s.idxmax() if HIGHER_IS_BETTER.get(metric, True) else s.idxmin())

    def leakage_table(
        self, metrics=("f1", "balanced_accuracy", "sensitivity", "specificity")
    ) -> pd.DataFrame:
        """Honest (out-of-fold threshold) vs legacy (test-tuned threshold) fold metrics."""
        n_tr, n_te = self._sizes()
        rows = []
        for model, g in self.folds.groupby("model", sort=False):
            for m in metrics:
                d = (g[f"leaky_{m}"] - g[m]).dropna().to_numpy()
                t, p = corrected_ttest(d, n_tr, n_te)
                rows.append(
                    {
                        "model": model,
                        "metric": m,
                        "honest": g[m].mean(),
                        "test_tuned": g[f"leaky_{m}"].mean(),
                        "optimism": d.mean(),
                        "p": p,
                    }
                )
        return pd.DataFrame(rows)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.folds.to_csv(out / "folds.csv", index=False)
        self.oof.to_csv(out / "oof_predictions.csv.gz", index=False)
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, out_dir: str | Path) -> CVResult:
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
        return cls(pd.read_csv(out / "folds.csv"), pd.read_csv(out / "oof_predictions.csv.gz"), meta)


def _run(tasks, n_jobs: int, verbose: int):
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(fit_fold)(*t[:-1], **t[-1]) for t in tasks)
    rows, preds = zip(*results) if results else ([], [])
    return pd.DataFrame(list(rows)), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    spec: DatasetSpec,
    cfg: dict,
    models: list[str] | None = None,
    repeats: int | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
    variants: dict[str, dict] | None = None,
) -> CVResult:
    """Repeated stratified K-fold CV of ``models`` (all models share the same splits).

    ``variants`` maps a display name to ``{"model": key, "cfg": cfg}`` for ablations that
    need a different configuration (e.g. without missing-value indicators).
    """
    ev = cfg["evaluation"]
    repeats = int(repeats or ev["repeats"])
    k = int(ev["outer_splits"])
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=int(cfg.get("seed", 42)))
    splits = list(rskf.split(X, y))
    jobs = [(key, key, cfg) for key in (models or ev["models"])]
    jobs += [(name, v["model"], v["cfg"]) for name, v in (variants or {}).items()]
    tasks = []
    for tag, key, c in jobs:
        for i, (tr, te) in enumerate(splits):
            tasks.append((key, spec, c, X, y, tr, te, {"repeat": i // k, "fold": i % k, "tag": tag}))
    folds, oof = _run(tasks, n_jobs, verbose)
    meta = {
        "dataset": spec.name,
        "n": len(y),
        "prevalence": float(y.mean()),
        "repeats": repeats,
        "outer_splits": k,
        "inner_splits": ev["inner_splits"],
        "tune": ev.get("tune"),
        "threshold": _threshold_kwargs(cfg),
        "seed": cfg.get("seed"),
    }
    return CVResult(folds, oof, meta)


def site_validation(
    X: pd.DataFrame,
    y: pd.Series,
    sites: pd.Series,
    spec: DatasetSpec,
    cfg: dict,
    models: list[str],
    n_jobs: int = -1,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-site-out ("internal–external") validation.

    Each hospital is held out in turn; tuning and threshold selection use only the other
    hospitals. Returns (per-site metrics with bootstrap AUC CIs, held-out predictions).
    """
    from sklearn.metrics import roc_auc_score

    site_names = list(dict.fromkeys(sites))
    tasks = []
    for f, site in enumerate(site_names):
        te = np.where(sites.to_numpy() == site)[0]
        tr = np.where(sites.to_numpy() != site)[0]
        for key in models:
            tasks.append((key, spec, cfg, X, y, tr, te, {"repeat": 0, "fold": f, "tag": key}))
    rows, preds = _run(tasks, n_jobs, 0)
    rows["site"] = [site_names[f] for f in rows["fold"]]
    preds["site"] = sites.to_numpy()[preds["row"].to_numpy()]
    cis = []
    for _, r in rows.iterrows():
        g = preds[(preds["model"] == r["model"]) & (preds["site"] == r["site"])]
        _, lo, hi = bootstrap_ci(g["y"], g["prob"], roc_auc_score, n_boot=n_boot, seed=0)
        cis.append((lo, hi, int(g["y"].sum()), int((1 - g["y"]).sum())))
    rows[["roc_auc_lo", "roc_auc_hi", "n_pos", "n_neg"]] = pd.DataFrame(cis, index=rows.index)
    return rows, preds
