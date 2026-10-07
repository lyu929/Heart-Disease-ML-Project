"""Figures for the report (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from .calibration import reliability_table  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def roc_pr(pooled: dict[str, tuple[np.ndarray, np.ndarray]], path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    for name, (y, p) in pooled.items():
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, lw=1.4, label=f"{name} ({roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        b.plot(rec, prec, lw=1.4, label=name)
    a.plot([0, 1], [0, 1], "k:", lw=0.8)
    a.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (pooled out-of-fold)")
    a.legend(fontsize=7, loc="lower right")
    prev = np.mean(next(iter(pooled.values()))[0])
    b.axhline(prev, color="k", ls=":", lw=0.8)
    b.set(xlabel="recall", ylabel="precision", title="Precision–recall")
    return _save(fig, path)


def calibration(pooled: dict[str, tuple[np.ndarray, np.ndarray]], path, n_bins: int = 10) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    a.plot([0, 1], [0, 1], "k:", lw=0.8)
    for name, (y, p) in pooled.items():
        t = reliability_table(y, p, n_bins)
        a.errorbar(
            t["mean_predicted"],
            t["observed"],
            yerr=[t["observed"] - t["lo"], t["hi"] - t["observed"]],
            marker="o",
            ms=3,
            lw=1.2,
            capsize=2,
            label=name,
        )
        b.hist(p, bins=30, histtype="step", lw=1.2, label=name)
    a.set(
        xlabel="mean predicted risk",
        ylabel="observed frequency",
        title="Reliability (deciles, Wilson 95% CI)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    a.legend(fontsize=7)
    b.set(xlabel="predicted risk", ylabel="patients", title="Risk distribution")
    return _save(fig, path)


def decision_curve(dca: pd.DataFrame, path, prevalence: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in dca.groupby("strategy", sort=False):
        style = {"treat all": dict(color="0.5", ls="--"), "treat none": dict(color="k", ls=":")}.get(name, {})
        ax.plot(g["threshold"], g["net_benefit"], lw=1.4, label=name, **style)
    top = dca["net_benefit"].max()
    ax.set(
        ylim=(-0.05, top * 1.1),
        xlim=(0, 0.9),
        xlabel="threshold probability",
        ylabel="net benefit",
        title="Decision curve (pooled out-of-fold)",
    )
    ax.legend(fontsize=7)
    return _save(fig, path)


def forest(summary: pd.DataFrame, metric: str, path, title: str | None = None) -> Path:
    s = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(s) + 1.2))
    yy = np.arange(len(s))
    ax.errorbar(
        s[metric],
        yy,
        xerr=[s[metric] - s[f"{metric}_lo"], s[f"{metric}_hi"] - s[metric]],
        fmt="o",
        capsize=3,
        color="C0",
    )
    ax.set_yticks(yy, s["model"])
    ax.set(xlabel=f"{metric} (mean, corrected 95% CI)", title=title or metric)
    return _save(fig, path)


def leakage(table: pd.DataFrame, metric: str, path, title: str) -> Path:
    t = table[table["metric"] == metric]
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(t) + 1.7))
    yy = np.arange(len(t))
    ax.barh(yy - 0.2, t["honest"], height=0.4, label="threshold from training OOF (honest)")
    ax.barh(yy + 0.2, t["test_tuned"], height=0.4, label="threshold tuned on test fold (legacy)")
    ax.set_yticks(yy, t["model"])
    lo = min(t["honest"].min(), t["test_tuned"].min())
    ax.set(xlim=(max(0, lo - 0.08), min(1, t["test_tuned"].max() + 0.04)), xlabel=metric, title=title)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    return _save(fig, path)


def importance(df: pd.DataFrame, path, title: str = "Permutation importance") -> Path:
    d = df.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(d) + 1.2))
    ax.barh(d["feature"], d["mean"], xerr=d["std"], capsize=2, color="C2")
    ax.set(xlabel="drop in held-out ROC-AUC when permuted", title=title)
    return _save(fig, path)


def sites(df: pd.DataFrame, path) -> Path:
    site_names = list(dict.fromkeys(df["site"]))
    models = list(dict.fromkeys(df["model"]))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        g = df[df["model"] == m].set_index("site").loc[site_names]
        x = np.arange(len(site_names)) + (i - (len(models) - 1) / 2) * w
        ax.errorbar(
            x,
            g["roc_auc"],
            yerr=[g["roc_auc"] - g["roc_auc_lo"], g["roc_auc_hi"] - g["roc_auc"]],
            fmt="o",
            capsize=3,
            label=m,
        )
    labels = [
        f"{s}\n(n={int(g.loc[s, 'n_pos'] + g.loc[s, 'n_neg'])}, {int(g.loc[s, 'n_neg'])} neg)"
        for s in site_names
    ]
    ax.set_xticks(np.arange(len(site_names)), labels, fontsize=7)
    ax.set(
        ylabel="ROC-AUC on held-out hospital",
        title="Leave-one-hospital-out validation (bootstrap 95% CI)",
        ylim=(0.4, 1.0),
    )
    ax.legend(fontsize=7)
    return _save(fig, path)


def threshold_tradeoff(y, p, chosen: float, path) -> Path:
    from .metrics import threshold_metrics

    ts = np.linspace(0.02, 0.98, 97)
    m = pd.DataFrame([threshold_metrics(y, p, t) for t in ts])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for col in ("sensitivity", "specificity", "ppv", "npv", "f1"):
        ax.plot(ts, m[col], lw=1.3, label=col)
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"chosen ({chosen:.2f})")
    ax.set(xlabel="decision threshold", ylabel="value", title="Operating characteristics (out-of-fold)")
    ax.legend(fontsize=7)
    return _save(fig, path)


def waterfall(
    expl: pd.DataFrame, base: float, prob: float, path, title: str = "Why this prediction?"
) -> Path:
    d = expl.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(d) + 1.4))
    colors = ["C3" if c > 0 else "C0" for c in d["contribution"]]
    labels = [
        f"{f} = {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{f} = {v}"
        for f, v in zip(d["feature"], d["value"])
    ]
    ax.barh(labels, d["contribution"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(
        xlabel="contribution to predicted probability",
        title=f"{title}\nbaseline {base:.2f} → prediction {prob:.2f}",
    )
    return _save(fig, path)
