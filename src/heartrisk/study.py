"""End-to-end study: every number and figure in ``reports/REPORT.md`` is produced here.

``heartrisk study`` runs, in order: data audit, nested repeated CV of all models with
calibration and cleaning ablations, leakage audit (real and permuted labels), pooled
calibration / decision curves, permutation importance, leave-one-hospital-out validation,
cross-cohort benchmark, training of the deployable bundle and an example explanation.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import __version__, plots
from .data import DATASETS, clean, data_card, infer_sites, load_dataset
from .dca import decision_curve
from .evaluation import CVResult, cross_validate, site_validation
from .explain import explain_patient, permutation_importance_cv
from .models import build_model, label
from .predict import prepare
from .schema import EXAMPLE
from .train import train_bundle


def quick_config(cfg: dict) -> dict:
    """Small settings for CI / smoke tests (minutes -> seconds)."""
    c = copy.deepcopy(cfg)
    c["evaluation"].update(
        repeats=1, outer_splits=3, inner_splits=3, tune=False, bootstrap=100, models=["logreg", "xgb"]
    )
    c["benchmark"].update(datasets=["heart", "cleveland"], models=["logreg"], repeats=1)
    c["site_validation"]["models"] = ["logreg"]
    return c


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ci(row, m, fmt=".3f") -> str:
    return f"{row[m]:{fmt}} ({row[m + '_lo']:{fmt}}–{row[m + '_hi']:{fmt}})"


def _variants(cfg: dict) -> dict[str, dict]:
    base = cfg["evaluation"]["baseline"]
    out = {}
    for method in ("sigmoid", "isotonic"):
        c = copy.deepcopy(cfg)
        c["wrap_calibration"] = method
        out[f"{base} + {method}"] = {"model": base, "cfg": c}
    c = copy.deepcopy(cfg)
    c["cleaning"]["zero_as_missing"] = []
    out[f"{base} [zeros kept, legacy cleaning]"] = {"model": base, "cfg": c}
    c = copy.deepcopy(cfg)
    c["cleaning"]["missing_indicator"] = False
    out[f"{base} [no missing indicator]"] = {"model": base, "cfg": c}
    return out


def run_study(
    cfg: dict,
    out_dir: str | Path = "reports",
    results_dir: str | Path = "results",
    bundle_path: str | Path = "models/heartrisk.joblib",
    n_jobs: int = -1,
    log=print,
    reuse: bool = False,
) -> Path:
    t_start = time.time()
    out, res_dir = Path(out_dir), Path(results_dir)
    fig, tab = out / "figures", out / "tables"
    for d in (fig, tab, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    ev = cfg["evaluation"]
    seed = int(cfg.get("seed", 42))
    data_dir = cfg.get("data_dir", "data")
    baseline = ev["baseline"]
    R: dict = {"config": cfg, "version": __version__}

    # 1. data audit ----------------------------------------------------------------
    log("[1/9] data audit")
    X, y, spec = load_dataset("heart", data_dir)
    zero_cols = cfg["cleaning"]["zero_as_missing"]
    Xc = clean(X.assign(**{spec.target: y}), spec, zero_cols)
    card = data_card(Xc[spec.features], y)
    card.to_csv(tab / "data_card.csv", index=False)
    sites = infer_sites(X.assign(**{spec.target: y}))
    site_tab = (
        pd.DataFrame({"site": sites, "y": y, "chol_missing": Xc["Cholesterol"].isna()})
        .groupby("site", sort=False)
        .agg(patients=("y", "size"), prevalence=("y", "mean"), cholesterol_missing=("chol_missing", "mean"))
        .reset_index()
    )
    site_tab.to_csv(tab / "sites.csv", index=False)
    chol0 = X["Cholesterol"] == 0
    R["chol_zero"] = {
        "n": int(chol0.sum()),
        "prev_zero": float(y[chol0].mean()),
        "prev_rest": float(y[~chol0].mean()),
    }

    # 2. nested repeated CV ----------------------------------------------------------
    log(f"[2/9] nested CV: {ev['models']} + ablations ({ev['repeats']}x{ev['outer_splits']} folds)")
    variants = _variants(cfg)
    cv_dir = res_dir / "cv_heart"
    cv = CVResult.load(cv_dir) if reuse and (cv_dir / "folds.csv").exists() else None
    expected = {*ev["models"], *variants}
    if cv is None or set(cv.models) != expected or cv.meta.get("repeats") != ev["repeats"]:
        cv = cross_validate(X, y, spec, cfg, ev["models"], n_jobs=n_jobs, variants=variants)
        cv.save(cv_dir)
    else:
        log("      reusing cached CV results from " + str(cv_dir))
    summary = cv.summary()
    summary.to_csv(tab / "cv_summary.csv", index=False)
    comp_auc = cv.compare(baseline, "roc_auc")
    comp_brier = cv.compare(baseline, "brier")
    pd.concat([comp_auc, comp_brier]).to_csv(tab / "comparisons.csv", index=False)
    main_models = list(ev["models"])
    plots.forest(
        summary[summary["model"].isin(main_models)].assign(model=lambda d: d["model"].map(label)),
        "roc_auc",
        fig / "model_auc.png",
        "ROC-AUC, 5x5 nested CV",
    )
    plots.forest(
        summary.assign(model=lambda d: d["model"].map(label)),
        "brier",
        fig / "model_brier.png",
        "Brier score (lower is better), incl. ablations",
    )

    pooled = {label(m): cv.pooled(m) for m in main_models}
    pooled_rows = []
    for m in main_models:
        y_p, p_p = cv.pooled(m)
        auc = cv.pooled_ci(m, roc_auc_score, n_boot=ev["bootstrap"], seed=seed)
        pooled_rows.append(
            {
                "model": label(m),
                "pooled_auc": auc[0],
                "auc_lo": auc[1],
                "auc_hi": auc[2],
                "pr_auc": average_precision_score(y_p, p_p),
                "brier": brier_score_loss(y_p, p_p),
            }
        )
    pooled_tab = pd.DataFrame(pooled_rows)
    pooled_tab.to_csv(tab / "pooled_oof.csv", index=False)
    plots.roc_pr(pooled, fig / "roc_pr.png")
    cal_models = [baseline, f"{baseline} + sigmoid", f"{baseline} + isotonic"] + [
        m for m in main_models if m in ("rf", "xgb", "hgb")
    ]
    plots.calibration({label(m): cv.pooled(m) for m in cal_models}, fig / "calibration.png")
    y_b, p_b = cv.pooled(baseline)
    dca = decision_curve(
        y_b, {label(m): cv.pooled(m)[1] for m in main_models if m in (baseline, "xgb", "rf")}
    )
    dca.to_csv(tab / "decision_curve.csv", index=False)
    plots.decision_curve(dca, fig / "decision_curve.png")
    thr_base = float(cv.folds[cv.folds["model"] == baseline]["threshold"].median())
    plots.threshold_tradeoff(y_b, p_b, thr_base, fig / "threshold_tradeoff.png")

    # 3. leakage audit ---------------------------------------------------------------
    log("[3/9] leakage audit")
    leak = cv.leakage_table()
    leak = leak[leak["model"].isin(main_models)].assign(model=lambda d: d["model"].map(label))
    leak.to_csv(tab / "leakage_real.csv", index=False)
    plots.leakage(leak, "f1", fig / "leakage_f1.png", "F1: honest vs test-tuned threshold (25 folds)")
    perm_cfg = copy.deepcopy(cfg)
    perm_cfg["evaluation"]["threshold"]["strategy"] = "youden"
    perm_cfg["evaluation"]["tune"] = False
    y_perm = pd.Series(np.random.default_rng(seed).permutation(y.to_numpy()), index=y.index, name=y.name)
    perm_models = [m for m in ("logreg", "xgb") if m in main_models] or main_models[:1]
    cv_perm = cross_validate(
        X, y_perm, spec, perm_cfg, perm_models, repeats=max(1, ev["repeats"] // 2 + 1), n_jobs=n_jobs
    )
    leak_perm = cv_perm.leakage_table(("balanced_accuracy",)).assign(model=lambda d: d["model"].map(label))
    leak_perm["roc_auc"] = [
        cv_perm.folds.loc[cv_perm.folds["model"] == m, "roc_auc"].mean() for m in perm_models
    ]
    leak_perm.to_csv(tab / "leakage_permuted.csv", index=False)

    # 4. importance --------------------------------------------------------------------
    log("[4/9] permutation importance")
    imp = permutation_importance_cv(
        build_model(baseline, spec, cfg, seed), X, y, n_splits=5, n_repeats=10, seed=seed
    )
    imp.to_csv(tab / "importance.csv", index=False)
    plots.importance(imp, fig / "importance.png", f"Permutation importance – {label(baseline)}")

    # 5. leave-one-hospital-out ----------------------------------------------------------
    log("[5/9] leave-one-hospital-out validation")
    sv_models = cfg["site_validation"]["models"]
    site_res, _ = site_validation(X, y, sites, spec, cfg, sv_models, n_jobs=n_jobs, n_boot=ev["bootstrap"])
    site_res = site_res.assign(model=lambda d: d["model"].map(label))
    keep = [
        "site",
        "model",
        "n_pos",
        "n_neg",
        "roc_auc",
        "roc_auc_lo",
        "roc_auc_hi",
        "brier",
        "cal_intercept",
        "cal_slope",
        "threshold",
        "sensitivity",
        "specificity",
    ]
    site_res[keep].to_csv(tab / "site_validation.csv", index=False)
    plots.sites(site_res, fig / "site_validation.png")

    # 6. cross-cohort benchmark -------------------------------------------------------------
    log("[6/9] cohort benchmark")
    bm = cfg["benchmark"]
    bench_rows = []
    for ds in bm["datasets"]:
        Xd, yd, sd = load_dataset(ds, data_dir)
        bcfg = copy.deepcopy(cfg)
        bcfg["evaluation"]["tune"] = False
        r = cross_validate(Xd, yd, sd, bcfg, bm["models"], repeats=bm["repeats"], n_jobs=n_jobs)
        s = r.summary(["roc_auc", "pr_auc", "brier", "f1", "sensitivity", "specificity"])
        s.insert(0, "dataset", ds)
        s.insert(1, "n", len(yd))
        s.insert(2, "prevalence", yd.mean())
        bench_rows.append(s)
    bench = pd.concat(bench_rows, ignore_index=True).assign(model=lambda d: d["model"].map(label))
    bench.to_csv(tab / "benchmark.csv", index=False)

    # 7. deployable bundle -------------------------------------------------------------------
    log("[7/9] training deployable bundle")
    bundle = train_bundle(cfg, log=log)
    bundle.save(bundle_path)

    # 8. example explanation -------------------------------------------------------------------
    log("[8/9] example explanation")
    row = prepare(bundle, pd.DataFrame([EXAMPLE]))
    expl = explain_patient(bundle, row)
    plots.waterfall(
        expl,
        expl.attrs["base_value"],
        expl.attrs["probability"],
        fig / "example_explanation.png",
        "Example patient (schema example)",
    )

    # 9. report ----------------------------------------------------------------------------------
    log("[9/9] writing report")
    R.update(minutes=(time.time() - t_start) / 60)
    path = write_report(
        out,
        cfg,
        card,
        site_tab,
        cv,
        summary,
        comp_auc,
        comp_brier,
        pooled_tab,
        leak,
        leak_perm,
        imp,
        site_res[keep],
        bench,
        bundle,
        R,
    )
    (res_dir / "study_meta.json").write_text(
        json.dumps({k: v for k, v in R.items() if k != "config"}, indent=2, default=str)
    )
    log(f"done in {R['minutes']:.1f} min -> {path}")
    return path


def write_report(
    out: Path,
    cfg,
    card,
    site_tab,
    cv: CVResult,
    summary,
    comp_auc,
    comp_brier,
    pooled_tab,
    leak,
    leak_perm,
    imp,
    site_res,
    bench,
    bundle,
    R,
) -> Path:
    ev = cfg["evaluation"]
    base = ev["baseline"]
    main = list(ev["models"])
    s = summary.set_index("model")

    def model_rows(models):
        rows = []
        for m in models:
            r = s.loc[m]
            rows.append(
                {
                    "model": label(m),
                    "ROC-AUC": _ci(r, "roc_auc"),
                    "PR-AUC": f"{r['pr_auc']:.3f}",
                    "Brier": _ci(r, "brier"),
                    "ECE": f"{r['ece']:.3f}",
                    "cal. slope": f"{r['cal_slope']:.2f}",
                    "F1": f"{r['f1']:.3f}",
                    "sens.": f"{r['sensitivity']:.3f}",
                    "spec.": f"{r['specificity']:.3f}",
                    "threshold": f"{r['threshold']:.2f}",
                }
            )
        return pd.DataFrame(rows)

    variants = [m for m in cv.models if m not in main]
    comp = comp_auc[["model", "mean_diff", "ci_lo", "ci_hi", "p", "p_holm"]].merge(
        comp_brier[["model", "mean_diff", "p_holm"]], on="model", suffixes=("_auc", "_brier")
    )
    comp = comp.rename(
        columns={
            "mean_diff_auc": "ΔAUC",
            "ci_lo": "ΔAUC lo",
            "ci_hi": "ΔAUC hi",
            "p": "p (AUC)",
            "p_holm_auc": "p Holm (AUC)",
            "mean_diff_brier": "ΔBrier",
            "p_holm_brier": "p Holm (Brier)",
        }
    )
    comp["model"] = comp["model"].map(label)
    lk = leak[leak["metric"] == "f1"][["model", "honest", "test_tuned", "optimism", "p"]]
    lp = leak_perm[["model", "roc_auc", "honest", "test_tuned", "optimism"]].rename(
        columns={"honest": "bal. acc. honest", "test_tuned": "bal. acc. test-tuned"}
    )
    b = bench[["dataset", "n", "prevalence", "model"]].copy()
    for m in ("roc_auc", "brier", "f1"):
        b[m] = bench.apply(lambda r, m=m: _ci(r, m), axis=1)
    perf = bundle.performance
    sens = cfg["evaluation"]["threshold"]
    meta = cv.meta
    best_auc = s.loc[main, "roc_auc"].idxmax()
    better = comp_auc[
        (comp_auc["mean_diff"] > 0) & (comp_auc["p_holm"] < 0.05) & comp_auc["model"].isin(main)
    ]
    if better.empty:
        decision = (
            f"No model is significantly better than {label(base)} after Holm correction, so the simplest "
            f"model is deployed ({label(base)}: transparent coefficients, good calibration, cheap exact "
            "explanations)."
        )
    else:
        decision = (
            f"{', '.join(label(m) for m in better['model'])} significantly outperform(s) {label(base)} "
            "(Holm-adjusted p < 0.05); consider deploying it via `deploy.model`."
        )
    brier_better = comp_brier[
        (comp_brier["mean_diff"] < 0) & (comp_brier["p_holm"] < 0.05) & comp_brier["model"].isin(main)
    ]
    if len(brier_better):
        decision += (
            " "
            + "; ".join(
                f"{label(r.model)} has a significantly lower Brier score (Δ {r.mean_diff:+.4f}, Holm p = {r.p_holm:.3f})"
                for r in brier_better.itertuples()
            )
            + ", a gain that is small next to its cost in complexity and runtime."
        )
    cal_rows = comp_brier[comp_brier["model"].isin([m for m in cv.models if "+" in m])]
    if len(cal_rows) and not ((cal_rows["mean_diff"] < 0) & (cal_rows["p_holm"] < 0.05)).any():
        cal_note = (
            f"Neither Platt (sigmoid) nor isotonic recalibration improves the Brier score of {label(base)} "
            "(it is already fitted by maximum likelihood), so no recalibration layer is deployed for it."
        )
    else:
        cal_note = (
            "Recalibration changes the Brier score significantly for at least one method; see the table."
        )
    deploy_note = ""
    no_ind = f"{base} [no missing indicator]"
    if bundle.metadata.get("cleaning", {}).get("missing_indicator") is False and no_ind in s.index:
        row = comp_auc.set_index("model").loc[no_ind]
        deploy_note = (
            "The deployed recipe drops the missing-value indicator. Cholesterol is missing for the whole Swiss "
            "cohort (93% prevalence) and a quarter of the VA cohort (75%), so the indicator mostly tells the model "
            "that a patient came from a high-prevalence hospital, and an unmeasured cholesterol would raise the "
            "predicted risk by itself. Dropping it "
            f"costs ΔAUC {row['mean_diff']:+.4f} (Holm p = {row['p_holm']:.2f}, not significant). Nested-CV "
            f"performance of exactly this recipe: ROC-AUC {_ci(s.loc[no_ind], 'roc_auc')}, Brier "
            f"{s.loc[no_ind, 'brier']:.3f}, calibration slope {s.loc[no_ind, 'cal_slope']:.2f}."
        )
    swiss = site_res[site_res["site"] == "switzerland"]
    swiss_neg = int(swiss["n_neg"].iloc[0]) if len(swiss) else 0

    text = f"""# heartrisk – evaluation report

*Generated by `heartrisk study` (heartrisk {R["version"]}, {R["minutes"]:.1f} min). Do not edit by hand.*

## 1. Data

`data/heart.csv`: {meta["n"]} patients, prevalence {meta["prevalence"]:.3f}. Cholesterol is recorded as 0 for
{R["chol_zero"]["n"]} patients; their disease rate is {R["chol_zero"]["prev_zero"]:.2f} vs
{R["chol_zero"]["prev_rest"]:.2f} for the rest. Those zeros are missing measurements, not values
(the whole Switzerland cohort has no cholesterol). Zeros in `Cholesterol`/`RestingBP` are converted to missing
**inside** the pipeline and imputed with training-fold medians plus a missing-indicator.

{md_table(card)}

The file concatenates four hospitals in a fixed order. Site labels are recovered from the published cohort sizes
and verified against the published number of positives per cohort (`heartrisk.data.infer_sites`):

{md_table(site_tab)}

## 2. Protocol

* {meta["repeats"]}× repeated stratified {meta["outer_splits"]}-fold CV; all models share the same splits (paired).
* Inside every outer training part: {meta["inner_splits"]}-fold grid search (`{ev["tune_scoring"]}`), then the
  decision threshold (`{sens["strategy"]}`) is chosen on **inner out-of-fold** predictions, then the model is
  refitted and the untouched test fold is scored.
* Uncertainty: corrected resampled t-test / CI (Nadeau & Bengio 2003) because CV folds overlap; Holm correction
  for multiple comparisons; percentile bootstrap for pooled predictions.

## 3. Results

### 3.1 Model comparison (mean over {len(cv.folds[cv.folds["model"] == base])} outer folds, corrected 95% CI)

{md_table(model_rows(main))}

![AUC](figures/model_auc.png)

Paired comparison against **{label(base)}** (corrected t-test, Holm-adjusted):

{md_table(comp[comp["model"].isin([label(m) for m in main])], ".4f")}

Highest mean ROC-AUC: **{label(best_auc)}**. {decision}

Pooled out-of-fold predictions (each patient's probability averaged over repeats; bootstrap 95% CI):

{md_table(pooled_tab)}

![ROC / PR](figures/roc_pr.png)

### 3.2 Calibration and clinical utility

{md_table(model_rows([m for m in variants if "+" in m]))}

![calibration](figures/calibration.png)

{cal_note} Calibration slope > 1 means predictions are too conservative, < 1 too extreme. The decision curve shows the net
benefit of acting on the model compared with treating everybody or nobody:

![DCA](figures/decision_curve.png)

![threshold](figures/threshold_tradeoff.png)

### 3.3 Ablations (cleaning)

{md_table(model_rows([m for m in variants if "[" in m]))}

### 3.4 Leakage audit

The course version chose the F1-optimal threshold **on the test fold** and reported test metrics at that
threshold. Each outer fold here records both numbers for the same fitted model:

{md_table(lk, ".4f")}

![leakage](figures/leakage_f1.png)

Sanity check with **randomly permuted labels** (no signal; Youden threshold): a test-tuned threshold still produces
better-than-chance balanced accuracy, the honest protocol does not:

{md_table(lp, ".3f")}

### 3.5 Which variables matter

![importance](figures/importance.png)

{md_table(imp.rename(columns={"mean": "ΔAUC mean", "std": "ΔAUC sd"}), ".4f")}

### 3.6 Leave-one-hospital-out validation

Each hospital is held out in turn; tuning and the threshold use only the other three. This is the closest the
available data gets to external validation.

{md_table(site_res)}

![sites](figures/site_validation.png)

Switzerland has only {swiss_neg} negatives and no cholesterol values, so its AUC interval is wide. A
calibration intercept far from 0 means the overall risk level learned at the other hospitals does not transfer
(different referral populations and prevalence), even when discrimination does.

### 3.7 Same protocol on other cohorts

Within-cohort CV ({cfg["benchmark"]["repeats"]}× repeated, no tuning). The cohorts are **not** pooled: Framingham
predicts 10-year incident CHD (a different outcome), and the processed files were standardised before release.

{md_table(b)}

## 4. Deployed model

{deploy_note}

`{bundle.model_label}` with `{bundle.calibration}` calibration, threshold {bundle.threshold:.3f}
(`{bundle.threshold_strategy}` on out-of-fold predictions), trained on all {bundle.metadata["n_train"]} patients.
Out-of-fold ROC-AUC {perf["roc_auc"]["value"]:.3f} ({perf["roc_auc"]["ci95"][0]:.3f}–{perf["roc_auc"]["ci95"][1]:.3f}),
Brier {perf["brier"]["value"]:.3f}, sensitivity {perf["sensitivity"]:.3f}, specificity {perf["specificity"]:.3f}.

![example](figures/example_explanation.png)

Exact Shapley values over the 11 raw variables (2,048 coalitions against a {len(bundle.background)}-patient
reference sample); they sum exactly to prediction − baseline.
"""
    path = out / "REPORT.md"
    path.write_text(text)
    return path


__all__ = ["run_study", "quick_config", "md_table", "DATASETS"]
