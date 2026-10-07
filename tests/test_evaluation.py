import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from heartrisk.data import infer_sites
from heartrisk.evaluation import CVResult, cross_validate, fit_fold, site_validation


def _split(X, y):
    return next(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))


def test_test_labels_never_influence_threshold_or_predictions(heart, cfg):
    """The core leakage guarantee: scrambling the test-fold labels changes nothing that was fitted."""
    X, y, spec = heart
    tr, te = _split(X, y)
    row1, pred1 = fit_fold("logreg", spec, cfg, X, y, tr, te)
    y2 = y.copy()
    y2.iloc[te] = np.random.default_rng(0).permutation(y.iloc[te].to_numpy())
    row2, pred2 = fit_fold("logreg", spec, cfg, X, y2, tr, te)
    assert row1["threshold"] == row2["threshold"]
    np.testing.assert_allclose(pred1["prob"], pred2["prob"])
    # ...whereas the legacy (test-tuned) threshold does depend on them
    assert row1["leaky_threshold"] != row2["leaky_threshold"]


def test_honest_metrics_never_beat_test_tuned_ones(heart, cfg):
    X, y, spec = heart
    tr, te = _split(X, y)
    row, _ = fit_fold("logreg", spec, cfg, X, y, tr, te)
    assert row["leaky_f1"] >= row["f1"] - 1e-12


def test_cross_validate_structure_and_paired_splits(heart, cfg):
    X, y, spec = heart
    res = cross_validate(X, y, spec, cfg, ["logreg", "xgb"], repeats=2, n_jobs=1)
    k = cfg["evaluation"]["outer_splits"]
    assert len(res.folds) == 2 * 2 * k
    for _, g in res.oof.groupby(["model", "repeat"]):
        assert sorted(g["row"]) == list(range(len(y)))  # every patient scored exactly once per repeat
    a = res.oof[res.oof.model == "logreg"].sort_values(["repeat", "row"])
    b = res.oof[res.oof.model == "xgb"].sort_values(["repeat", "row"])
    assert (a["y"].to_numpy() == b["y"].to_numpy()).all()
    s = res.summary()
    assert {"roc_auc", "roc_auc_lo", "roc_auc_hi", "brier"} <= set(s.columns)
    assert (s["roc_auc"] > 0.85).all()
    comp = res.compare("logreg")
    assert comp["model"].tolist() == ["xgb"] and 0 <= comp["p_holm"].iloc[0] <= 1
    yy, pp = res.pooled("logreg")
    assert len(yy) == len(y) and ((pp >= 0) & (pp <= 1)).all()
    assert set(res.leakage_table()["metric"]) == {"f1", "balanced_accuracy", "sensitivity", "specificity"}


def test_permuted_labels_give_chance_auc(heart, cfg):
    X, y, spec = heart
    y_perm = pd.Series(np.random.default_rng(1).permutation(y.to_numpy()), index=y.index, name=y.name)
    res = cross_validate(X, y_perm, spec, cfg, ["logreg"], repeats=1, n_jobs=1)
    assert abs(res.folds["roc_auc"].mean() - 0.5) < 0.08


def test_variants_and_roundtrip(heart, cfg, tmp_path):
    X, y, spec = heart
    variant_cfg = {**cfg, "wrap_calibration": "sigmoid"}
    res = cross_validate(
        X,
        y,
        spec,
        cfg,
        ["logreg"],
        repeats=1,
        n_jobs=1,
        variants={"logreg + sigmoid": {"model": "logreg", "cfg": variant_cfg}},
    )
    assert res.models == ["logreg", "logreg + sigmoid"]
    res.save(tmp_path)
    back = CVResult.load(tmp_path)
    pd.testing.assert_frame_equal(back.folds, res.folds, check_dtype=False)


def test_site_validation_holds_out_each_site(heart, cfg):
    X, y, spec = heart
    sites = infer_sites(X.assign(HeartDisease=y))
    rows, preds = site_validation(X, y, sites, spec, cfg, ["logreg"], n_jobs=1, n_boot=50)
    assert rows["site"].tolist() == ["hungarian", "switzerland", "va_long_beach", "cleveland"]
    assert (rows["n_train"] + rows["n_test"] == len(y)).all()
    for site, g in preds.groupby("site"):
        assert (sites.iloc[g["row"]] == site).all()
    assert (rows["roc_auc_lo"] <= rows["roc_auc"]).all() and (rows["roc_auc"] <= rows["roc_auc_hi"]).all()
