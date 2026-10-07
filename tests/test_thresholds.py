import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from heartrisk.metrics import threshold_metrics
from heartrisk.thresholds import oof_threshold, select_threshold

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 400)
P = np.clip(0.35 * Y + rng.normal(0.35, 0.2, 400), 0, 1)


def test_f1_threshold_is_the_exact_optimum():
    t = select_threshold(Y, P, "f1")
    best = max(threshold_metrics(Y, P, c)["f1"] for c in np.unique(P))
    assert threshold_metrics(Y, P, t)["f1"] == pytest.approx(best)


def test_youden_threshold_is_the_exact_optimum():
    t = select_threshold(Y, P, "youden")
    j = lambda c: threshold_metrics(Y, P, c)["sensitivity"] + threshold_metrics(Y, P, c)["specificity"]  # noqa: E731
    assert j(t) == pytest.approx(max(j(c) for c in np.unique(P)))


def test_sensitivity_target_is_met_with_the_highest_threshold():
    t = select_threshold(Y, P, "sensitivity", target_sensitivity=0.9)
    assert threshold_metrics(Y, P, t)["sensitivity"] >= 0.9
    higher = np.unique(P)[np.unique(P) > t]
    assert all(threshold_metrics(Y, P, c)["sensitivity"] < 0.9 for c in higher)


def test_fixed_and_degenerate_cases():
    assert select_threshold(Y, P, "fixed", fixed=0.3) == 0.3
    assert select_threshold(np.ones(5), np.linspace(0, 1, 5), "f1", fixed=0.5) == 0.5
    with pytest.raises(ValueError):
        select_threshold(Y, P, "magic")


def test_oof_threshold_uses_only_given_rows():
    X = (P + rng.normal(0, 0.1, 400)).reshape(-1, 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    t1, oof = oof_threshold(LogisticRegression(), X[:300], Y[:300], cv)
    t2, _ = oof_threshold(LogisticRegression(), X[:300], Y[:300], cv)
    assert t1 == t2 and len(oof) == 300 and 0 < t1 < 1
