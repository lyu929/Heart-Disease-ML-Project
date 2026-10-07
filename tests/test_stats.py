import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from heartrisk.stats import bootstrap_ci, corrected_ci, corrected_ttest, holm


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 25)
    _, p_corr = corrected_ttest(d, n_train=734, n_test=184)
    p_naive = sps.ttest_1samp(d, 0).pvalue
    assert p_corr > p_naive


def test_corrected_ttest_formula():
    d = np.array([0.01, 0.02, 0.03, 0.00, 0.04])
    t, _ = corrected_ttest(d, 80, 20)
    assert t == pytest.approx(d.mean() / np.sqrt((1 / 5 + 20 / 80) * d.var(ddof=1)))


def test_corrected_ci_contains_mean():
    v = np.array([0.90, 0.92, 0.91, 0.93, 0.89])
    lo, hi = corrected_ci(v, 80, 20)
    assert lo < v.mean() < hi


def test_holm_known_values():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]).max() <= 1


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    point, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=300)
    assert lo < point < hi
