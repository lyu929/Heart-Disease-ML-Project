import numpy as np
import pytest

from heartrisk.metrics import (
    calibration_intercept_slope,
    compute_metrics,
    confusion_counts,
    expected_calibration_error,
    threshold_metrics,
)


def test_ece_known_value():
    y = np.array([0, 1, 1, 1])
    p = np.array([0.05, 0.95, 0.95, 0.95])  # bins: 0 -> (0.05 vs 0), 9 -> (0.95 vs 1)
    assert expected_calibration_error(y, p) == pytest.approx(0.25 * 0.05 + 0.75 * 0.05)


def test_calibration_slope_detects_overconfidence():
    rng = np.random.default_rng(1)
    lp = rng.normal(0, 1.5, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-lp))
    good = 1 / (1 + np.exp(-lp))
    over = 1 / (1 + np.exp(-2 * lp))
    a, b = calibration_intercept_slope(y, good)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05
    assert calibration_intercept_slope(y, over)[1] == pytest.approx(0.5, abs=0.05)


def test_threshold_metrics_from_known_confusion():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1])
    assert confusion_counts(y, p, 0.5) == (2, 1, 3, 1)
    m = threshold_metrics(y, p, 0.5)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(4 / 6)


def test_compute_metrics_handles_single_class():
    m = compute_metrics(np.ones(5), np.full(5, 0.7))
    assert np.isnan(m["roc_auc"]) and m["brier"] == pytest.approx(0.09)
