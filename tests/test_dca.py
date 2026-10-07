import numpy as np
import pytest

from heartrisk.dca import decision_curve, net_benefit, treat_all


def test_perfect_model_net_benefit_equals_prevalence():
    y = np.array([1, 0, 0, 1, 0])
    nb = net_benefit(y, y.astype(float), [0.1, 0.5, 0.9])
    assert np.allclose(nb, 0.4)


def test_treat_all_formula():
    y = np.array([1, 0, 0, 0])
    assert treat_all(y, [0.2])[0] == pytest.approx(0.25 - 0.75 * 0.25)
    # a model predicting 1 for everyone is the treat-all strategy
    assert np.allclose(net_benefit(y, np.ones(4), [0.2, 0.4]), treat_all(y, [0.2, 0.4]))


def test_decision_curve_table():
    y = np.array([1, 0, 1, 0])
    df = decision_curve(y, {"m": np.array([0.9, 0.1, 0.8, 0.3])}, thresholds=[0.2, 0.5])
    assert set(df["strategy"]) == {"treat all", "treat none", "m"} and len(df) == 6
