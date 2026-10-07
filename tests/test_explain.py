import numpy as np
import pandas as pd
import pytest

from heartrisk.explain import explain_patient, permutation_importance_cv, shapley_values
from heartrisk.models import build_model
from heartrisk.predict import prepare
from heartrisk.schema import EXAMPLE


def test_exact_shapley_of_additive_function():
    w = np.array([0.5, -1.0, 2.0])
    f = lambda df: df.to_numpy(float) @ w  # noqa: E731
    bg = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)), columns=list("abc"))
    x = pd.DataFrame([[1.0, 2.0, 3.0]], columns=list("abc"))
    phi, base, pred = shapley_values(f, x, bg)
    np.testing.assert_allclose(phi, w * (x.to_numpy()[0] - bg.mean().to_numpy()), atol=1e-12)
    assert pred == pytest.approx(f(x)[0]) and base == pytest.approx(f(bg).mean())


def test_monte_carlo_estimator_matches_exact_for_interactions():
    f = lambda df: (df["a"] * df["b"] + df["c"]).to_numpy()  # noqa: E731
    bg = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 3)), columns=list("abc"))
    x = pd.DataFrame([[1.0, 2.0, -1.0]], columns=list("abc"))
    exact, _, _ = shapley_values(f, x, bg)
    mc, base, pred = shapley_values(f, x, bg, max_exact=0, n_permutations=400)
    np.testing.assert_allclose(mc, exact, atol=0.05)
    assert mc.sum() == pytest.approx(pred - base)


def test_bundle_explanation_is_efficient(bundle):
    row = prepare(bundle, pd.DataFrame([EXAMPLE]))
    table = explain_patient(bundle, row)
    assert set(table["feature"]) == set(bundle.features)
    assert table["contribution"].sum() == pytest.approx(
        table.attrs["probability"] - table.attrs["base_value"]
    )
    assert table.attrs["probability"] == pytest.approx(bundle.predict_proba(row)[0])


def test_permutation_importance_ranks_st_slope_high(heart, cfg):
    X, y, spec = heart
    imp = permutation_importance_cv(build_model("logreg", spec, cfg), X, y, n_splits=3, n_repeats=3)
    assert set(imp["feature"]) == set(X.columns)
    assert "ST_Slope" in imp["feature"].head(3).tolist()
