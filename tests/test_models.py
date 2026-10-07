import importlib.util

import numpy as np
import pytest
from sklearn.base import clone

from heartrisk.models import MODELS, build_model, label, param_grid

KEYS = [k for k in MODELS if k != "dnn" or importlib.util.find_spec("torch")]


@pytest.mark.parametrize("key", KEYS)
def test_every_model_fits_and_predicts_probabilities(key, heart, cfg):
    X, y, spec = heart
    idx = np.random.default_rng(0).choice(len(y), 300, replace=False)
    c = {**cfg, "dnn": {**cfg.get("dnn", {}), "epochs": 5}}
    est = build_model(key, spec, c).fit(X.iloc[idx], y.iloc[idx])
    p = est.predict_proba(X.iloc[:50])[:, 1]
    assert p.shape == (50,) and np.all((p >= 0) & (p <= 1))
    clone(est)  # must stay cloneable for CV


def test_grid_keys_are_valid_parameters(heart, cfg):
    _, _, spec = heart
    for key in ("logreg", "rf", "xgb", "hgb", "mlp"):
        for c in (cfg, {**cfg, "wrap_calibration": "sigmoid"}):
            est = build_model(key, spec, c)
            params = est.get_params()
            assert all(k in params for k in param_grid(key, c)), key


def test_labels():
    assert label("logreg") == "Logistic regression"
    assert label("logreg + sigmoid") == "Logistic regression + sigmoid"
    assert label("unknown") == "unknown"
    with pytest.raises(KeyError):
        build_model("nope", None, {})
