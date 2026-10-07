import json

import numpy as np
import pandas as pd
import pytest

from heartrisk.bundle import ModelBundle
from heartrisk.predict import predict_frame, prepare, risk_band
from heartrisk.schema import EXAMPLE


def test_bundle_roundtrip_and_card(bundle, tmp_path, heart):
    X, _, _ = heart
    path = bundle.save(tmp_path / "b.joblib")
    again = ModelBundle.load(path)
    np.testing.assert_allclose(again.predict_proba(X.head(20)), bundle.predict_proba(X.head(20)))
    card = json.loads((tmp_path / "b_card.json").read_text())
    assert card["model_key"] == "logreg" and card["metadata"]["data_sha256"]
    assert 0 < card["threshold"] < 1 and "intended_use" in card


def test_load_rejects_foreign_objects(tmp_path):
    import joblib

    joblib.dump({"not": "a bundle"}, tmp_path / "x.joblib")
    with pytest.raises(TypeError):
        ModelBundle.load(tmp_path / "x.joblib")


def test_predict_frame_and_warnings(bundle):
    rows = pd.DataFrame([EXAMPLE, {**EXAMPLE, "Cholesterol": 0, "Age": 95, "ChestPainType": "XYZ"}])
    out = predict_frame(bundle, rows)
    assert out["probability"].between(0, 1).all()
    assert (out["prediction"] == (out["probability"] >= bundle.threshold)).all()
    assert out.loc[0, "warnings"] == []
    w = " ".join(out.loc[1, "warnings"])
    assert "Cholesterol missing" in w and "Age=95" in w and "never seen" in w


def test_zero_cholesterol_equals_missing(bundle):
    a = prepare(bundle, pd.DataFrame([{**EXAMPLE, "Cholesterol": 0}]))
    b = prepare(bundle, pd.DataFrame([{**EXAMPLE, "Cholesterol": None}]))
    assert bundle.predict_proba(a)[0] == pytest.approx(bundle.predict_proba(b)[0])


def test_missing_column_raises(bundle):
    with pytest.raises(ValueError):
        predict_frame(bundle, pd.DataFrame([{k: v for k, v in EXAMPLE.items() if k != "Age"}]))


def test_risk_bands():
    bands = [[0.1, "low"], [0.3, "borderline"], [0.6, "elevated"], [1.01, "high"]]
    assert [risk_band(p, bands) for p in (0.05, 0.1, 0.5, 0.99)] == ["low", "borderline", "elevated", "high"]
