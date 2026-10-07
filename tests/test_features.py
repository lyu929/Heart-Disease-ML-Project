import numpy as np
import pandas as pd

from heartrisk.data import get_spec
from heartrisk.features import ZeroAsMissing, build_preprocessor


def test_zero_as_missing_is_stateless_and_targeted():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 2.0]})
    out = ZeroAsMissing(("a",)).fit(df).transform(df)
    assert np.isnan(out.loc[0, "a"]) and out.loc[0, "b"] == 0
    assert df.loc[0, "a"] == 0  # input untouched


def test_imputation_statistics_come_from_training_rows_only(heart, cfg):
    X, y, spec = heart
    prep = build_preprocessor(spec, cfg).fit(X.iloc[:300])
    imputer = prep.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    chol = X["Cholesterol"].iloc[:300]
    assert imputer.statistics_[spec.numeric.index("Cholesterol")] == chol[chol > 0].median()


def test_missing_indicator_and_unknown_categories(heart, cfg):
    X, y, spec = heart
    prep = build_preprocessor(spec, cfg).fit(X)
    names = list(prep.get_feature_names_out())
    assert "missingindicator_Cholesterol" in names
    row = X.iloc[[0]].copy()
    row["ChestPainType"] = "UNSEEN"
    out = prep.transform(row)
    assert out.shape[1] == len(names) and np.isfinite(out).all()


def test_indicator_can_be_switched_off(heart, cfg):
    X, y, spec = heart
    c = {**cfg, "cleaning": {**cfg["cleaning"], "missing_indicator": False}}
    names = build_preprocessor(spec, c).fit(X).get_feature_names_out()
    assert not any("missingindicator" in n for n in names)


def test_cleveland_spec_has_no_zero_rule():
    assert "Cholesterol" not in get_spec("cleveland").features
