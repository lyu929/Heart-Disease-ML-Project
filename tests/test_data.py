import numpy as np
import pandas as pd
import pytest

from heartrisk.data import DATASETS, clean, data_card, get_spec, infer_sites, load_dataset, read_raw


def test_heart_shapes_and_types(heart):
    X, y, spec = heart
    assert X.shape == (918, 11)
    assert list(X.columns) == spec.features
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.553) < 1e-3
    for c in spec.categorical:
        assert X[c].dtype == object
    # loading keeps raw values: zeros are handled inside the model pipeline
    assert (X["Cholesterol"] == 0).sum() == 172


def test_clean_turns_impossible_zeros_into_missing(heart):
    X, y, spec = heart
    c = clean(X.assign(HeartDisease=y), spec, ["Cholesterol", "RestingBP"])
    assert c["Cholesterol"].isna().sum() == 172
    assert c["RestingBP"].isna().sum() == 1
    assert (c["Cholesterol"].dropna() > 0).all()


def test_infer_sites_matches_published_composition(heart):
    X, y, _ = heart
    sites = infer_sites(X.assign(HeartDisease=y))
    counts = sites.value_counts().to_dict()
    assert counts == {"cleveland": 302, "hungarian": 293, "va_long_beach": 200, "switzerland": 123}
    assert (X.loc[sites == "switzerland", "Cholesterol"] == 0).all()


def test_infer_sites_rejects_shuffled_file(heart):
    X, y, _ = heart
    df = X.assign(HeartDisease=y).sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError):
        infer_sites(df)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_registered_dataset_loads(name):
    from tests.conftest import DATA

    X, y, spec = load_dataset(name, DATA)
    assert len(X) == len(y) > 100
    assert X.columns.tolist() == spec.features
    assert y.isin([0, 1]).all()


def test_unknown_dataset_and_missing_file(tmp_path):
    with pytest.raises(KeyError):
        get_spec("nope")
    with pytest.raises(FileNotFoundError):
        read_raw("heart", tmp_path)


def test_data_card(heart):
    X, y, _ = heart
    card = data_card(X, y)
    assert card["feature"].tolist()[-1] == "HeartDisease"
    assert set(card["type"]) == {"numeric", "categorical", "target"}
    assert isinstance(card, pd.DataFrame) and np.all(card["missing"] >= 0)
