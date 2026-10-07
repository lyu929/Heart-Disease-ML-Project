from pathlib import Path

import pytest

from heartrisk.config import load_config
from heartrisk.data import load_dataset
from heartrisk.study import quick_config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def cfg():
    c = quick_config(load_config())
    c["data_dir"] = str(DATA)
    return c


@pytest.fixture(scope="session")
def heart():
    return load_dataset("heart", DATA)


@pytest.fixture(scope="session")
def bundle(cfg, tmp_path_factory):
    from heartrisk.bundle import ModelBundle
    from heartrisk.train import train_bundle

    b = train_bundle(cfg, "logreg", "sigmoid", log=lambda *_: None)
    path = b.save(tmp_path_factory.mktemp("bundle") / "heartrisk.joblib")
    return ModelBundle.load(path)
