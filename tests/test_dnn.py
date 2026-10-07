import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from heartrisk.dnn import TorchMLPClassifier  # noqa: E402

pytestmark = pytest.mark.torch


def test_fit_predict_pickle_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    a = TorchMLPClassifier(hidden=(8,), epochs=150, random_state=3).fit(X, y)
    b = TorchMLPClassifier(hidden=(8,), epochs=150, random_state=3).fit(X, y)
    pa = a.predict_proba(X)
    np.testing.assert_allclose(pa, b.predict_proba(X), atol=1e-6)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, pa[:, 1]) > 0.95
    np.testing.assert_allclose(pickle.loads(pickle.dumps(a)).predict_proba(X), pa)
