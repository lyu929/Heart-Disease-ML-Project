import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from heartrisk.api import create_app  # noqa: E402
from heartrisk.schema import EXAMPLE  # noqa: E402


@pytest.fixture(scope="module")
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


def test_health_and_model_info(client, bundle):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/v1/model").json()
    assert info["threshold"] == pytest.approx(bundle.threshold)
    assert info["features"] == bundle.features


def test_predict_single_and_batch(client):
    r = client.post("/v1/predict", json=EXAMPLE)
    assert r.status_code == 200 and "X-Process-Time-ms" in r.headers
    res = r.json()["results"]
    assert len(res) == 1 and 0 <= res[0]["probability"] <= 1
    batch = client.post("/v1/predict", json=[EXAMPLE, {**EXAMPLE, "Cholesterol": None}]).json()["results"]
    assert len(batch) == 2 and any("missing" in w for w in batch[1]["warnings"])


@pytest.mark.parametrize(
    "bad",
    [
        {**EXAMPLE, "Sex": "X"},
        {**EXAMPLE, "Cholesterol": 0},
        {**EXAMPLE, "Age": 5},
        {**EXAMPLE, "unexpected": 1},
        {k: v for k, v in EXAMPLE.items() if k != "ST_Slope"},
    ],
)
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/v1/predict", json=bad).status_code == 422


def test_explain(client):
    body = client.post("/v1/explain", json=EXAMPLE).json()
    total = sum(c["contribution"] for c in body["contributions"])
    assert total == pytest.approx(body["probability"] - body["base_value"], abs=1e-6)


def test_service_without_model_returns_503(tmp_path):
    with TestClient(create_app(bundle_path=str(tmp_path / "missing.joblib"))) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/v1/predict", json=EXAMPLE).status_code == 503
