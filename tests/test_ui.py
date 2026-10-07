import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import ROOT  # noqa: E402


def test_streamlit_app_renders(bundle, tmp_path, monkeypatch):
    path = bundle.save(tmp_path / "ui.joblib")
    monkeypatch.setenv("HEARTRISK_BUNDLE", str(path))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Predicted probability" in labels and "Risk band" in labels
    at.checkbox[1].uncheck().run()  # cholesterol not measured -> imputed, warning shown
    assert not at.exception
    assert any("Cholesterol missing" in w.value for w in at.warning)
