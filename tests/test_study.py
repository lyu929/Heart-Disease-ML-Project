import pytest

from heartrisk.study import md_table, run_study


def test_md_table_formats_numbers():
    import pandas as pd

    t = md_table(pd.DataFrame({"a": [1.23456, float("nan")], "b": ["x", "y"]}))
    assert "1.235" in t and t.count("\n") == 3


@pytest.mark.slow
def test_quick_study_writes_report(cfg, tmp_path):
    path = run_study(
        cfg, tmp_path / "reports", tmp_path / "results", tmp_path / "m.joblib", n_jobs=1, log=lambda *_: None
    )
    text = path.read_text()
    for section in (
        "## 1. Data",
        "### 3.4 Leakage audit",
        "### 3.6 Leave-one-hospital-out",
        "## 4. Deployed model",
    ):
        assert section in text
    assert (tmp_path / "reports" / "figures" / "site_validation.png").exists()
    assert (tmp_path / "m.joblib").exists()
