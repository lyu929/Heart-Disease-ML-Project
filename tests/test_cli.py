import json

import pandas as pd

from heartrisk.cli import main
from heartrisk.schema import EXAMPLE
from tests.conftest import DATA


def test_train_predict_explain_info(tmp_path, capsys):
    bundle = tmp_path / "m.joblib"
    assert main(["train", "--quick", "--set", f"data_dir={DATA}", "--out", str(bundle)]) == 0
    assert bundle.exists()
    csv = tmp_path / "p.csv"
    pd.DataFrame([EXAMPLE, {**EXAMPLE, "Age": 40, "ST_Slope": "Up"}]).to_csv(csv, index=False)
    out = tmp_path / "pred.csv"
    assert main(["predict", str(csv), "--bundle", str(bundle), "-o", str(out)]) == 0
    assert pd.read_csv(out)["probability"].between(0, 1).all()
    js = tmp_path / "one.json"
    js.write_text(json.dumps(EXAMPLE))
    assert main(["explain", str(js), "--bundle", str(bundle)]) == 0
    assert main(["info", "--bundle", str(bundle)]) == 0
    assert "logreg" in capsys.readouterr().out


def test_evaluate_command(tmp_path, capsys):
    rc = main(
        [
            "evaluate",
            "--quick",
            "--set",
            f"data_dir={DATA}",
            "--models",
            "logreg,xgb",
            "-j",
            "1",
            "--out",
            str(tmp_path / "ev"),
        ]
    )
    assert rc == 0 and (tmp_path / "ev" / "folds.csv").exists()
    assert "Paired corrected t-test" in capsys.readouterr().out
