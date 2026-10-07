"""Self-describing model bundle: fitted pipeline + threshold + reference data + metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__

INTENDED_USE = (
    "Research and teaching demonstrator for estimating the probability of angiographic coronary "
    "artery disease from routine clinical variables. Not a medical device; not validated for "
    "clinical decision making."
)


def _versions() -> dict:
    import sklearn

    out = {
        "heartrisk": __version__,
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import xgboost

        out["xgboost"] = xgboost.__version__
    except ImportError:  # pragma: no cover
        pass
    return out


def git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


@dataclass
class ModelBundle:
    pipeline: object
    model_key: str
    model_label: str
    dataset: str
    features: list[str]
    threshold: float
    threshold_strategy: str
    calibration: str
    background: pd.DataFrame
    ranges: dict
    levels: dict
    risk_bands: list
    performance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        return f"{self.model_key}-{self.metadata.get('created', 'unknown')[:10]}"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def card(self) -> dict:
        """JSON-serialisable summary (also written next to the bundle as model_card.json)."""
        d = {k: v for k, v in asdict(self).items() if k not in ("pipeline", "background")}
        d["version"] = self.version
        d["intended_use"] = INTENDED_USE
        return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.metadata["versions"] = _versions()
        joblib.dump(self, path, compress=3)
        path.with_name(path.stem + "_card.json").write_text(json.dumps(self.card(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a heartrisk ModelBundle")
        saved = obj.metadata.get("versions", {}).get("scikit-learn")
        import sklearn

        if saved and saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            warnings.warn(
                f"bundle was trained with scikit-learn {saved}, running {sklearn.__version__}; "
                "retrain with `heartrisk train` if predictions look wrong",
                stacklevel=2,
            )
        return obj
