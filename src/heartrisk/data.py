"""Dataset registry, loading and cleaning.

Every cohort is described by a :class:`DatasetSpec` so the same evaluation code runs
on all of them. ``heart.csv`` is the primary dataset (918 patients, the Kaggle
"Heart Failure Prediction" merge of four UCI hospital cohorts). The two ``*_processed``
files were standardised on the full data before they were committed, so they are only
used for within-cohort benchmarking (see ``docs/data.md``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file: str
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    binary: tuple[str, ...] = ()
    outcome: str = ""
    description: str = ""
    pre_standardised: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.binary, *self.categorical]


DATASETS: dict[str, DatasetSpec] = {
    "heart": DatasetSpec(
        name="heart",
        file="heart.csv",
        target="HeartDisease",
        numeric=("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"),
        binary=("FastingBS",),
        categorical=("Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"),
        outcome="Angiographic coronary artery disease (>50% diameter narrowing)",
        description="Cleveland + Hungarian + Switzerland + VA Long Beach (UCI), duplicates removed",
        notes=(
            "Cholesterol==0 in 172 rows and RestingBP==0 in 1 row are missing values, not measurements.",
            "Row order follows the source hospitals; see infer_sites().",
        ),
    ),
    "cleveland": DatasetSpec(
        name="cleveland",
        file="cleveland_processed.csv",
        target="target",
        numeric=("age", "trestbps", "chol", "thalach", "oldpeak", "ca"),
        binary=("sex", "fbs", "exang"),
        categorical=("cp", "restecg", "slope", "thal"),
        outcome="Angiographic coronary artery disease",
        description="UCI Cleveland cohort with the extra 'ca' and 'thal' variables",
        pre_standardised=True,
        notes=(
            "Continuous columns were z-scored on all 303 rows before being committed.",
            "These patients are also contained in heart.csv.",
        ),
    ),
    "framingham": DatasetSpec(
        name="framingham",
        file="framingham_processed.csv",
        target="target",
        numeric=("age", "cigsPerDay", "chol", "trestbps", "diaBP", "BMI", "thalach", "glucose"),
        binary=("sex", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"),
        categorical=("education",),
        outcome="10-year coronary heart disease (prognostic, not diagnostic)",
        description="Framingham Heart Study teaching extract (4240 participants, 15% events)",
        pre_standardised=True,
        notes=(
            "Continuous columns were imputed and z-scored on all rows before being committed.",
            "Different outcome and population: never pooled with the diagnostic cohorts.",
        ),
    ),
}

# Published per-hospital composition of heart.csv (rows, positives) in file order.
# Used to recover the source hospital of every row; verified on load.
HEART_SITES: tuple[tuple[str, int, int], ...] = (
    ("hungarian", 293, 106),
    ("switzerland", 123, 115),
    ("va_long_beach", 200, 149),
    ("cleveland", 302, 138),
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from exc


def read_raw(name: str, data_dir: str | Path = "data") -> pd.DataFrame:
    spec = get_spec(name)
    path = Path(data_dir) / spec.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run from the repository root or set data_dir)")
    df = pd.read_csv(path)
    missing = [c for c in [*spec.features, spec.target] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def clean(
    df: pd.DataFrame, spec: DatasetSpec, zero_as_missing: list[str] | tuple[str, ...] = ()
) -> pd.DataFrame:
    """Return a cleaned copy: impossible zeros -> NaN, categoricals as strings, target as int."""
    out = df.copy()
    for col in zero_as_missing:
        if col in out.columns:
            out[col] = out[col].astype(float).mask(out[col] == 0)
    for col in spec.categorical:
        out[col] = out[col].astype("string").astype(object).where(out[col].notna(), np.nan)
    for col in (*spec.numeric, *spec.binary):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out[spec.target] = out[spec.target].astype(int)
    return out


def load_dataset(
    name: str = "heart", data_dir: str | Path = "data"
) -> tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    """Load ``name`` and return ``(X, y, spec)`` with types fixed but values untouched.

    Value-level cleaning (impossible zeros -> missing) happens inside the model pipeline
    (:class:`heartrisk.features.ZeroAsMissing`) so that training, CV folds and the API apply
    exactly the same rule and the rule itself can be ablated.
    """
    spec = get_spec(name)
    df = clean(read_raw(name, data_dir), spec)
    return df[spec.features], df[spec.target], spec


def infer_sites(df: pd.DataFrame) -> pd.Series:
    """Recover the source hospital of each heart.csv row.

    heart.csv concatenates the UCI cohorts in a fixed order. The block boundaries are
    taken from the published cohort sizes and *verified* against the published number
    of positives per cohort and the fact that the Switzerland cohort has no cholesterol
    values. Raises ``ValueError`` if the file does not match (e.g. it was shuffled).
    """
    target = "HeartDisease"
    if len(df) != sum(n for _, n, _ in HEART_SITES):
        raise ValueError("site inference only applies to the original 918-row heart.csv")
    labels, start = [], 0
    for site, n, positives in HEART_SITES:
        block = df.iloc[start : start + n]
        if int(block[target].sum()) != positives:
            raise ValueError(f"block for {site} does not match the published composition")
        labels += [site] * n
        start += n
    chol = df["Cholesterol"].fillna(0).to_numpy()
    swiss = np.asarray(labels) == "switzerland"
    if not (chol[swiss] == 0).all() or (chol[np.asarray(labels) == "cleveland"] == 0).any():
        raise ValueError("cholesterol pattern does not match the source cohorts")
    return pd.Series(labels, index=df.index, name="site")


def data_card(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-feature summary used in the report (type, missingness, range / levels)."""
    rows = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            desc = f"{s.min():.4g} – {s.max():.4g} (median {s.median():.4g})"
            kind = "numeric"
        else:
            vc = s.value_counts()
            desc = ", ".join(f"{k} ({v})" for k, v in vc.items())
            kind = "categorical"
        rows.append({"feature": col, "type": kind, "missing": int(s.isna().sum()), "values": desc})
    rows.append(
        {
            "feature": y.name,
            "type": "target",
            "missing": int(y.isna().sum()),
            "values": f"prevalence {y.mean():.3f} (n={len(y)})",
        }
    )
    return pd.DataFrame(rows)
