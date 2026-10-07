"""FastAPI service.

Run with ``heartrisk serve --bundle models/heartrisk.joblib`` or
``uvicorn heartrisk.api:app`` (bundle path from ``HEARTRISK_BUNDLE``).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .bundle import INTENDED_USE, ModelBundle
from .explain import explain_patient
from .predict import predict_frame, prepare
from .schema import Contribution, Explanation, ModelInfo, Patient, Prediction, PredictionResponse

log = logging.getLogger("heartrisk.api")
MAX_BATCH = 1000


def _frame(patients: list[Patient]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump() for p in patients])


def create_app(bundle: ModelBundle | None = None, bundle_path: str | None = None) -> FastAPI:
    state: dict = {"bundle": bundle}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["bundle"] is None:
            path = bundle_path or os.environ.get("HEARTRISK_BUNDLE", "models/heartrisk.joblib")
            try:
                state["bundle"] = ModelBundle.load(path)
                log.info("loaded model bundle %s", path)
            except FileNotFoundError:
                log.warning("no model bundle at %s; run `heartrisk train` first", path)
        yield

    app = FastAPI(
        title="heartrisk",
        version=__version__,
        description="Coronary artery disease risk estimation. " + INTENDED_USE,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    def get_bundle() -> ModelBundle:
        if state["bundle"] is None:
            raise HTTPException(503, "model bundle not loaded; run `heartrisk train` and restart")
        return state["bundle"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["bundle"] is not None, "version": __version__}

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        b = get_bundle()
        return ModelInfo(
            model=b.model_label,
            model_version=b.version,
            threshold=b.threshold,
            threshold_strategy=b.threshold_strategy,
            calibration=b.calibration,
            trained_on={k: b.metadata.get(k) for k in ("data_file", "data_sha256", "n_train", "prevalence")},
            cv_performance=b.performance,
            risk_bands=b.risk_bands,
            features=b.features,
            intended_use=INTENDED_USE,
        )

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(patients: Patient | list[Patient]) -> PredictionResponse:
        b = get_bundle()
        items = patients if isinstance(patients, list) else [patients]
        if not 1 <= len(items) <= MAX_BATCH:
            raise HTTPException(422, f"send between 1 and {MAX_BATCH} patients")
        out = predict_frame(b, _frame(items))
        results = [Prediction(**r) for r in out.to_dict(orient="records")]
        return PredictionResponse(model=b.model_label, model_version=b.version, results=results)

    @app.post("/v1/explain", response_model=Explanation)
    def explain(patient: Patient) -> Explanation:
        b = get_bundle()
        row = prepare(b, _frame([patient]))
        table = explain_patient(b, row)
        contribs = [
            Contribution(
                feature=r.feature,
                value=None if pd.isna(r.value) else r.value,
                contribution=float(r.contribution),
            )
            for r in table.itertuples()
        ]
        return Explanation(
            model=b.model_label,
            probability=table.attrs["probability"],
            base_value=table.attrs["base_value"],
            contributions=contribs,
        )

    return app


app = create_app()
