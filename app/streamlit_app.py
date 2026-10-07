"""Streamlit front-end: single-patient risk + explanation, and batch scoring of a CSV.

streamlit run app/streamlit_app.py            # uses models/heartrisk.joblib
HEARTRISK_BUNDLE=path/to.joblib streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from heartrisk.bundle import INTENDED_USE, ModelBundle
from heartrisk.explain import explain_patient
from heartrisk.predict import predict_frame, prepare
from heartrisk.schema import EXAMPLE

st.set_page_config(page_title="heartrisk", layout="wide")


@st.cache_resource
def load_bundle(path: str) -> ModelBundle:
    return ModelBundle.load(path)


path = os.environ.get("HEARTRISK_BUNDLE", "models/heartrisk.joblib")
try:
    bundle = load_bundle(path)
except FileNotFoundError:
    st.error(f"No model bundle at `{path}`. Run `heartrisk train` first.")
    st.stop()

st.title("Coronary artery disease risk")
st.caption(INTENDED_USE)

single, batch, card = st.tabs(["Single patient", "Batch (CSV)", "Model card"])

with single:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 18, 100, EXAMPLE["Age"])
        sex = st.selectbox("Sex", ["M", "F"])
        cp = st.selectbox(
            "Chest pain type",
            ["ASY", "NAP", "ATA", "TA"],
            help="ASY asymptomatic, NAP non-anginal, ATA atypical angina, TA typical angina",
        )
        ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    with c2:
        bp_known = st.checkbox("Resting BP measured", True)
        bp = st.number_input("Resting BP (mmHg)", 60, 250, EXAMPLE["RestingBP"], disabled=not bp_known)
        chol_known = st.checkbox("Cholesterol measured", True)
        chol = st.number_input(
            "Cholesterol (mg/dL)", 80, 700, EXAMPLE["Cholesterol"], disabled=not chol_known
        )
        fbs = st.radio("Fasting blood sugar > 120 mg/dL", [0, 1], horizontal=True)
    with c3:
        maxhr = st.number_input("Max heart rate (bpm)", 50, 230, EXAMPLE["MaxHR"])
        angina = st.radio("Exercise-induced angina", ["N", "Y"], horizontal=True)
        oldpeak = st.number_input(
            "ST depression (Oldpeak, mm)", -3.0, 7.0, float(EXAMPLE["Oldpeak"]), step=0.1
        )
        slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    record = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingBP": bp if bp_known else None,
        "Cholesterol": chol if chol_known else None,
        "FastingBS": fbs,
        "RestingECG": ecg,
        "MaxHR": maxhr,
        "ExerciseAngina": angina,
        "Oldpeak": oldpeak,
        "ST_Slope": slope,
    }
    df = pd.DataFrame([record])
    res = predict_frame(bundle, df).iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted probability", f"{res['probability']:.1%}")
    m2.metric("Risk band", res["risk_band"])
    m3.metric(
        "Classification",
        "positive" if res["prediction"] else "negative",
        help=f"positive if probability ≥ {bundle.threshold:.2f}",
    )
    for w in res["warnings"]:
        st.warning(w)

    table = explain_patient(bundle, prepare(bundle, df))
    st.subheader("Why? (exact Shapley values)")
    st.caption(
        f"Average predicted probability in the reference sample: {table.attrs['base_value']:.1%}. "
        "Bars show how each variable moves this patient away from it (percentage points)."
    )
    chart = table.assign(
        label=lambda d: d["feature"] + " = " + d["value"].astype(str),
        points=lambda d: 100 * d["contribution"],
    ).set_index("label")["points"]
    st.bar_chart(chart, horizontal=True)

with batch:
    st.write("Upload a CSV with the columns:", ", ".join(bundle.features))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        try:
            out = pd.concat([data, predict_frame(bundle, data)], axis=1)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
        except ValueError as exc:
            st.error(str(exc))

with card:
    st.json(bundle.card())
