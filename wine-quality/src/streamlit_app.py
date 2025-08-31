import json
import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")
st.title("🍷 Wine Quality Predictor")

MODEL_PATH = Path("artifacts/model/model.pkl")
FEATS_PATH = Path("artifacts/model/feature_names.json")
RAW_PATH   = Path("data/raw/winequality-red.csv")
CM_PATH    = Path("artifacts/metrics/confusion_matrix.png")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found at artifacts/model/model.pkl. "
            "Run the pipeline first: `dvc repro`."
        )
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_raw():
    if not RAW_PATH.exists():
        raise FileNotFoundError("Raw CSV not found at data/raw/winequality-red.csv.")
    return pd.read_csv(RAW_PATH)

@st.cache_data
def get_feature_names():
    if FEATS_PATH.exists():
        return json.loads(FEATS_PATH.read_text())
    df = load_raw()
    return [c for c in df.columns if c != "quality"]

def probabilities_frame(model, X):
    try:
        proba = model.predict_proba(X)
        cols = [f"class_{c}" for c in model.classes_] if hasattr(model, "classes_") else [f"class_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, columns=cols)
    except Exception:
        return None

# --- UI ---
try:
    model = load_model()
    feature_names = get_feature_names()
    raw_df = load_raw()
except Exception as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch CSV", "Artifacts"])

with tab1:
    st.subheader("Single Prediction (sliders)")
    # Build sliders from data ranges
    ranges = {}
    for col in feature_names:
        cmin = float(raw_df[col].min())
        cmax = float(raw_df[col].max())
        cmean = float(raw_df[col].mean())
        ranges[col] = (cmin, cmax, cmean)

    cols_ui = st.columns(2)
    inputs = {}
    for i, col in enumerate(feature_names):
        cmin, cmax, cmean = ranges[col]
        with cols_ui[i % 2]:
            inputs[col] = st.slider(
                col, min_value=round(cmin, 3), max_value=round(cmax, 3),
                value=round(cmean, 3), step=0.001
            )

    if st.button("Predict"):
        X = pd.DataFrame([inputs], columns=feature_names)
        pred = model.predict(X)[0]
        st.success(f"Predicted quality: **{pred}**")
        proba_df = probabilities_frame(model, X)
        if proba_df is not None:
            st.caption("Class probabilities")
            st.dataframe(proba_df.style.format("{:.3f}"))

with tab2:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV that contains **only** these columns (in any order):")
    st.code(", ".join(feature_names))

    # Offer a template download
    template = pd.DataFrame([raw_df[feature_names].mean(numeric_only=True)])
    csv_bytes = template.to_csv(index=False).encode("utf-8")
    st.download_button("Download input template", csv_bytes, file_name="wine_input_template.csv", mime="text/csv")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in feature_names if c not in df.columns]
            extra = [c for c in df.columns if c not in feature_names]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                # Keep only known features (ignore extra columns)
                X = df[feature_names].copy()
                preds = model.predict(X)
                out = df.copy()
                out["pred_quality"] = preds
                st.success(f"Predicted {len(out)} rows.")
                st.dataframe(out.head(20))

                proba_df = probabilities_frame(model, X)
                if proba_df is not None:
                    st.caption("Class probabilities (first 20 rows shown)")
                    st.dataframe(proba_df.head(20).style.format("{:.3f}"))

                # Download results
                out_csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", out_csv, file_name="predictions.csv", mime="text/csv")

        except Exception as ex:
            st.error(f"Failed to process CSV: {ex}")

with tab3:
    st.subheader("Evaluation Artifacts")
    if CM_PATH.exists():
        st.image(str(CM_PATH), caption="Confusion Matrix (from evaluate stage)", use_column_width=True)
    else:
        st.info("No confusion matrix found. Run `dvc repro` to generate evaluation artifacts.")
    if Path("artifacts/metrics/metrics.json").exists():
        st.json(json.loads(Path("artifacts/metrics/metrics.json").read_text()))
    else:
        st.info("No metrics.json found yet.")
