import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Predictor")
st.write("Enter the physicochemical properties to predict quality (0–10).")

# Load model
model_path = Path('artifacts/model/model.pkl')
if not model_path.exists():
    st.error("Model not found. Please run the DVC pipeline (train stage) first.")
    st.stop()

model = joblib.load(model_path)

# Load feature names (saved during training)
feat_file = Path('artifacts/model/feature_names.json')
if feat_file.exists():
    feature_names = json.loads(feat_file.read_text())
else:
    # Fallback: infer from raw CSV
    raw = pd.read_csv('data/raw/winequality-red.csv')
    feature_names = [c for c in raw.columns if c != 'quality']

# Build inputs
raw_df = pd.read_csv('data/raw/winequality-red.csv')
ranges = {}
for col in feature_names:
    col_min = float(raw_df[col].min())
    col_max = float(raw_df[col].max())
    col_mean = float(raw_df[col].mean())
    ranges[col] = (col_min, col_max, col_mean)

st.subheader("Input Features")
input_vals = {}
for col in feature_names:
    cmin, cmax, cmean = ranges[col]
    input_vals[col] = st.slider(
        col, min_value=round(cmin, 3), max_value=round(cmax, 3),
        value=round(cmean, 3), step=0.001
    )

if st.button("Predict Quality"):
    X = pd.DataFrame([input_vals], columns=feature_names)
    pred = model.predict(X)[0]
    st.success(f"Predicted quality: **{pred}**")