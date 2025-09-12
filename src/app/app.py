import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st, pandas as pd, numpy as np
from src.utils.paths import data_dir


st.set_page_config(page_title="Solar Forecast Dashboard", layout="wide")
st.title("Solar Generation Forecasting (Demo)")

site = st.sidebar.text_input("Site name", "boston_demo")
feat_path = data_dir("processed", f"{site}_features.parquet")
base_pred_path = data_dir("processed", f"{site}_baseline_preds.parquet")
torch_pred_path = data_dir("processed", f"{site}_torch_preds.parquet")

if not feat_path.exists():
    st.warning("No processed features found. Run: synth → features → baselines/torch.")
    st.stop()

df = pd.read_parquet(feat_path)
st.caption(f"Rows: {len(df):,} | {df.index.min()} → {df.index.max()} (UTC)")

# Baseline preds: rename and DROP any columns that already exist in df (like 'y')
if base_pred_path.exists():
    preds = pd.read_parquet(base_pred_path).rename(columns={"naive": "naive_pred", "xgb": "xgb_pred"})
    overlap = preds.columns.intersection(df.columns)
    if len(overlap):
        preds = preds.drop(columns=list(overlap))
    df = df.join(preds, how="left")

# Torch preds (no overlap expected)
if torch_pred_path.exists():
    torch_preds = pd.read_parquet(torch_pred_path)
    overlap = torch_preds.columns.intersection(df.columns)
    if len(overlap):
        torch_preds = torch_preds.drop(columns=list(overlap))
    df = df.join(torch_preds, how="left")

st.subheader("Recent Performance")
days = st.slider("Last N days", 7, 60, 14)
recent = df.last(f"{days}D")
plot_df = recent[["y"]].rename(columns={"y":"Actual (kW)"})
if "xgb_pred" in recent: plot_df["XGB Pred (kW)"] = recent["xgb_pred"]
if "pred_p50_avg24h" in recent: plot_df["Torch P50 (avg next 24h)"] = recent["pred_p50_avg24h"]
st.line_chart(plot_df)

st.markdown("---")
st.subheader("Feature Explorer")
cands = [c for c in df.columns if c not in {"y","pv_kw","naive_pred","xgb_pred","pred_p50_avg24h"}]
sel = st.multiselect("Plot extra features", cands[:12], default=cands[:2], max_selections=4)
if sel:
    st.line_chart(recent[["y"] + sel])
else:
    st.info("Pick up to 4 features to overlay.")

st.markdown("---")
st.caption("Replace synthetic data with real PV + forecasted weather when ready. See README.")
