
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from src.utils.paths import data_dir, models_dir
from src.models.metrics import daylight_mask, mae, rmse, mape

def train_naive(df):
    y = df["y"].values
    yhat = pd.Series(y).shift(24).bfill().values
    return yhat

def train_xgb(df, features):
    X = df[features].values
    y = df["y"].values
    model = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=7
    )
    model.fit(X, y)
    yhat = model.predict(X)
    return model, yhat

def main(site_name: str):
    df = pd.read_parquet(data_dir("processed", f"{site_name}_features.parquet")).sort_index()
    features = [c for c in df.columns if c not in {"y","pv_kw"}]
    yhat_naive = train_naive(df)
    mask = daylight_mask(df["y"].values, threshold=5.0)
    print("[Naive] MAE=%.2f RMSE=%.2f MAPE=%.1f%%" % (
        mae(df["y"].values, yhat_naive, mask),
        rmse(df["y"].values, yhat_naive, mask),
        mape(df["y"].values, yhat_naive, mask)
    ))
    model, yhat_xgb = train_xgb(df, features)
    print("[XGB]   MAE=%.2f RMSE=%.2f MAPE=%.1f%%" % (
        mae(df["y"].values, yhat_xgb, mask),
        rmse(df["y"].values, yhat_xgb, mask),
        mape(df["y"].values, yhat_xgb, mask)
    ))
    out_pred = data_dir("processed", f"{site_name}_baseline_preds.parquet")
    pd.DataFrame({"y": df["y"], "naive": yhat_naive, "xgb": yhat_xgb}, index=df.index).to_parquet(out_pred)
    print(f"Wrote baseline predictions -> {out_pred}")
    models_dir().mkdir(parents=True, exist_ok=True)
    (models_dir("xgb_features.txt")).write_text("\n".join(features))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site_name", required=True)
    args = ap.parse_args()
    main(args.site_name)
