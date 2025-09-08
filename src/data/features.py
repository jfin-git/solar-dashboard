
import argparse, yaml, numpy as np, pandas as pd
from math import pi
from src.utils.paths import data_dir, configs_dir

def cyc(series, period):
    angle = 2 * pi * (series % period) / period
    return np.sin(angle), np.cos(angle)

def build(site_name: str, cfg_name="train.yaml"):
    cfg = yaml.safe_load(open(configs_dir(cfg_name)))
    df = pd.read_parquet(data_dir("raw", f"{site_name}.parquet")).sort_index()
    df["hour"] = df.index.hour
    df["doy"] = df.index.dayofyear
    if cfg["features"].get("cyclical_time", True):
        df["hour_sin"], df["hour_cos"] = cyc(df["hour"], 24)
        df["doy_sin"], df["doy_cos"] = cyc(df["doy"], 365)
    for lag in cfg["features"]["lags"]:
        for col in ["cloud","temp","wind","pv_kw"]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for roll in cfg["features"]["rolls"]:
        w = roll["window"]
        for stat in roll["stats"]:
            for col in ["cloud","temp","wind","pv_kw"]:
                df[f"{col}_r{w}_{stat}"] = getattr(df[col].rolling(w), stat)()
    df["y"] = df["pv_kw"]
    df = df.dropna()
    out = data_dir("processed", f"{site_name}_features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote processed features -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site_name", required=True)
    args = ap.parse_args()
    build(args.site_name)
