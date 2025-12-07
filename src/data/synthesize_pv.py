
import argparse
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from math import sin, pi
from src.utils.paths import data_dir
from datetime import timedelta

def clearsky_profile(ts, lat, lon):
    # Simple bell via solar elevation proxy between sunrise and sunset
    date = ts.date()
    city = LocationInfo(latitude=lat, longitude=lon)
    s = sun(city.observer, date=date, tzinfo=None)
    if ts < s['sunrise'] or ts > s['sunset']:
        return 0.0
    daylen = (s['sunset'] - s['sunrise']).total_seconds()
    x = (ts - s['sunrise']).total_seconds() / daylen
    return max(0.0, sin(pi * x))

def generate(days=120, lat=42.36, lon=-71.06, dc_kw=50.0, seed=7):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.utcnow().floor("h").tz_convert("UTC")
    idx = pd.date_range(end=end, periods=24*days, freq="1h")
    df = pd.DataFrame(index=idx)
    df["cloud"] = np.clip(rng.normal(0.5, 0.25, len(df)), 0, 1)
    df["temp"] = 10 + 10*np.sin(2*pi*df.index.dayofyear/365.0) + rng.normal(0, 3, len(df))
    df["wind"] = np.abs(rng.normal(3.0, 1.0, len(df)))
    cs = np.array([clearsky_profile(ts.to_pydatetime(), lat, lon) for ts in df.index])
    effective = cs * (0.6 + 0.4*(1 - df["cloud"])) * (1 + rng.normal(0, 0.05, len(df)))
    df["pv_kw"] = np.clip(effective * dc_kw, 0, dc_kw)
    df["cloud_fcst"] = np.clip(df["cloud"].shift(1).bfill() + rng.normal(0, 0.1, len(df)), 0, 1)
    df["temp_fcst"] = df["temp"].shift(1).bfill() + rng.normal(0, 0.5, len(df))
    df["wind_fcst"] = np.abs(df["wind"].shift(1).bfill() + rng.normal(0, 0.3, len(df)))
    return df

def main(args):
    df = generate(days=args.days)
    out = data_dir("raw", f"{args.site_name}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote synthetic raw data -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--site_name", type=str, required=True)
    args = ap.parse_args()
    main(args)
