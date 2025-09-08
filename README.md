
# Solar Generation Forecasting Dashboard (PyTorch + Streamlit)

A portfolio-ready project that predicts day-ahead PV generation for a site in Massachusetts and visualizes predictions, uncertainty, and errors in a Streamlit dashboard.

## Quickstart

> **Prereqs:** Python 3.10+ recommended. Install **PyTorch** first following the official instructions: https://pytorch.org/get-started/locally/

```bash
git clone <your-fork-url> solar-dashboard
cd solar-dashboard
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt

# Generate synthetic data (so it runs end-to-end immediately)
python -m src.data.synthesize_pv --days 120 --site_name "boston_demo"

# Build features
python -m src.data.features --site_name "boston_demo"

# Train baselines
python -m src.models.baselines --site_name "boston_demo"

# Train PyTorch GRU
python -m src.models.torch_gru --site_name "boston_demo"

# Launch dashboard
streamlit run src/app/app.py
```

## Repo layout

```
solar-dashboard/
  data/ (raw, interim, processed)
  models/ (artifacts)
  src/
    data/ (ingest, synth, features)
    models/ (baselines, torch, metrics)
    app/ (Streamlit)
    utils/
  configs/
  tests/
```

Notes:
- Synthetic data is for development; swap in real PV + forecast weather for production.
- Metrics computed for daylight hours by default.
