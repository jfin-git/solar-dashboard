
.PHONY: setup synth features train_baselines train_torch app test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

synth:
	python -m src.data.synthesize_pv --days 120 --site_name "boston_demo"

features:
	python -m src.data.features --site_name "boston_demo"

train_baselines:
	python -m src.models.baselines --site_name "boston_demo"

train_torch:
	python -m src.models.torch_gru --site_name "boston_demo"

app:
	streamlit run src/app/app.py

test:
	pytest -q
