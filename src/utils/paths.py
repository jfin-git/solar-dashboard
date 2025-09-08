
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
def data_dir(*parts): return BASE / "data" / Path(*parts)
def models_dir(*parts): return BASE / "models" / Path(*parts)
def configs_dir(*parts): return BASE / "configs" / Path(*parts)
