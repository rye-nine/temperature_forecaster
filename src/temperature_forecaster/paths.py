# src/temperature_forecaster/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# data
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
#models
FOURIER_MODELS = PROJECT_ROOT / "models" /  "fourier_models"
AUTOREGRESSION_MODELS = PROJECT_ROOT / "models" / "residual_autoregression_models"
