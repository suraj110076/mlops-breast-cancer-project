from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_PATH = DATA_DIR / "raw.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
RANDOM_STATE = 42
TARGET_COLUMN = "target"
MLFLOW_EXPERIMENT = "breast-cancer-classifier"
