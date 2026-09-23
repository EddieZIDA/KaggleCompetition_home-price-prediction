"""Central configuration: paths, seeds and cross-validation settings."""

from pathlib import Path

RANDOM_SEED = 42

# Evaluation folds and tuning folds use different seeds so that Optuna does not
# overfit the exact splits used to report scores.
N_FOLDS = 5
CV_SEED = RANDOM_SEED
TUNING_CV_SEED = 2024

TARGET = "SalePrice"
ID_COL = "Id"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

MODELS_DIR = ROOT_DIR / "models"
PARAMS_DIR = MODELS_DIR / "params"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

N_TEST_ROWS = 1459
