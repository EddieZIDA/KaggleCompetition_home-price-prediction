"""Predict prices for new houses with the saved ensemble.

    python -m src.predict data/raw/test.csv --out submissions/predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import ID_COL, MODELS_DIR, TARGET


def load_ensemble(path: Path = MODELS_DIR / "final_ensemble.joblib"):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.pipeline` first.")
    return joblib.load(path)


def predict_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    houses = pd.read_csv(input_path)
    ids = houses[ID_COL] if ID_COL in houses.columns else pd.Series(range(len(houses)), name=ID_COL)
    features = houses.drop(columns=[c for c in (ID_COL, TARGET) if c in houses.columns])
    predictions = pd.DataFrame({ID_COL: ids, TARGET: load_ensemble().predict_price(features)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="CSV with the Kaggle raw columns")
    parser.add_argument("--out", type=Path, default=Path("submissions/predictions.csv"))
    args = parser.parse_args()
    predictions = predict_file(args.input, args.out)
    print(f"{len(predictions)} predictions written to {args.out}")


if __name__ == "__main__":
    main()
