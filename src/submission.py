"""Kaggle submission writing with sanity checks."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import N_TEST_ROWS


class InvalidSubmissionError(ValueError):
    pass


def validate_submission(submission: pd.DataFrame, expected_rows: int = N_TEST_ROWS) -> None:
    """Raise if the file would be rejected by Kaggle or is obviously not in dollars."""
    problems = []
    if list(submission.columns) != ["Id", "SalePrice"]:
        problems.append(f"columns are {list(submission.columns)}, expected ['Id', 'SalePrice']")
    if len(submission) != expected_rows:
        problems.append(f"{len(submission)} rows, expected {expected_rows}")
    if submission["Id"].duplicated().any():
        problems.append("duplicated Id values")
    prices = submission["SalePrice"]
    if prices.isna().any() or not np.isfinite(prices).all():
        problems.append("missing or infinite prices")
    elif prices.min() < 10_000 or prices.max() > 2_000_000:
        # Catches predictions left in log space (~12) or a double expm1.
        problems.append(f"prices outside a plausible range: {prices.min():.2f} – {prices.max():.2f}")
    if problems:
        raise InvalidSubmissionError("; ".join(problems))


def write_submission(ids: pd.Series, prices: np.ndarray, path: Path) -> pd.DataFrame:
    submission = pd.DataFrame({"Id": ids.to_numpy(), "SalePrice": prices})
    validate_submission(submission)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission
