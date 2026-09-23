import numpy as np
import pandas as pd
import pytest

from src.data import partial_sale_mansions, remove_outliers, to_price
from src.ensemble import EnsembleRegressor, blend_cv_scores, optimize_weights
from src.submission import InvalidSubmissionError, validate_submission


def _submission(prices, n=10):
    return pd.DataFrame({"Id": np.arange(1461, 1461 + n), "SalePrice": prices})


def test_valid_submission_passes():
    validate_submission(_submission(np.linspace(80_000, 400_000, 10)), expected_rows=10)


@pytest.mark.parametrize("prices, message", [
    (np.full(10, 11.7), "plausible"),          # left in log space: the old submission_blend.csv bug
    (np.full(10, np.nan), "missing"),
])
def test_bad_prices_are_rejected(prices, message):
    with pytest.raises(InvalidSubmissionError, match=message):
        validate_submission(_submission(prices), expected_rows=10)


def test_missing_rows_are_rejected():
    with pytest.raises(InvalidSubmissionError, match="rows"):
        validate_submission(_submission(np.full(8, 150_000.0), n=8), expected_rows=10)


def test_outliers_removed_only_when_huge_and_cheap():
    train = pd.DataFrame({"GrLivArea": [4500, 4500, 1500], "SalePrice": [160_000, 700_000, 160_000]})
    assert remove_outliers(train)["SalePrice"].tolist() == [700_000, 160_000]


def test_log_round_trip():
    prices = np.array([50_000.0, 200_000.0])
    assert np.allclose(to_price(np.log1p(prices)), prices)


def test_optimize_weights_prefers_the_accurate_model():
    rng = np.random.default_rng(0)
    y = rng.normal(12, 0.4, 300)
    oof = np.column_stack([y + rng.normal(0, 0.02, 300), y + rng.normal(0, 0.3, 300)])
    weights = optimize_weights(oof, y)
    assert weights.sum() == pytest.approx(1) and (weights >= 0).all()
    assert weights[0] > 0.9
    assert blend_cv_scores(oof, y).mean() < 0.03


class _Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_blend_ensemble_predicts_weighted_log_price():
    models = {"a": _Constant(12.0), "b": _Constant(11.0)}
    ensemble = EnsembleRegressor("blend", models, {"a": 0.75, "b": 0.25})
    X = pd.DataFrame({"x": [0, 1]})
    assert np.allclose(ensemble.predict(X), 11.75)
    assert np.allclose(ensemble.predict_price(X), np.expm1(11.75))


def test_partial_sale_mansions_rule():
    houses = pd.DataFrame({
        "GrLivArea": [5000, 5000, 5000, 1500],
        "Neighborhood": ["Edwards", "NoRidge", "Edwards", "Edwards"],
        "SaleCondition": ["Partial", "Partial", "Normal", "Partial"],
    })
    assert partial_sale_mansions(houses).tolist() == [True, False, False, False]
    assert not partial_sale_mansions(pd.DataFrame({"x": [1]})).any()   # missing columns: no match


def test_mansion_override_only_touches_matching_rows():
    ensemble = EnsembleRegressor("blend", {"a": _Constant(13.5)}, {"a": 1.0}, mansion_log_price=12.0)
    X = pd.DataFrame({"GrLivArea": [5000, 5000], "Neighborhood": ["Edwards", "NoRidge"],
                      "SaleCondition": ["Partial", "Partial"]})
    assert np.allclose(ensemble.predict(X), [12.0, 13.5])
