"""End-to-end checks on the real Kaggle files (skipped when data/raw is empty)."""

import numpy as np
import pytest

from src.config import N_TEST_ROWS
from src.data import load_test, load_train, to_price
from src.evaluation import cross_validate
from src.models import build_model
from tests.conftest import requires_kaggle_data

pytestmark = [pytest.mark.integration, requires_kaggle_data]


def test_target_is_log_transformed_exactly_once():
    _, y = load_train()
    assert 10 < y.min() and y.max() < 14          # log1p(dollars); a double log would be ~2.5


def test_every_test_house_gets_a_price():
    X, y = load_train()
    X_test, ids = load_test()
    model = build_model("ridge").fit(X, y)
    prices = to_price(model.predict(X_test))
    assert len(prices) == len(ids) == N_TEST_ROWS
    assert np.isfinite(prices).all() and 30_000 < np.median(prices) < 400_000


def test_ridge_beats_previous_pipeline():
    X, y = load_train()
    assert cross_validate(build_model("ridge"), X, y).mean < 0.12
