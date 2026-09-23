import numpy as np
import pandas as pd
import pytest

from src.features import HouseFeatureEngineer, SkewCorrector, build_preprocessor


def test_missing_means_absence(raw_train):
    out = HouseFeatureEngineer().fit(raw_train).transform(raw_train)
    no_fireplace = raw_train["FireplaceQu"].isna()
    assert (out.loc[no_fireplace, "FireplaceQu"] == 0).all()   # "None" -> ordinal 0, not the mode
    no_garage = raw_train["GarageType"].isna()
    assert (out.loc[no_garage, "GarageType"] == "None").all()
    assert (out.loc[no_garage, "GarageArea"] == 0).all()
    assert (out.loc[no_garage, "GarageYrBlt"] == raw_train.loc[no_garage, "YearBuilt"]).all()


def test_garage_year_typo_is_fixed(raw_train):
    raw = raw_train.copy()
    raw.loc[0, ["GarageYrBlt", "GarageType", "GarageArea"]] = [2207, "Attchd", 400]
    out = HouseFeatureEngineer().fit(raw).transform(raw)
    assert out.loc[0, "GarageYrBlt"] == raw.loc[0, "YearBuilt"]


def test_lot_frontage_uses_statistics_learned_at_fit(raw_train):
    engineer = HouseFeatureEngineer().fit(raw_train)
    new_house = raw_train.iloc[[0]].copy()
    new_house["LotFrontage"] = np.nan
    expected = raw_train.loc[raw_train.Neighborhood == new_house.Neighborhood.iloc[0], "LotFrontage"].median()
    assert engineer.transform(new_house)["LotFrontage"].iloc[0] == pytest.approx(expected)


def test_engineered_features(raw_train):
    out = HouseFeatureEngineer().fit(raw_train).transform(raw_train)
    expected_sf = raw_train["TotalBsmtSF"] + raw_train["1stFlrSF"] + raw_train["2ndFlrSF"]
    assert np.allclose(out["TotalSF"], expected_sf)
    assert (out["HouseAge"] >= 0).all()
    assert out["MSSubClass"].dtype == object
    assert "Utilities" not in out.columns


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_train_and_test_get_identical_columns(raw_train, raw_test, kind):
    """The old pipeline encoded test with its own OneHotEncoder; this must never diverge."""
    raw_test = raw_test.copy()
    raw_test["Heating"] = "GasA"                 # a category level disappears in test
    raw_test.loc[:5, "MSZoning"] = "C (all)"     # an unseen level appears in test
    raw_test.loc[:3, "KitchenQual"] = np.nan     # missing value only in test
    prep = build_preprocessor(kind).fit(raw_train)
    X_train, X_test = prep.transform(raw_train), prep.transform(raw_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert not np.isnan(X_train).any() and not np.isnan(X_test).any()


def test_heating_encoding_is_consistent(raw_train):
    prep = build_preprocessor("tree").fit(raw_train)
    names = list(prep.named_steps["columns"].get_feature_names_out())
    gas_a = pd.DataFrame(prep.transform(raw_train.assign(Heating="GasA")), columns=names)
    assert (gas_a["Heating_GasA"] == 1).all()


def test_skew_corrector_learns_mask_on_fit():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.lognormal(0, 1, 500), rng.normal(0, 1, 500)])
    corrector = SkewCorrector().fit(X)
    assert corrector.mask_.tolist() == [True, False]   # negative values are never logged
    assert np.allclose(corrector.transform(X)[:, 0], np.log1p(X[:, 0]))


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_preprocessor("deep")
