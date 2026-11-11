"""Cross-validation: folds partition the listings, test rows are never dropped, results repeat."""

import numpy as np
import pandas as pd

from vehicle_pricing.evaluation import VARIANTS, assign_folds, cross_validate, summarize


def _variant(name):
    return next(v for v in VARIANTS if v.name == name)


NO_REMOVAL = "Mileage + age, no outlier removal"
THREE_PASSES = "Mileage + age, 3 passes"
WEEK_3 = "Week 3 style (mileage only, own year)"


def test_folds_partition_the_listings(camry_listings):
    n = len(camry_listings)

    result = cross_validate(camry_listings, _variant(NO_REMOVAL))

    assert result.index.equals(camry_listings.index)  # each listing once, in order
    assert result["fold"].isin(range(5)).all()
    sizes = result["fold"].value_counts()
    assert len(sizes) == 5
    assert sizes.min() >= n // 5 and sizes.max() <= n // 5 + 1  # about n / 5 each
    assert (result["fold"].to_numpy() == assign_folds(n, 5, 0)).all()


def test_test_rows_are_never_removed_even_with_planted_outliers(camry_listings):
    fake = camry_listings.iloc[[2, 11, 20, 29, 38]].assign(price=500.0)
    fake.index = fake.index + 1000  # distinct labels
    listings = pd.concat([camry_listings, fake])

    result = cross_validate(listings, _variant(THREE_PASSES))

    assert len(result) == len(listings)
    assert result.index.is_unique and set(result.index) == set(listings.index)
    assert (result.loc[fake.index, "price"] == 500).all()  # planted rows kept, unaltered
    assert result["predicted"].notna().all()  # and predicted like any other


def test_same_seed_gives_same_folds():
    first, again = assign_folds(44, 5, seed=0), assign_folds(44, 5, seed=0)

    assert (first == again).all()
    assert not (first == assign_folds(44, 5, seed=1)).all()  # another seed shuffles differently


def test_coverage_of_the_three_pass_model_on_clean_data(camry_listings):
    result = cross_validate(camry_listings, _variant(THREE_PASSES))

    summary = summarize(result, pd.Series(True, index=result.index))

    assert 0.85 <= summary.loc["all", "coverage"] <= 1.0
    assert summary.loc["all", "n_predicted"] == len(camry_listings)


def test_week_3_style_leaves_years_with_too_few_listings_empty(camry_listings):
    # The fixture has 4 listings per model year: far below MIN_LISTINGS for one year.
    result = cross_validate(camry_listings, _variant(WEEK_3))

    summary = summarize(result, pd.Series(True, index=result.index))

    assert result["predicted"].isna().all()
    assert summary.loc["all", "n_predicted"] == 0
    assert summary.loc["all", "n_listings"] == len(camry_listings)


def test_week_3_style_predicts_years_with_enough_listings():
    rng = np.random.default_rng(seed=4)
    n = 160  # two model years of 80: about 64 training listings of each per fold
    odometer = rng.uniform(10_000, 200_000, size=n)
    year = np.repeat([2015, 2016], n // 2)
    price = 20_000 * np.exp(-odometer / 120_000) * rng.uniform(0.9, 1.1, size=n)
    listings = pd.DataFrame({"year": year, "odometer": odometer, "price": price.round()})

    result = cross_validate(listings, _variant(WEEK_3))

    summary = summarize(result, pd.Series(True, index=result.index))
    assert result["predicted"].notna().all()
    assert 0.85 <= summary.loc["all", "coverage"] <= 1.0
    assert summary.loc["all", "median_abs_pct_error"] < 15
