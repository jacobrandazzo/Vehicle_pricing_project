"""The prediction interval is built correctly and depreciation goes the right way."""

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from vehicle_pricing.data import vehicle_age
from vehicle_pricing.model import (
    _half_width,
    car_model_year,
    depreciation_curves,
    drop_outliers,
    fit_model,
    predict_interval,
    predict_price,
    query_age,
    typical_mileage_range,
)
from vehicle_pricing.schemas import LATEST_MODEL_YEAR, VehicleQuery


def test_predicted_price_is_inside_its_interval(camry_model):
    query = VehicleQuery(manufacturer="toyota", model="camry", odometer=90_000)

    result = predict_price(camry_model, query)

    assert result.lower_bound <= result.predicted_price <= result.upper_bound


def test_more_miles_means_lower_predicted_price(camry_model):
    low_miles = VehicleQuery(manufacturer="toyota", model="camry", odometer=40_000)
    high_miles = low_miles.model_copy(update={"odometer": 150_000})

    low_result = predict_price(camry_model, low_miles)
    high_result = predict_price(camry_model, high_miles)

    assert high_result.predicted_price < low_result.predicted_price


def test_price_falls_with_age_on_fixture(camry_model):
    assert camry_model.coef[camry_model.columns.index("age")] < 0


def test_older_car_gets_lower_price_at_same_mileage(camry_model):
    newer = VehicleQuery(manufacturer="toyota", model="camry", year=2018, odometer=90_000)
    older = newer.model_copy(update={"year": 2012})

    assert predict_price(camry_model, older).predicted_price < (
        predict_price(camry_model, newer).predicted_price
    )


def test_interval_with_one_predictor_matches_textbook_formula():
    # With x0 = [1, x], the (X^T X)^-1 form must equal the familiar
    # 1/n + (x - mean)^2 / Sxx form of the one-predictor interval.
    rng = np.random.default_rng(seed=1)
    x = rng.uniform(0, 10, size=25)
    X = np.column_stack([np.ones_like(x), x])
    x_new = np.array([0.0, 4.2, 12.0])
    x0 = np.column_stack([np.ones_like(x_new), x_new])
    s, dof, alpha = 1.7, x.size - 2, 0.05

    got = _half_width(x0, np.linalg.inv(X.T @ X), s, dof, alpha)

    sxx = np.sum((x - x.mean()) ** 2)
    expected = stats.t.ppf(1 - alpha / 2, dof) * s * np.sqrt(
        1 + 1 / x.size + (x_new - x.mean()) ** 2 / sxx
    )
    np.testing.assert_allclose(got, expected)


def test_query_age_is_never_negative(camry_model):
    next_year = VehicleQuery(manufacturer="toyota", year=LATEST_MODEL_YEAR)

    assert query_age(camry_model, next_year) == 0


def test_query_age_uses_range_midpoint_without_year(camry_model):
    query = VehicleQuery(manufacturer="toyota", year_min=2012, year_max=2016)

    assert query_age(camry_model, query) == pytest.approx(camry_model.ref_year - 2014)


def test_fit_refuses_listings_all_from_one_model_year(camry_listings):
    one_year = camry_listings.assign(year=2015)
    one_year["age"] = vehicle_age(one_year["posting_date"], one_year["year"])

    with pytest.raises(ValueError, match="several model years"):
        fit_model(one_year)


CAMRY_2015_90K = VehicleQuery(manufacturer="toyota", model="camry", year=2015, odometer=90_000)


def test_depreciation_curves_fall_with_mileage_and_age(camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    mileage = curves.mileage_curve.to_frame()
    age = curves.age_curve.to_frame()
    assert mileage["odometer"].is_monotonic_increasing
    assert mileage["price"].is_monotonic_decreasing
    assert age["age"].is_monotonic_increasing
    assert age["price"].is_monotonic_decreasing


def test_car_sits_exactly_on_both_curves(camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)
    car = curves.car

    # Each curve evaluated exactly at the car's value, not at the nearest grid point.
    on_mileage_curve, _, _ = predict_interval(camry_model, car.odometer, curves.fixed_age)
    on_age_curve, _, _ = predict_interval(camry_model, curves.fixed_odometer, car.age)

    assert curves.fixed_age == car.age
    assert curves.fixed_odometer == car.odometer
    assert np.isclose(car.price, on_mileage_curve)
    assert np.isclose(car.price, on_age_curve)
    assert np.isclose(car.price, predict_price(camry_model, CAMRY_2015_90K).predicted_price)


def test_curve_prices_lie_inside_their_intervals(camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    for frame in (curves.mileage_curve.to_frame(), curves.age_curve.to_frame()):
        assert (frame["lower"] <= frame["price"]).all()
        assert (frame["price"] <= frame["upper"]).all()
    car = curves.car
    assert car.lower <= car.price <= car.upper


def test_without_mileage_age_curve_uses_median_and_car_has_no_price(camry_model):
    query = CAMRY_2015_90K.model_copy(update={"odometer": None})

    curves = depreciation_curves(camry_model, query)

    assert curves.fixed_odometer == camry_model.median_odometer
    assert not curves.odometer_is_cars
    assert curves.car.odometer is None
    assert curves.car.price is None and curves.car.lower is None and curves.car.upper is None


def test_age_effect_is_clear_on_fixture(camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    assert curves.age_effect_clear
    assert curves.age_t_stat < 0  # older means cheaper


def test_drop_outliers_removes_planted_500_dollar_listings(camry_listings):
    # Five "down-payment ads": real rows (spread over model years) re-priced at $500.
    fake = camry_listings.iloc[[2, 11, 20, 29, 38]].assign(price=500.0)
    fake.index = fake.index + 1000  # distinct labels, so we can tell them apart
    listings = pd.concat([camry_listings, fake])

    kept = drop_outliers(listings)

    assert not kept.index.isin(fake.index).any()  # every fake one dropped
    real_kept = kept.index.isin(camry_listings.index).sum()
    assert real_kept >= 0.95 * len(camry_listings)


def test_drop_outliers_keeps_everything_without_outliers(camry_listings):
    # The fixture's exact price curve, with only +/-1% noise.
    rng = np.random.default_rng(seed=2)
    exact = (
        30_000
        * np.exp(-camry_listings["odometer"] / 150_000)
        * np.exp(-0.08 * camry_listings["age"])
    )
    clean = camry_listings.assign(price=exact * rng.uniform(0.99, 1.01, size=len(exact)))

    assert len(drop_outliers(clean)) == len(clean)


def test_typical_mileage_range_on_fixture(camry_model, camry_listings):
    # 2015 has 2014-2016 around it: 12 listings, enough for a range.
    low, high = typical_mileage_range(camry_model, 2015)
    near = camry_listings[camry_listings["year"].between(2014, 2016)]["odometer"]

    assert near.min() <= low < high <= near.max()
    # 2010 has only 2010-2011 around it (8 listings), and 2030 has none.
    assert typical_mileage_range(camry_model, 2010) is None
    assert typical_mileage_range(camry_model, 2030) is None


def test_typical_mileage_is_not_extrapolated_but_edge_of_overall_range_is(camry_model):
    low, high = typical_mileage_range(camry_model, 2015)
    typical = VehicleQuery(manufacturer="toyota", model="camry", year=2015,
                           odometer=(low + high) / 2)
    # The lowest mileage the model was fitted on: inside the overall range,
    # but unusual for a 2015.
    edge = typical.model_copy(update={"odometer": camry_model.odometer_min})
    assert camry_model.odometer_min < low

    assert not predict_price(camry_model, typical).extrapolated
    assert predict_price(camry_model, edge).extrapolated


def test_car_model_year_uses_year_or_rounded_range_midpoint(camry_model):
    assert car_model_year(camry_model, VehicleQuery(manufacturer="toyota", year=2015)) == 2015
    odd_range = VehicleQuery(manufacturer="toyota", year_min=2012, year_max=2015)
    assert car_model_year(camry_model, odd_range) == 2014  # 2013.5 rounds up


def test_every_curve_point_has_typical_and_car_is_typical(camry_model):
    low, high = typical_mileage_range(camry_model, 2015)
    query = CAMRY_2015_90K.model_copy(update={"odometer": (low + high) / 2})

    curves = depreciation_curves(camry_model, query)

    for curve in (curves.mileage_curve, curves.age_curve):
        assert all(isinstance(point.typical, bool) for point in curve.points)
    assert curves.car.typical is True
    assert curves.typical_range == (low, high)
    # The mileage curve covers more than the typical range, so some points aren't.
    assert not all(point.typical for point in curves.mileage_curve.points)


def test_clean_title_feature_gets_positive_coefficient(camry_listings):
    # Every other listing (22 of 44, enough for MIN_FEATURE_COUNT) is salvage, priced 40% lower.
    salvage = np.arange(len(camry_listings)) % 2 == 0
    listings = camry_listings.assign(
        title_status=np.where(salvage, "salvage", "clean"),
        price=np.where(salvage, camry_listings["price"] * 0.6, camry_listings["price"]).round(),
    )

    model = fit_model(listings, features=("clean_title",))

    assert model.features == ["clean_title"]
    assert model.columns == ["intercept", "odometer", "age", "clean_title"]
    assert model.coef[model.columns.index("clean_title")] > 0


def test_feature_with_too_few_listings_is_skipped(camry_listings):
    # Only 5 of 44 listings are 4wd: fewer than MIN_FEATURE_COUNT (20).
    drive = np.where(np.arange(len(camry_listings)) < 5, "4wd", "fwd")

    model = fit_model(camry_listings.assign(drive=drive), features=("four_wd",))

    assert model.features == []
    assert model.columns == ["intercept", "odometer", "age"]
    assert model.feature_shares["four_wd"] == pytest.approx(5 / 44)


def test_rare_but_real_feature_is_kept():
    # 30 problem titles out of 1,000 listings: only 3%, but 30 >= MIN_FEATURE_COUNT.
    rng = np.random.default_rng(seed=3)
    n = 1_000
    year = rng.integers(2010, 2021, size=n)
    posting_date = pd.Series(
        pd.Timestamp("2021-04-01", tz="UTC") + pd.to_timedelta(np.arange(n) % 30, unit="D")
    )
    age = vehicle_age(posting_date, year)
    odometer = rng.uniform(10_000, 200_000, size=n)
    problem = np.arange(n) < 30
    price = (
        30_000 * np.exp(-odometer / 150_000) * np.exp(-0.08 * age)
        * np.where(problem, 0.6, 1.0) * rng.uniform(0.85, 1.15, size=n)
    )
    listings = pd.DataFrame({
        "year": year, "odometer": odometer, "price": price.round(), "posting_date": posting_date,
        "age": age, "title_status": np.where(problem, "salvage", "clean"),
    })

    model = fit_model(listings, features=("clean_title",))

    assert model.features == ["clean_title"]
    assert model.coef[model.columns.index("clean_title")] > 0
