"""The prediction interval is built correctly and depreciation goes the right way."""

import numpy as np
import pytest
import scipy.stats as stats

from vehicle_pricing.data import vehicle_age
from vehicle_pricing.model import (
    _half_width,
    depreciation_curves,
    fit_model,
    predict_interval,
    predict_price,
    query_age,
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
    assert camry_model.b2 < 0


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
