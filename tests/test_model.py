"""The prediction interval is built correctly and depreciation goes the right way."""

import numpy as np
import pytest
import scipy.stats as stats

from vehicle_pricing.data import vehicle_age
from vehicle_pricing.model import _half_width, fit_model, predict_price, query_age
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
