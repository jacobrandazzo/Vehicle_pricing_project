"""The prediction interval is built correctly and depreciation goes the right way."""

from vehicle_pricing.model import predict_price
from vehicle_pricing.schemas import VehicleQuery


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
