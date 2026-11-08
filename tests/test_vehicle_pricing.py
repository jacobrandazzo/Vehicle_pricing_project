"""Offline tests for the vehicle_pricing package. None call Claude or read the real data."""

import pytest
from pydantic import ValidationError

from vehicle_pricing import agent
from vehicle_pricing.data import filter_listings
from vehicle_pricing.model import predict_price
from vehicle_pricing.schemas import VehicleQuery


def test_vehicle_query_rejects_bad_year_and_negative_odometer():
    with pytest.raises(ValidationError) as bad_year:
        VehicleQuery(manufacturer="honda", year_min="banana")
    assert bad_year.value.errors()[0]["loc"] == ("year_min",)

    with pytest.raises(ValidationError) as bad_miles:
        VehicleQuery(manufacturer="honda", odometer=-5)
    assert bad_miles.value.errors()[0]["loc"] == ("odometer",)


def test_tool_input_schema_matches_vehicle_query():
    assert agent._TOOL["input_schema"] == VehicleQuery.model_json_schema()


def test_filter_listings_returns_only_requested_manufacturer(listings):
    result = filter_listings(listings, VehicleQuery(manufacturer="Toyota"))

    assert len(result) == (listings["manufacturer"] == "toyota").sum()  # all of them...
    assert set(result["manufacturer"]) == {"toyota"}  # ...and nothing else


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
