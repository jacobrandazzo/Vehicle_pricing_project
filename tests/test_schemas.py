"""VehicleQuery rejects bad values, and the agent's tool schema comes from it."""

import pytest
from pydantic import ValidationError

from vehicle_pricing import agent
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
