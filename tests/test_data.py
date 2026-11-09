"""filter_listings keeps exactly the listings the query asks for."""

from vehicle_pricing.data import filter_listings
from vehicle_pricing.schemas import VehicleQuery


def test_filter_listings_returns_only_requested_manufacturer(listings):
    result = filter_listings(listings, VehicleQuery(manufacturer="Toyota"))

    assert len(result) == (listings["manufacturer"] == "toyota").sum()  # all of them...
    assert set(result["manufacturer"]) == {"toyota"}  # ...and nothing else
