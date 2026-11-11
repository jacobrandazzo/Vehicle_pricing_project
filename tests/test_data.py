"""filter_listings keeps exactly the listings the query asks for."""

import pandas as pd

from vehicle_pricing.data import comparison_years, filter_listings
from vehicle_pricing.schemas import YEAR_WINDOW, VehicleQuery


def test_filter_listings_returns_only_requested_manufacturer(listings):
    result = filter_listings(listings, VehicleQuery(manufacturer="Toyota"))

    assert len(result) == (listings["manufacturer"] == "toyota").sum()  # all of them...
    assert set(result["manufacturer"]) == {"toyota"}  # ...and nothing else


def test_comparison_years_single_year_uses_window():
    query = VehicleQuery(manufacturer="toyota", year=2015)

    assert comparison_years(query) == (2015 - YEAR_WINDOW, 2015 + YEAR_WINDOW)


def test_comparison_years_range_is_returned_as_given():
    query = VehicleQuery(manufacturer="honda", year_min=1996, year_max=2000)

    assert comparison_years(query) == (1996, 2000)


def test_comparison_years_without_any_year_is_open():
    assert comparison_years(VehicleQuery(manufacturer="honda")) == (None, None)


def test_filter_listings_single_year_keeps_window_edges(listings):
    # The fixture already has a 2010 Toyota (prius); add Toyotas just inside and
    # just outside the 2010-2020 window.
    extra = pd.DataFrame(
        [
            ("toyota", "camry", 2009, 150_000, 6_000),
            ("toyota", "camry", 2020, 20_000, 23_000),
            ("toyota", "camry", 2021, 10_000, 25_000),
        ],
        columns=listings.columns,
    )
    df = pd.concat([listings, extra], ignore_index=True)

    result = filter_listings(df, VehicleQuery(manufacturer="toyota", year=2015))

    years = set(result["year"])
    assert {2010, 2020} <= years  # both ends of the window are included
    assert years.isdisjoint({2009, 2021})  # one year past either end is not
