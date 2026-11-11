"""filter_listings keeps exactly the listings the query asks for."""

import pandas as pd

from vehicle_pricing.data import clean_vehicles, comparison_years, filter_listings
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


def test_clean_vehicles_stores_text_as_categories():
    # A tiny made-up raw table, shaped like the CSV (no file is read).
    raw = pd.DataFrame({
        "price": [13_500, 9_800, 21_000],
        "odometer": [90_000.0, 110_000.0, 50_000.0],
        "year": [2015, 2014, 2016],
        "posting_date": ["2021-04-10T10:00:00-0400", "2021-04-11T09:30:00-0700",
                         "2021-04-12T15:45:00-0500"],
        "manufacturer": [" Toyota ", "honda", "FORD"],
        "model": ["Camry", "civic", "Mustang"],
        "region": ["SF bay area", "SF bay area", "boston"],
    })

    out = clean_vehicles(raw)

    for col in ("manufacturer", "model", "region"):
        assert isinstance(out[col].dtype, pd.CategoricalDtype), col
    for col in ("price", "odometer", "year", "age"):
        assert pd.api.types.is_numeric_dtype(out[col]), col
    assert pd.api.types.is_datetime64_any_dtype(out["posting_date"])
    # Only the storage changes: values are cleaned as before, region keeps its case.
    assert out["manufacturer"].tolist() == ["toyota", "honda", "ford"]
    assert out["region"].tolist() == ["SF bay area", "SF bay area", "boston"]


def test_filter_listings_same_rows_for_category_and_plain_text(listings):
    as_categories = listings.astype(
        {col: "category" for col in listings.columns if pd.api.types.is_string_dtype(listings[col])}
    )
    assert isinstance(as_categories["manufacturer"].dtype, pd.CategoricalDtype)

    for query in (
        VehicleQuery(manufacturer="Toyota"),
        VehicleQuery(manufacturer="honda", model="Civic"),
        VehicleQuery(manufacturer="ford", year_min=2015),
    ):
        plain = filter_listings(listings, query)
        category = filter_listings(as_categories, query)

        assert len(plain) > 0
        assert category.index.tolist() == plain.index.tolist()
