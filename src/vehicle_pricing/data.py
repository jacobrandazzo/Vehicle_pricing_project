"""Load and clean the Craigslist listings, and pick the listings that match a query."""

from __future__ import annotations

import difflib
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vehicle_pricing.schemas import YEAR_WINDOW, VehicleQuery

log = logging.getLogger(__name__)

# Built from this file's location, not the working directory, so it works from any folder.
# src/vehicle_pricing/data.py -> parents[2] is the project root.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "vehicles_slim.csv.gz"

# Columns kept when loading the (large) Craigslist CSV.
KEEP_COLUMNS = frozenset({
    "price", "year", "manufacturer", "model", "odometer", "region", "state",
    "condition", "cylinders", "fuel", "title_status", "transmission",
    "drive", "size", "type", "paint_color", "posting_date",
})

# Rows that match on all of these are treated as duplicate postings.
DEDUP_COLUMNS = ["region", "year", "manufacturer", "model", "price", "odometer"]

# VehicleQuery fields that match a column exactly (case-insensitive).
CATEGORICAL_FIELDS = ("manufacturer", "model", "condition", "fuel", "transmission", "cylinders", "drive")


def vehicle_age(posting_date: pd.Series, model_year: pd.Series) -> pd.Series:
    """Return each vehicle's age in years (a decimal) on the day it was posted.

    The posting date becomes a decimal year:
    ``year + (day_of_year - 1) / days_in_that_year``, so 1 January 2021 is
    2021.0 and 2 July 2021 is about 2021.5. The model year is treated as
    1 January of that year. Age is the difference between the two.

    The result is clipped at 0 because dealers list next year's models
    early. A 2022 car posted in April 2021 would otherwise get an age of
    about -0.7.
    """
    days_in_year = np.where(posting_date.dt.is_leap_year, 366, 365)
    decimal_year = posting_date.dt.year + (posting_date.dt.dayofyear - 1) / days_in_year
    return (decimal_year - model_year).clip(lower=0)


def clean_vehicles(
    df: pd.DataFrame,
    *,
    price_range: tuple[float, float] = (500, 150_000),
    odometer_range: tuple[float, float] = (1_000, 400_000),
    year_range: tuple[int, int | None] = (1980, None),
) -> pd.DataFrame:
    """Drop obviously bad listings and normalise text columns.

    Craigslist data contains placeholder prices ($1, $123456), placeholder
    odometers (1, 999999), missing values and duplicate postings. The default
    ranges are conservative starting points -- tune them to your data.
    """
    required = ["price", "odometer", "year", "posting_date"]
    out = df.dropna(subset=required).copy()
    out["year"] = out["year"].astype(int)
    # Postings mix time-zone offsets (-0400, -0700, ...), so parse them all to UTC.
    out["posting_date"] = pd.to_datetime(out["posting_date"], utc=True)
    out["age"] = vehicle_age(out["posting_date"], out["year"])

    max_year = year_range[1] if year_range[1] is not None else date.today().year + 1
    keep = (
        out["price"].between(*price_range)
        & out["odometer"].between(*odometer_range)
        & out["year"].between(year_range[0], max_year)
    )
    out = out[keep]

    for col in ("manufacturer", "model"):
        if col in out.columns:
            out[col] = out[col].str.strip().str.lower()

    dedup_cols = [c for c in DEDUP_COLUMNS if c in out.columns]
    out = out.drop_duplicates(subset=dedup_cols)

    log.info("Cleaned listings: %d -> %d rows", len(df), len(out))
    return out.reset_index(drop=True)


def load_data(path: str | Path = DATA_PATH, **clean_kwargs: Any) -> pd.DataFrame:
    """Read the listings CSV (only the columns we use) and clean it."""
    raw = pd.read_csv(path, usecols=lambda col: col in KEEP_COLUMNS)
    return clean_vehicles(raw, **clean_kwargs)


class UnknownManufacturerError(ValueError):
    """The query's manufacturer isn't in the listings. str(e) is safe to show users."""


def check_manufacturer(df: pd.DataFrame, query: VehicleQuery) -> None:
    """Raise UnknownManufacturerError unless the query's make appears in the listings.

    Catches makes the parser got wrong or invented (e.g. "<unknown>" when the
    description named no make), which would otherwise just match 0 listings.
    """
    known = df["manufacturer"].dropna().unique().tolist()
    if query.manufacturer in known:
        return
    close = difflib.get_close_matches(query.manufacturer, known, n=3, cutoff=0.6)
    if close:
        raise UnknownManufacturerError(
            f"I don't have listings for '{query.manufacturer}'. Did you mean {' or '.join(close)}?"
        )
    raise UnknownManufacturerError(
        "I couldn't tell which make of vehicle that is. "
        "Please include the make, e.g. 'Toyota Camry'."
    )


def filter_listings(df: pd.DataFrame, query: VehicleQuery) -> pd.DataFrame:
    """Return the listings matching every filter set on the query.

    Unset (None) fields don't filter. Model years come from comparison_years();
    the range is inclusive, and either end may be open. The query's odometer is
    the mileage being priced, not a filter, so it is ignored here.
    """
    filters: dict[str, Any] = {
        field: getattr(query, field)
        for field in CATEGORICAL_FIELDS
        if getattr(query, field) is not None
    }
    year_range = comparison_years(query)
    if year_range != (None, None):
        filters["year_range"] = year_range
    return _filter_vehicles(df, **filters)


def comparison_years(query: VehicleQuery) -> tuple[int | None, int | None]:
    """Return the (earliest, latest) model years of the comparison listings.

    This is the one place the comparison window is decided, so a later
    lookup (e.g. by model generation) can replace it without touching
    anything else.

    - If the query sets year_min or year_max, they are returned as given
      (either may be None, meaning that end is open).
    - Otherwise, if it sets year, the window is year +/- YEAR_WINDOW.
    - Otherwise there is no window: (None, None).
    """
    if query.year_min is not None or query.year_max is not None:
        return query.year_min, query.year_max
    if query.year is not None:
        return query.year - YEAR_WINDOW, query.year + YEAR_WINDOW
    return None, None


def _filter_vehicles(
    df: pd.DataFrame,
    *,
    year_range: tuple[float | None, float | None] | None = None,
    price_range: tuple[float | None, float | None] | None = None,
    odometer_range: tuple[float | None, float | None] | None = None,
    **categorical: Any,
) -> pd.DataFrame:
    """Return the listings matching every given filter (None = no filter).

    ``*_range`` arguments are inclusive ``(low, high)`` pairs; either end may
    be None. Any other keyword is a column name mapped to one value or a list
    of values, e.g. ``manufacturer="toyota"`` or ``state=["ca", "or"]``.
    String columns are compared case-insensitively.
    """
    mask = pd.Series(True, index=df.index)

    for col, bounds in (
        ("year", year_range),
        ("price", price_range),
        ("odometer", odometer_range),
    ):
        if bounds is not None:
            low, high = bounds
            mask &= df[col].between(
                -np.inf if low is None else low,
                np.inf if high is None else high,
            )

    for col, values in categorical.items():
        if col not in df.columns:
            raise KeyError(f"Unknown column {col!r}. Available: {sorted(df.columns)}")
        values = np.atleast_1d(values)
        column = df[col]
        if pd.api.types.is_string_dtype(column):
            mask &= column.str.lower().isin({str(v).lower() for v in values})
        else:
            mask &= column.isin(values)

    return df[mask].copy()
