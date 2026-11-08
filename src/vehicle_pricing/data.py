"""Load and clean the Craigslist listings, and pick the listings that match a query."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vehicle_pricing.schemas import VehicleQuery

log = logging.getLogger(__name__)

# Built from this file's location, not the working directory, so it works from any folder.
# src/vehicle_pricing/data.py -> parents[2] is the project root.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "vehicles_slim.csv.gz"

# Columns kept when loading the (large) Craigslist CSV.
KEEP_COLUMNS = frozenset({
    "price", "year", "manufacturer", "model", "odometer", "region", "state",
    "condition", "cylinders", "fuel", "title_status", "transmission",
    "drive", "size", "type", "paint_color",
})

# Rows that match on all of these are treated as duplicate postings.
DEDUP_COLUMNS = ["region", "year", "manufacturer", "model", "price", "odometer"]

# VehicleQuery fields that match a column exactly (case-insensitive).
CATEGORICAL_FIELDS = ("manufacturer", "model", "condition", "fuel", "transmission", "cylinders", "drive")


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
    required = ["price", "odometer", "year"]
    out = df.dropna(subset=required).copy()
    out["year"] = out["year"].astype(int)

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


def filter_listings(df: pd.DataFrame, query: VehicleQuery) -> pd.DataFrame:
    """Return the listings matching every filter set on the query.

    Unset (None) fields don't filter. The year range is inclusive, and either
    end may be unset. The query's odometer is the mileage being priced, not a
    filter, so it is ignored here.
    """
    filters: dict[str, Any] = {
        field: getattr(query, field)
        for field in CATEGORICAL_FIELDS
        if getattr(query, field) is not None
    }
    if query.year_min is not None or query.year_max is not None:
        filters["year_range"] = (query.year_min, query.year_max)
    return _filter_vehicles(df, **filters)


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
