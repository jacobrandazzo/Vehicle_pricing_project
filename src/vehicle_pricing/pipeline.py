"""Turn a VehicleQuery into a price estimate, its depreciation curves and its warnings.

The terminal app (main.py) and the website both call estimate(), so they can
never disagree on a price. Parsing a description (agent.parse_description,
which calls Claude) stays separate: the website builds a VehicleQuery directly
from its dropdowns and skips Claude.
"""

from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vehicle_pricing.data import check_manufacturer, filter_listings
from vehicle_pricing.model import (
    DEFAULT_FEATURES,
    DepreciationCurves,
    FittedModel,
    depreciation_curves,
    drop_outliers,
    fit_model,
    predict_price,
)
from vehicle_pricing.schemas import MIN_LISTINGS, PriceResult, VehicleQuery


class TooFewListingsError(ValueError):
    """Too few listings match the query to estimate a price. str(e) is safe to show users."""


def _require_enough_listings(n: int) -> None:
    """Raise TooFewListingsError if n listings are fewer than the model needs (MIN_LISTINGS)."""
    if n == 0:  # nothing matched, so "I need at least 30" would be beside the point
        raise TooFewListingsError(
            "I couldn't find any listings like that. "
            "Check the make and model name, or try a wider year range."
        )
    if n < MIN_LISTINGS:
        noun = "listing" if n == 1 else "listings"
        raise TooFewListingsError(
            f"I found only {n:,} {noun} like that (I need at least {MIN_LISTINGS} "
            "to estimate a price). Try a wider year range, or check the model name."
        )


class WarningCode(StrEnum):
    """Something the user should know about an estimate. The wording is in warning_message()."""

    EXTRAPOLATED = "extrapolated"  # the mileage is unusual for a car of this age
    DRIVETRAIN_AVERAGED = "drivetrain_averaged"  # no drivetrain given, so the 4WD/2WD mix is used
    NO_CLEAR_AGE_EFFECT = "no_clear_age_effect"  # the listings can't pin down an age effect


class Estimate(BaseModel):
    """Everything estimate() works out for one query: all an app needs to show it.

    arbitrary_types_allowed lets the `listings` field hold a pandas DataFrame,
    a type pydantic can't validate itself; pydantic then only checks that the
    value is a DataFrame. frozen stops fields being reassigned, but a
    DataFrame's contents can still be changed in place, so treat `listings` as
    read-only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    query: VehicleQuery = Field(
        description="The query that was estimated. Units are set per field.",
    )
    model: FittedModel = Field(
        description="The price model fitted to `listings`. Units are set per field.",
    )
    result: PriceResult | None = Field(
        description=(
            "The predicted price and its 95% range, in US dollars. "
            "None if the query has no mileage."
        ),
    )
    curves: DepreciationCurves = Field(
        description="Price against mileage and against age for this car, in US dollars.",
    )
    listings: pd.DataFrame = Field(
        description=(
            "The listings the model was fitted on (after outlier removal), "
            "one row per listing. The plots need them."
        ),
    )
    n_matched: int = Field(
        ge=0,
        description="Listings that matched the query before outlier removal, as a count.",
    )
    n_dropped: int = Field(
        ge=0,
        description="Matched listings dropped as implausible prices, as a count.",
    )
    warnings: tuple[WarningCode, ...] = Field(
        description="Warnings to show the user, in display order. Codes, no units.",
    )


def estimate(vehicles: pd.DataFrame, query: VehicleQuery) -> Estimate:
    """Price one car: the whole chain from the cleaned listings to the curves.

    check_manufacturer -> filter_listings -> drop_outliers -> fit_model ->
    predict_price (only if the query has a mileage) -> depreciation_curves,
    using DEFAULT_FEATURES for both drop_outliers and fit_model.

    Raises UnknownManufacturerError (a ValueError) if the make isn't in the
    listings, and TooFewListingsError (also a ValueError) if too few listings
    match to fit the model.
    """
    check_manufacturer(vehicles, query)
    matched = filter_listings(vehicles, query)
    # Checked before drop_outliers, which fits a model itself and would raise
    # fit_model's blunter error first.
    _require_enough_listings(len(matched))
    # Fits once to find prices far off the model, then again without them.
    listings = drop_outliers(matched, features=DEFAULT_FEATURES)
    _require_enough_listings(len(listings))  # dropping outliers can leave too few
    model = fit_model(listings, features=DEFAULT_FEATURES)
    result = predict_price(model, query) if query.odometer is not None else None
    curves = depreciation_curves(model, query)

    warnings = []
    if result is not None and result.extrapolated:
        warnings.append(WarningCode.EXTRAPOLATED)
    if "four_wd" in model.features and query.drive is None:
        warnings.append(WarningCode.DRIVETRAIN_AVERAGED)
    if not curves.age_effect_clear:
        warnings.append(WarningCode.NO_CLEAR_AGE_EFFECT)

    return Estimate(
        query=query,
        model=model,
        result=result,
        curves=curves,
        listings=listings,
        n_matched=len(matched),
        n_dropped=len(matched) - len(listings),
        warnings=tuple(warnings),
    )


def warning_message(code: WarningCode, est: Estimate) -> str:
    """The sentence shown to the user for a warning, the same in every app."""
    match code:
        case WarningCode.EXTRAPOLATED:
            return ("Warning: that mileage is unusual for a car of this age, so the "
                    "estimate leans on other model years and is less reliable.")
        case WarningCode.DRIVETRAIN_AVERAGED:
            return "Drivetrain not given; price averaged over the 4WD/2WD mix in the listings."
        case WarningCode.NO_CLEAR_AGE_EFFECT:
            return (f"Note: these listings show no clear age effect "
                    f"(t = {est.curves.age_t_stat:.2f}), so the age curve is only a rough guide.")
    raise ValueError(f"No message for warning {code!r}")


def describe_years(low: int | None, high: int | None) -> str:
    """The comparison window as words, e.g. '2010–2020 model years'. None means that end is open."""
    if low is None and high is None:
        return "all model years"
    if low is None:
        return f"{high} and earlier model years"
    if high is None:
        return f"{low} and later model years"
    if low == high:
        return f"{low} model year"
    return f"{low}–{high} model years"
