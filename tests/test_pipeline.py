"""estimate() runs the same chain as before, and raises the right warnings."""

import numpy as np
import pandas as pd
import pytest

from vehicle_pricing.data import UnknownManufacturerError, check_manufacturer, filter_listings
from vehicle_pricing.model import (
    DEFAULT_FEATURES,
    depreciation_curves,
    drop_outliers,
    fit_model,
    predict_price,
)
from vehicle_pricing.pipeline import (
    TooFewListingsError,
    WarningCode,
    _require_enough_listings,
    describe_years,
    estimate,
    warning_message,
)
from vehicle_pricing.schemas import MIN_LISTINGS, VehicleQuery


def camry(**fields):
    """A Camry query, e.g. camry(year=2015, odometer=90_000)."""
    return VehicleQuery(manufacturer="toyota", model="camry", **fields)


@pytest.fixture
def vehicles(camry_listings):
    """The Camry listings plus the two columns DEFAULT_FEATURES read.

    Every title is clean and no drivetrain is known, so both features are
    skipped (each needs at least 20 listings at 1 and 20 at 0).
    """
    return camry_listings.assign(title_status="clean", drive=np.nan)


@pytest.fixture
def mixed_drive_vehicles(vehicles):
    """The listings twice, once as 4wd and once as fwd (88 rows), so four_wd is used.

    A 50/50 split of the 44 rows wouldn't do: naming a drive would filter
    down to 22 listings, under MIN_LISTINGS.
    """
    return pd.concat(
        [vehicles.assign(drive="4wd"), vehicles.assign(drive="fwd")], ignore_index=True
    )


@pytest.fixture
def ageless_vehicles(vehicles):
    """Listings whose price depends on mileage only, not age.

    Every model year gets the same four mileages and the same four noise
    factors, so the age coefficient comes out close to zero (t near 0).
    Random noise could fake an age effect.
    """
    odometer = np.tile([40_000, 80_000, 120_000, 160_000], 11)
    noise = np.tile([0.97, 1.03, 1.01, 0.99], 11)
    price = 30_000 * np.exp(-odometer / 150_000) * noise
    return vehicles.assign(odometer=odometer.astype(float), price=price.round())


def test_estimate_matches_running_the_steps_by_hand(vehicles):
    query = camry(year=2015, odometer=90_000)

    est = estimate(vehicles, query)

    check_manufacturer(vehicles, query)
    matched = filter_listings(vehicles, query)
    listings = drop_outliers(matched, features=DEFAULT_FEATURES)
    model = fit_model(listings, features=DEFAULT_FEATURES)
    # Compared field by field: `est == ...` can't work, because comparing two
    # DataFrames with == gives a table of True/False, not one answer.
    assert est.query == query
    assert est.model == model
    assert est.result == predict_price(model, query)
    assert est.curves == depreciation_curves(model, query)
    assert est.listings.equals(listings)
    assert est.n_matched == len(matched)
    assert est.n_dropped == len(matched) - len(listings)


def test_typical_car_has_no_warnings(vehicles):
    low, high = estimate(vehicles, camry(year=2015)).curves.typical_range

    est = estimate(vehicles, camry(year=2015, odometer=(low + high) / 2))

    assert est.warnings == ()
    assert not est.result.extrapolated


def test_no_mileage_gives_no_price_but_still_curves(vehicles):
    est = estimate(vehicles, camry(year=2015))

    assert est.result is None
    assert len(est.curves.mileage_curve.points) > 0
    assert len(est.curves.age_curve.points) > 0
    assert est.curves.car.price is None
    assert WarningCode.EXTRAPOLATED not in est.warnings  # no price, so nothing to extrapolate


def test_very_high_mileage_is_extrapolated(vehicles):
    est = estimate(vehicles, camry(year=2015, odometer=400_000))

    assert est.result.extrapolated
    assert WarningCode.EXTRAPOLATED in est.warnings


def test_drivetrain_averaged_only_with_four_wd_in_model_and_no_drive(
    vehicles, mixed_drive_vehicles
):
    no_drive = estimate(mixed_drive_vehicles, camry(year=2015, odometer=90_000))
    drive_given = estimate(mixed_drive_vehicles, camry(year=2015, odometer=90_000, drive="4wd"))
    no_four_wd = estimate(vehicles, camry(year=2015, odometer=90_000))

    assert "four_wd" in no_drive.model.features
    assert WarningCode.DRIVETRAIN_AVERAGED in no_drive.warnings
    assert WarningCode.DRIVETRAIN_AVERAGED not in drive_given.warnings
    assert "four_wd" not in no_four_wd.model.features  # skipped: no 4wd listings
    assert WarningCode.DRIVETRAIN_AVERAGED not in no_four_wd.warnings


def test_ageless_prices_give_no_clear_age_effect(ageless_vehicles):
    est = estimate(ageless_vehicles, camry(year=2015, odometer=80_000))

    assert abs(est.curves.age_t_stat) < 2
    assert not est.curves.age_effect_clear
    assert WarningCode.NO_CLEAR_AGE_EFFECT in est.warnings


def test_unknown_make_raises(vehicles):
    with pytest.raises(UnknownManufacturerError):
        estimate(vehicles, VehicleQuery(manufacturer="tesla", model="model 3"))
    # main.py catches ValueError, so the error must be one.
    assert issubclass(UnknownManufacturerError, ValueError)


def test_too_few_matching_listings_raises_a_friendly_error(vehicles):
    # 2015 and 2016 are 4 listings each: 8 in all, under MIN_LISTINGS.
    query = camry(year_min=2015, year_max=2016, odometer=90_000)

    with pytest.raises(TooFewListingsError) as excinfo:
        estimate(vehicles, query)

    message = str(excinfo.value)
    assert "only 8 listings" in message
    assert f"at least {MIN_LISTINGS}" in message
    assert "Need at least" not in message  # not fit_model's blunt wording
    # main.py catches ValueError, so the error must be one.
    assert isinstance(excinfo.value, ValueError)


def test_too_few_listings_left_after_dropping_outliers_raises(vehicles, monkeypatch):
    # Stand-in for drop_outliers that throws away all but 10 listings.
    monkeypatch.setattr(
        "vehicle_pricing.pipeline.drop_outliers", lambda matched, **kwargs: matched.head(10)
    )

    with pytest.raises(TooFewListingsError, match="only 10 listings"):
        estimate(vehicles, camry(year=2015, odometer=90_000))  # 44 match before the drop


def test_no_matching_listings_says_none_were_found(vehicles):
    # A make that is in the data, with a model that isn't.
    query = VehicleQuery(manufacturer="toyota", model="cybertruck", year=2015)

    with pytest.raises(TooFewListingsError) as excinfo:
        estimate(vehicles, query)

    message = str(excinfo.value)
    assert "couldn't find any listings" in message
    assert "0" not in message  # no "only 0 listings" and no "at least 30"


def test_one_listing_is_singular():
    with pytest.raises(TooFewListingsError, match=r"only 1 listing like that"):
        _require_enough_listings(1)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (2010, 2020, "2010–2020 model years"),
        (2015, 2015, "2015 model year"),
        (None, 2005, "2005 and earlier model years"),
        (2018, None, "2018 and later model years"),
        (None, None, "all model years"),
    ],
)
def test_describe_years(low, high, expected):
    assert describe_years(low, high) == expected


def test_every_warning_code_has_a_message(vehicles):
    est = estimate(vehicles, camry(year=2015, odometer=400_000))

    for code in WarningCode:
        assert warning_message(code, est).strip(), f"empty message for {code!r}"
