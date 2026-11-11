"""Fit price on mileage and age with Box-Cox transforms, and predict with an interval.

The model is a plane in Box-Cox-transformed space:

    boxcox(price, lambda_y) = b0 + b1 * boxcox(odometer, lambda_x)
                                 + b2 * boxcox(age + 1, lambda_age)
                                 [+ one coefficient per optional 0/1 feature]

Age is in years (see data.vehicle_age); 1 is added because Box-Cox needs
positive inputs and a car can be 0 years old. Optional features (see
FEATURES), such as a clean title, enter untransformed as 0/1 columns. The
coefficients come from least squares. The three lambdas are chosen by maximising the Box-Cox profile
log-likelihood (see fit_model). Predictions back-transform to dollars, so the
estimate is a median price, with a prediction interval for a single listing.
depreciation_curves returns price against mileage and against age, as data.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize
from scipy.special import boxcox

from vehicle_pricing.data import decimal_year
from vehicle_pricing.schemas import MIN_LISTINGS, PriceResult, VehicleQuery

DEFAULT_ALPHA = 0.05  # 95% prediction interval
BASE_COLUMNS = ("intercept", "odometer", "age")  # always in the model, in this order
MIN_FEATURE_COUNT = 20  # a feature is skipped unless at least this many listings are 1 and 0
# Features the app, baseline and figures use (fit_model and drop_outliers still default to
# none). Chosen by 5-fold CV: they gave the lowest median % error (docs/evaluation.md).
DEFAULT_FEATURES = ("clean_title", "four_wd")
PROBLEM_TITLES = frozenset({"salvage", "rebuilt", "parts only", "missing"})
MIN_AGE_SPREAD = 1.0  # years between the youngest and oldest listing needed to fit age
# Default drop_outliers passes (fit, score, drop). 5-fold CV (docs/evaluation.md) showed
# no real difference between 1 and 3 passes (at most 0.35 points of median % error), so
# use the simpler one.
OUTLIER_PASSES = 1
CURVE_POINTS = 100  # points along each depreciation curve
CLEAR_T_STAT = 2.0  # |t| at or above this counts as a clear age effect
TYPICAL_YEAR_WINDOW = 1  # typical mileage for a model year uses listings within +/- this many years
TYPICAL_PERCENTILES = (5, 95)  # the "typical" mileage band, as odometer percentiles
MIN_TYPICAL_LISTINGS = 10  # fewer listings than this: no typical range for that model year


class TypicalMileage(BaseModel):
    """The usual mileage of listings around one model year."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    year: int  # model year
    n: int  # listings with model year within +/- TYPICAL_YEAR_WINDOW of it
    low: float  # 5th percentile odometer of those listings, in miles
    high: float  # 95th percentile odometer, in miles


@dataclass(frozen=True)
class Feature:
    """An optional 0/1 predictor.

    column(listings) gives each listing's value (an array of 0s and 1s).
    for_query(query, listings_share) gives the car's value; listings_share is
    the share of 1s in the fitted listings, for when the query doesn't say.
    """

    description: str
    column: Callable[[pd.DataFrame], np.ndarray]
    for_query: Callable[[VehicleQuery, float], float]


def _clean_title_column(listings: pd.DataFrame) -> np.ndarray:
    """0 for a problem title, 1 otherwise.

    Problem titles (PROBLEM_TITLES) are "salvage", "rebuilt", "parts only"
    and "missing" (the literal value, meaning the car has no title).
    Everything else is 1: "clean", and "lien", which means an outstanding
    loan rather than damage (lien listings sell above clean ones).

    An empty title_status (NaN) is treated as clean, i.e. imputed with the
    most common value: clean is about 95% of titles, and NaN is only about
    1.5% of listings, too few to be a category of its own.
    """
    status = listings["title_status"].astype("string").str.strip().str.lower()
    return (~status.isin(PROBLEM_TITLES).fillna(False)).to_numpy(dtype=float)


def _four_wd_column(listings: pd.DataFrame) -> np.ndarray:
    """1 if the drivetrain is 4wd, else 0 (fwd, rwd or missing)."""
    return (listings["drive"] == "4wd").to_numpy(dtype=float)


def _four_wd_for_query(query: VehicleQuery, listings_share: float) -> float:
    """1 or 0 if the query names a drivetrain, otherwise the listings' share of 4wd.

    Using the share (a value between 0 and 1) gives a price averaged over the
    mix of drivetrains in the listings, rather than pretending to know which
    one this car has.
    """
    if query.drive is not None:
        return 1.0 if query.drive == "4wd" else 0.0
    return listings_share


FEATURES: dict[str, Feature] = {
    "clean_title": Feature(
        description="Clean (or unstated) title, versus salvage, rebuilt and the like.",
        column=_clean_title_column,
        for_query=lambda query, listings_share: 1.0,  # the car is priced as clean-titled
    ),
    "four_wd": Feature(
        description="Four-wheel drive.",
        column=_four_wd_column,
        for_query=_four_wd_for_query,
    ),
}


class FittedModel(BaseModel):
    """Everything needed to predict from a fitted model, without the listings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_x: float  # Box-Cox exponent for odometer
    lambda_age: float  # Box-Cox exponent for age + 1
    lambda_y: float  # Box-Cox exponent for price
    columns: list[str]  # design columns: ["intercept", "odometer", "age", *features]
    coef: list[float]  # one coefficient per column, in transformed space
    features: list[str]  # the optional features used (columns after "age")
    feature_shares: dict[str, float]  # share of 1s in the listings, per requested feature
    xtx_inv: list[list[float]]  # (X^T X)^-1 of the design matrix
    resid_se: float  # residual standard error s, in transformed space (n - len(coef) df)
    n: int  # number of listings the model was fit on
    odometer_min: float  # odometer range fitted on, in miles
    odometer_max: float
    median_odometer: float  # median mileage of the listings, in miles
    age_min: float  # age range fitted on, in years
    age_max: float
    median_age: float  # median age of the listings, in years
    ref_year: float  # median posting date of the listings, as a decimal year
    r2: float  # R^2 in transformed space
    log_likelihood: float  # maximised Box-Cox profile log-likelihood
    typical_mileage: list[TypicalMileage]  # one entry per model year in the listings


def fit_model(
    df: pd.DataFrame,
    *,
    lambda_range: tuple[float, float] = (-2.0, 2.0),
    grid_step: float = 0.25,
    min_samples: int = MIN_LISTINGS,
    features: Sequence[str] = (),
) -> FittedModel:
    """Fit the model to the listings' 'odometer', 'age', 'price' and 'posting_date' columns.

    The 'year' column is used only for typical_mileage (see _typical_mileage).

    `features` names optional 0/1 predictors from FEATURES, added after age.
    A feature is skipped if fewer than MIN_FEATURE_COUNT (20) listings have
    a 1, or fewer than 20 have a 0: with so few on one side its coefficient
    can't be estimated reliably. A count, not a share, so a rare but real
    category is still used once there are enough cars in it (e.g. problem
    titles are only about 4% of Camrys, but that is about 46 listings). The
    model records which were used (`features`) and each requested feature's
    share of 1s (`feature_shares`). Features are not transformed, so the lambda search
    is unchanged; they just add columns to each least-squares fit in it.

    The lambdas maximise the Box-Cox profile log-likelihood

        l(lambda_x, lambda_age, lambda_y) = -(n/2) * log(RSS / n)
                                            + (lambda_y - 1) * sum(log(price))

    where RSS is the residual sum of squares of the least-squares fit at those
    lambdas. The second term is needed because RSS values at different
    lambda_y are on different scales: transforming price with lambda_y = 0.5
    gives numbers in the hundreds, with lambda_y = -1 numbers below 1, so the
    raw RSS would always favour whichever lambda_y shrinks price the most.
    The term is the log of the transform's Jacobian. It puts every lambda_y
    on the scale of the original prices, so the likelihoods can be compared.
    Only price needs it: odometer and age are predictors, not the response.

    The search is a coarse grid over lambda_range for all three lambdas,
    then Nelder-Mead from the best grid point, within the same bounds.
    """
    x = np.asarray(df["odometer"], dtype=float)
    age = np.asarray(df["age"], dtype=float)
    y = np.asarray(df["price"], dtype=float)
    if not x.shape == age.shape == y.shape:
        raise ValueError("odometer, age and price must have the same length")
    if x.size < min_samples:
        raise ValueError(f"Need at least {min_samples} listings to fit, got {x.size}")
    if (x <= 0).any() or (y <= 0).any() or (age < 0).any():
        raise ValueError("odometer and price must be positive, and age not negative")
    if np.ptp(age) < MIN_AGE_SPREAD:
        raise ValueError(
            "These listings are all about the same age, so I can't tell how price "
            "changes with age. The model needs listings from several model years; "
            "try a wider year range."
        )

    unknown = [name for name in features if name not in FEATURES]
    if unknown:
        raise KeyError(f"Unknown feature(s) {unknown}. Available: {sorted(FEATURES)}")
    values = {name: FEATURES[name].column(df) for name in features}
    shares = {name: float(v.mean()) for name, v in values.items()}
    used = [name for name in features
            if min(values[name].sum(), (1 - values[name]).sum()) >= MIN_FEATURE_COUNT]
    columns = [*BASE_COLUMNS, *used]
    feature_values = {name: values[name] for name in used}
    sum_log_y = float(np.sum(np.log(y)))

    lam_x, lam_age, lam_y = _grid_search(
        x, age, feature_values, columns, y, sum_log_y, lambda_range, grid_step
    )
    refined = minimize(
        lambda p: -_log_likelihood(x, age, feature_values, columns, y, sum_log_y, *p),
        x0=[lam_x, lam_age, lam_y],
        method="Nelder-Mead",
        bounds=[lambda_range] * 3,
    )
    lam_x, lam_age, lam_y = (float(v) for v in refined.x)

    X = _build_design(_DesignSpec(lam_x, lam_age, columns), x, age, feature_values)
    ty = boxcox(y, lam_y)
    coef, *_ = np.linalg.lstsq(X, ty, rcond=None)
    resid = ty - X @ coef
    rss = float(resid @ resid)
    # (X^T X)^-1 = R^-1 R^-T from X = QR; more accurate than inverting X^T X directly.
    r_inv = np.linalg.inv(np.linalg.qr(X, mode="r"))

    return FittedModel(
        lambda_x=lam_x,
        lambda_age=lam_age,
        lambda_y=lam_y,
        columns=columns,
        coef=[float(c) for c in coef],
        features=used,
        feature_shares=shares,
        xtx_inv=(r_inv @ r_inv.T).tolist(),
        resid_se=float(np.sqrt(rss / (x.size - len(coef)))),
        n=int(x.size),
        odometer_min=float(x.min()),
        odometer_max=float(x.max()),
        median_odometer=float(np.median(x)),
        age_min=float(age.min()),
        age_max=float(age.max()),
        median_age=float(np.median(age)),
        ref_year=float(np.median(decimal_year(df["posting_date"]))),
        r2=float(1 - rss / np.sum((ty - ty.mean()) ** 2)),
        log_likelihood=_log_likelihood(
            x, age, feature_values, columns, y, sum_log_y, lam_x, lam_age, lam_y
        ),
        typical_mileage=_typical_mileage(df),
    )


def coef_t_stat(model: FittedModel, column: str) -> float:
    """A coefficient divided by its standard error, s * sqrt([(X^T X)^-1] on the diagonal).

    `column` is a name from model.columns, e.g. "age" or "clean_title".
    """
    if column not in model.columns:
        raise KeyError(f"The model has no column {column!r}. Columns: {model.columns}")
    i = model.columns.index(column)
    return float(model.coef[i] / (model.resid_se * np.sqrt(model.xtx_inv[i][i])))


def query_features(model: FittedModel, query: VehicleQuery) -> dict[str, float]:
    """The car's value for each feature the model uses (see Feature.for_query)."""
    return {name: FEATURES[name].for_query(query, model.feature_shares[name])
            for name in model.features}


def listing_features(model: FittedModel, listings: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each listing's value for each feature the model uses."""
    return {name: FEATURES[name].column(listings) for name in model.features}


def query_age(model: FittedModel, query: VehicleQuery) -> float:
    """The age, in years, of the car being priced, as of the listings' median posting date.

    - If the query sets year, the age is ref_year - year.
    - Otherwise, if it sets a year range, the midpoint year of the range is
      used. An open end is ignored, so a range with only year_min uses year_min.
    - Otherwise it is the median age of the fitted listings.

    Clipped at 0, like vehicle_age, because next year's models are listed early.
    """
    if query.year is not None:
        year = query.year
    elif query.year_min is not None or query.year_max is not None:
        ends = [y for y in (query.year_min, query.year_max) if y is not None]
        year = sum(ends) / len(ends)
    else:
        return model.median_age
    return max(model.ref_year - year, 0.0)


def car_model_year(model: FittedModel, query: VehicleQuery) -> int:
    """The model year of the car being priced, chosen the same way as query_age.

    query.year if set; otherwise the midpoint of the year range, rounded
    (half up; an open end is ignored); otherwise the model year of a car of
    the listings' median age.
    """
    if query.year is not None:
        return query.year
    if query.year_min is not None or query.year_max is not None:
        ends = [y for y in (query.year_min, query.year_max) if y is not None]
        return _round_half_up(sum(ends) / len(ends))
    return _round_half_up(model.ref_year - model.median_age)


def typical_mileage_range(model: FittedModel, model_year: int) -> tuple[float, float] | None:
    """(low, high) usual odometer, in miles, for cars of this model year, or None.

    It is the 5th to 95th percentile mileage of listings within +/- 1 model
    year. None if the listings have no cars of that model year, or fewer than
    MIN_TYPICAL_LISTINGS around it, so there is no reliable "usual" mileage.

    Age and mileage go together (older cars have more miles), so the fitted
    curves cover mileages that no car of a given age actually has. Outside
    this range a curve's shape comes from other model years.
    """
    for entry in model.typical_mileage:
        if entry.year == model_year:
            return (entry.low, entry.high) if entry.n >= MIN_TYPICAL_LISTINGS else None
    return None


def drop_outliers(
    listings: pd.DataFrame,
    z: float = 3.5,
    max_passes: int = OUTLIER_PASSES,
    features: Sequence[str] = (),
) -> pd.DataFrame:
    """Return a copy of the listings without those whose price is far off the fitted model.

    The model is fitted to the listings. Each listing's residual r is
    measured in transformed space (boxcox(price, lambda_y) minus the fitted
    value), and scored with a robust z-score:

        robust z = (r - median(r)) / (1.4826 * MAD),   MAD = median(|r - median(r)|)

    Listings with |robust z| > z are dropped. (1.4826 * MAD equals the
    standard deviation when residuals are normal, so z means roughly
    "standard deviations".)

    Why MAD-based: the mean and standard deviation of the residuals are
    themselves dragged by the outliers they are meant to catch (a few $500
    down-payment ads widen the standard deviation, so they look less
    extreme). The median and MAD barely move.

    Why relative, and not a price floor: a residual compares a listing with
    what the model expects for its own mileage and age. $1,500 is wildly
    low for a 2018 Camry with 40,000 miles but normal for a 1998 Civic with
    200,000, so a fixed floor would either miss the first or throw away the
    second. The same goes for too-high dealer prices.

    Why allow several passes: the first fit is made with the outliers still
    in, and they pull it (e.g. towards a lambda_y on which they look less
    extreme), so some survive. Refitting without the worst ones and scoring
    again catches more. It stops when a pass drops nothing, or after
    max_passes, so it can't keep eating into real listings. The default is
    one pass (OUTLIER_PASSES): in cross-validation more passes didn't
    improve predictions.

    Each pass fits with the given `features` (see fit_model).

    If MAD is 0 (most residuals identical), z can't be computed and that
    pass drops nothing. fit_model's errors (too few listings, no age
    spread) are raised as they are.
    """
    kept = listings
    for _ in range(max_passes):
        within = _within_robust_z(kept, z, features)
        if within.all():
            break
        kept = kept[within]
    return kept.copy()


def _within_robust_z(listings: pd.DataFrame, z: float, features: Sequence[str]) -> np.ndarray:
    """One pass of drop_outliers: True for listings with |robust z| <= z."""
    model = fit_model(listings, features=features)
    design = _build_design(
        model, listings["odometer"], listings["age"], listing_features(model, listings)
    )
    fitted = design @ np.array(model.coef)
    resid = boxcox(np.asarray(listings["price"], dtype=float), model.lambda_y) - fitted
    centre = np.median(resid)
    mad = np.median(np.abs(resid - centre))
    if mad == 0:
        return np.ones(len(listings), dtype=bool)
    return np.abs((resid - centre) / (1.4826 * mad)) <= z


def predict_price(
    model: FittedModel, query: VehicleQuery, alpha: float = DEFAULT_ALPHA
) -> PriceResult:
    """Predict the median price at the query's mileage and age, with a 1 - alpha interval.

    The result is marked extrapolated if the mileage or age is outside what
    the model was fitted on, or the mileage is unusual for the car's model
    year (outside typical_mileage_range, or there is no range). Features
    are set to the car's values (query_features).
    """
    if query.odometer is None:
        raise ValueError("A price prediction needs the vehicle's mileage (odometer).")
    age = query_age(model, query)
    estimate, lower, upper = predict_interval(
        model, query.odometer, age, alpha, feature_values=query_features(model, query)
    )
    typical_range = typical_mileage_range(model, car_model_year(model, query))
    return PriceResult(
        predicted_price=float(estimate),
        lower_bound=float(lower),
        upper_bound=float(upper),
        n_listings=model.n,
        extrapolated=not (
            model.odometer_min <= query.odometer <= model.odometer_max
            and model.age_min <= age <= model.age_max
            and _is_typical(query.odometer, typical_range)
        ),
        query=query,
    )


def predict_interval(
    model: FittedModel,
    odometer: ArrayLike,
    age: ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    feature_values: Mapping[str, ArrayLike] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (estimate, lower, upper) prices in dollars for one or many cars.

    odometer (miles), age (years) and any feature values are broadcast
    against each other, so each can be a single value. feature_values maps
    each feature the model uses to its value(s); it can be left out when the
    model uses none. The estimate is the *median* price (back-transforming a
    mean in Box-Cox space gives a median). The bounds form a ``1 - alpha``
    prediction interval for a single new listing.
    """
    x0 = _build_design(model, odometer, age, feature_values)
    y_hat = x0 @ np.array(model.coef)
    half_width = _half_width(
        x0, np.array(model.xtx_inv), model.resid_se, model.n - len(model.coef), alpha
    )
    return (
        boxcox_inverse(y_hat, model.lambda_y),
        boxcox_inverse(y_hat - half_width, model.lambda_y),
        boxcox_inverse(y_hat + half_width, model.lambda_y),
    )


class MileagePoint(BaseModel):
    """One point on the mileage curve. Prices are in US dollars."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    odometer: float  # miles
    price: float  # median price
    lower: float  # prediction interval
    upper: float
    typical: bool  # odometer is inside the car's typical mileage range


class AgePoint(BaseModel):
    """One point on the age curve. Prices are in US dollars."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    age: float  # years
    model_year: float  # ref_year - age, a decimal year
    price: float  # median price
    lower: float  # prediction interval
    upper: float
    typical: bool  # the fixed odometer is inside the typical range for this model year


class MileageCurve(BaseModel):
    """Price against mileage, with age held fixed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    points: list[MileagePoint]

    def to_frame(self) -> pd.DataFrame:
        """The points as a DataFrame (columns odometer, price, lower, upper), for plotting."""
        return pd.DataFrame([p.model_dump() for p in self.points])


class AgeCurve(BaseModel):
    """Price against age, with mileage held fixed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    points: list[AgePoint]

    def to_frame(self) -> pd.DataFrame:
        """The points as a DataFrame (columns age, model_year, price, lower, upper), for plotting."""
        return pd.DataFrame([p.model_dump() for p in self.points])


class CarPoint(BaseModel):
    """The car being priced, as a point on both curves.

    The price fields and typical are None when the query has no mileage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    age: float  # years, from query_age
    model_year: float  # ref_year - age, a decimal year
    odometer: float | None  # miles, from the query
    price: float | None
    lower: float | None
    upper: float | None
    typical: bool | None  # odometer is inside the typical range for the car's model year


class DepreciationCurves(BaseModel):
    """How the fitted price changes with mileage and with age, for one car.

    The mileage curve holds age at fixed_age (the car's own age). The age
    curve holds mileage at fixed_odometer: the car's mileage if the query has
    one (odometer_is_cars = True), otherwise the listings' median mileage.

    Each point's `typical` flag says whether real cars like it exist in the
    listings (see typical_mileage_range). Where it is False, the curve's
    shape there is borrowed from other model years.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mileage_curve: MileageCurve
    age_curve: AgeCurve
    fixed_age: float  # years; age held constant along the mileage curve
    fixed_odometer: float  # miles; mileage held constant along the age curve
    odometer_is_cars: bool  # True: fixed_odometer is the car's; False: the median
    car_year: int  # the car's model year, from car_model_year
    typical_range: tuple[float, float] | None  # typical mileage for the car's model year
    car: CarPoint
    age_t_stat: float  # the age coefficient divided by its standard error
    age_effect_clear: bool  # |age_t_stat| >= CLEAR_T_STAT
    alpha: float  # the bands are 1 - alpha prediction intervals


def depreciation_curves(
    model: FittedModel, query: VehicleQuery, alpha: float = DEFAULT_ALPHA
) -> DepreciationCurves:
    """Price curves for one car: against mileage, and against age. Data only, no plotting.

    - Mileage curve: price over the fitted mileage range, with the car's age
      held fixed, so it shows what extra miles alone cost this car.
    - Age curve: price over the fitted age range, with mileage held fixed at
      the car's mileage (or the listings' median mileage if the query has
      none), so it shows what extra years alone cost.

    Any features the model uses are held at the car's values (query_features).

    Each curve has CURVE_POINTS evenly spaced points, each with the median
    price and its 1 - alpha prediction interval. Everything, including the car's own
    point, comes from predict_interval, so the car sits exactly on both curves.

    Typical flags: a mileage-curve point is typical if its odometer is inside
    the typical mileage range for the car's model year. An age-curve point is
    typical if the fixed odometer is inside the typical range for that
    point's model year (rounded to a whole year). No range means not typical.

    age_t_stat says how clearly the listings show an age effect once mileage
    is accounted for: the age coefficient divided by its standard error.
    If |t| >= 2, the effect is about two standard errors or more from zero,
    so it is unlikely to be just noise (roughly a 95% test), and
    age_effect_clear is True. If |t| < 2, the age curve's slope may not mean
    much: these listings can't separate age from mileage and noise.
    """
    age = query_age(model, query)
    odometer_is_cars = query.odometer is not None
    odometer = query.odometer if odometer_is_cars else model.median_odometer

    car_features = query_features(model, query)

    miles = np.linspace(model.odometer_min, model.odometer_max, CURVE_POINTS)
    m_est, m_lo, m_hi = predict_interval(model, miles, age, alpha, car_features)
    ages = np.linspace(model.age_min, model.age_max, CURVE_POINTS)
    a_est, a_lo, a_hi = predict_interval(model, odometer, ages, alpha, car_features)

    if odometer_is_cars:
        c_est, c_lo, c_hi = (
            float(v) for v in predict_interval(model, odometer, age, alpha, car_features)
        )
    else:
        c_est = c_lo = c_hi = None

    car_year = car_model_year(model, query)
    typical_range = typical_mileage_range(model, car_year)
    mileage_typical = [_is_typical(o, typical_range) for o in miles]
    age_typical = [
        _is_typical(odometer, typical_mileage_range(model, _round_half_up(model.ref_year - a)))
        for a in ages
    ]

    t_stat = coef_t_stat(model, "age")

    return DepreciationCurves(
        mileage_curve=MileageCurve(points=[
            MileagePoint(odometer=o, price=p, lower=lo, upper=hi, typical=t)
            for o, p, lo, hi, t in zip(
                miles.tolist(), m_est.tolist(), m_lo.tolist(), m_hi.tolist(), mileage_typical
            )
        ]),
        age_curve=AgeCurve(points=[
            AgePoint(age=a, model_year=model.ref_year - a, price=p, lower=lo, upper=hi, typical=t)
            for a, p, lo, hi, t in zip(
                ages.tolist(), a_est.tolist(), a_lo.tolist(), a_hi.tolist(), age_typical
            )
        ]),
        fixed_age=age,
        fixed_odometer=float(odometer),
        odometer_is_cars=odometer_is_cars,
        car_year=car_year,
        typical_range=typical_range,
        car=CarPoint(
            age=age,
            model_year=model.ref_year - age,
            odometer=query.odometer,
            price=c_est,
            lower=c_lo,
            upper=c_hi,
            typical=_is_typical(odometer, typical_range) if odometer_is_cars else None,
        ),
        age_t_stat=t_stat,
        age_effect_clear=abs(t_stat) >= CLEAR_T_STAT,
        alpha=alpha,
    )


def boxcox_inverse(t: ArrayLike, lam: float) -> np.ndarray:
    """Undo a Box-Cox transform (map values back to original units).

    Values outside the transform's range have no real inverse. For lam > 0
    they map to 0; for lam < 0 they map to +inf, instead of producing NaN.
    """
    t = np.asarray(t, dtype=float)
    if lam == 0:
        return np.exp(t)
    base = lam * t + 1
    if lam > 0:
        return np.power(np.maximum(base, 0), 1.0 / lam)
    safe = np.where(base > 0, base, 1.0)
    return np.where(base > 0, np.power(safe, 1.0 / lam), np.inf)


def _half_width(
    x0: np.ndarray, xtx_inv: np.ndarray, resid_se: float, dof: int, alpha: float
) -> np.ndarray:
    """Half-width of the prediction interval, in transformed space, for rows x0.

    t(1 - alpha/2, dof) * s * sqrt(1 + x0^T (X^T X)^-1 x0). With one predictor
    (x0 = [1, x]) the quadratic form x0^T (X^T X)^-1 x0 equals
    1/n + (x - mean(x))^2 / Sxx, so the old one-predictor formula is a special
    case of this one.
    """
    leverage = np.einsum("...i,ij,...j->...", x0, xtx_inv, x0)
    return stats.t.ppf(1 - alpha / 2, dof) * resid_se * np.sqrt(1 + leverage)


def _typical_mileage(df: pd.DataFrame) -> list[TypicalMileage]:
    """Typical mileage for each model year in the listings (see typical_mileage_range)."""
    years = np.asarray(df["year"], dtype=int)
    odometer = np.asarray(df["odometer"], dtype=float)
    entries = []
    for year in np.unique(years):
        near = odometer[np.abs(years - year) <= TYPICAL_YEAR_WINDOW]
        low, high = np.percentile(near, TYPICAL_PERCENTILES)
        entries.append(TypicalMileage(year=int(year), n=near.size, low=float(low), high=float(high)))
    return entries


def _is_typical(odometer: float, typical_range: tuple[float, float] | None) -> bool:
    """True if the odometer is inside the range; False if outside or there is no range."""
    return bool(typical_range is not None and typical_range[0] <= odometer <= typical_range[1])


def _round_half_up(x: float) -> int:
    """Round to the nearest whole number, with .5 going up (Python's round() goes to even)."""
    return int(np.floor(x + 0.5))


class _DesignSpec(NamedTuple):
    """What _build_design needs, before a FittedModel exists (during the fit)."""

    lambda_x: float
    lambda_age: float
    columns: Sequence[str]


def _build_design(
    model: FittedModel | _DesignSpec,
    odometer: ArrayLike,
    age: ArrayLike,
    feature_values: Mapping[str, ArrayLike] | None = None,
) -> np.ndarray:
    """The design matrix: one row per car, one column per entry of model.columns.

    Columns are [1, boxcox(odometer, lambda_x), boxcox(age + 1, lambda_age),
    *features]. Feature values are looked up by name and put in the model's
    column order; odometer, age and the features are broadcast against each
    other, so any of them can be a single value. This is the only place a
    design matrix is built: fitting, the lambda search, prediction, outlier
    scoring and the curves all use it.
    """
    feature_values = feature_values or {}
    missing = [name for name in model.columns[len(BASE_COLUMNS):] if name not in feature_values]
    if missing:
        raise KeyError(
            f"The model uses feature(s) {missing}; pass their values in feature_values."
        )
    odometer, age, *extras = np.broadcast_arrays(
        np.asarray(odometer, dtype=float),
        np.asarray(age, dtype=float),
        *(np.asarray(feature_values[name], dtype=float)
          for name in model.columns[len(BASE_COLUMNS):]),
    )
    return np.stack(
        [np.ones_like(odometer), boxcox(odometer, model.lambda_x),
         boxcox(age + 1, model.lambda_age), *extras],
        axis=-1,
    )


def _log_likelihood(
    x: np.ndarray, age: np.ndarray, feature_values: Mapping[str, np.ndarray],
    columns: Sequence[str], y: np.ndarray, sum_log_y: float,
    lam_x: float, lam_age: float, lam_y: float,
) -> float:
    """Box-Cox profile log-likelihood at one set of lambdas (see fit_model)."""
    X = _build_design(_DesignSpec(lam_x, lam_age, columns), x, age, feature_values)
    rss = _rss(X, boxcox(y, lam_y)[:, None])
    return float(-(y.size / 2) * np.log(rss[0] / y.size) + (lam_y - 1) * sum_log_y)


def _grid_search(
    x: np.ndarray, age: np.ndarray, feature_values: Mapping[str, np.ndarray],
    columns: Sequence[str], y: np.ndarray, sum_log_y: float,
    lambda_range: tuple[float, float], step: float,
) -> tuple[float, float, float]:
    """Best (lambda_x, lambda_age, lambda_y) on a grid with the given step.

    For each (lambda_x, lambda_age) pair, every lambda_y is fitted in one
    least-squares call (one column of transformed prices per lambda_y).
    """
    grid = np.round(np.arange(lambda_range[0], lambda_range[1] + step / 2, step), 10)
    ty = np.column_stack([boxcox(y, ly) for ly in grid])
    best = (-np.inf, 0.0, 0.0, 0.0)
    for lx in grid:
        for la in grid:
            X = _build_design(_DesignSpec(lx, la, columns), x, age, feature_values)
            rss = _rss(X, ty)  # one RSS per lambda_y
            ll = -(y.size / 2) * np.log(rss / y.size) + (grid - 1) * sum_log_y
            i = int(np.argmax(ll))
            if ll[i] > best[0]:
                best = (float(ll[i]), float(lx), float(la), float(grid[i]))
    return best[1:]


def _rss(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Residual sum of squares of least-squares fits of each column of Y on X.

    The predictor columns are centred and scaled first. That doesn't change
    the fit or the RSS, but keeps lstsq accurate when a lambda of +/-2 makes
    the transformed values huge or tiny.
    """
    pred = X[:, 1:]
    scale = pred.std(axis=0)
    Xs = np.column_stack([X[:, 0], (pred - pred.mean(axis=0)) / np.where(scale > 0, scale, 1)])
    coef, *_ = np.linalg.lstsq(Xs, Y, rcond=None)
    resid = Y - Xs @ coef
    return np.sum(resid**2, axis=0)
