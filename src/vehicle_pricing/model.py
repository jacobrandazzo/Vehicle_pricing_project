"""Fit price on mileage and age with Box-Cox transforms, and predict with an interval.

The model is a plane in Box-Cox-transformed space:

    boxcox(price, lambda_y) = b0 + b1 * boxcox(odometer, lambda_x)
                                 + b2 * boxcox(age + 1, lambda_age)

Age is in years (see data.vehicle_age); 1 is added because Box-Cox needs
positive inputs and a car can be 0 years old. The coefficients come from
least squares. The three lambdas are chosen by maximising the Box-Cox profile
log-likelihood (see fit_model). Predictions back-transform to dollars, so the
estimate is a median price, with a prediction interval for a single listing.
depreciation_curves returns price against mileage and against age, as data.
"""

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
N_COEFS = 3  # b0, b1, b2
MIN_AGE_SPREAD = 1.0  # years between the youngest and oldest listing needed to fit age
CURVE_POINTS = 100  # points along each depreciation curve
CLEAR_T_STAT = 2.0  # |t| at or above this counts as a clear age effect


class FittedModel(BaseModel):
    """Everything needed to predict from a fitted model, without the listings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_x: float  # Box-Cox exponent for odometer
    lambda_age: float  # Box-Cox exponent for age + 1
    lambda_y: float  # Box-Cox exponent for price
    b0: float  # intercept, in transformed space
    b1: float  # coefficient on transformed odometer
    b2: float  # coefficient on transformed age + 1
    xtx_inv: list[list[float]]  # (X^T X)^-1 of the 3-column design matrix
    resid_se: float  # residual standard error s, in transformed space (n - 3 df)
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


def fit_model(
    df: pd.DataFrame,
    *,
    lambda_range: tuple[float, float] = (-2.0, 2.0),
    grid_step: float = 0.25,
    min_samples: int = MIN_LISTINGS,
) -> FittedModel:
    """Fit the model to the listings' 'odometer', 'age', 'price' and 'posting_date' columns.

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

    a1 = age + 1
    sum_log_y = float(np.sum(np.log(y)))

    lam_x, lam_age, lam_y = _grid_search(x, a1, y, sum_log_y, lambda_range, grid_step)
    refined = minimize(
        lambda p: -_log_likelihood(x, a1, y, sum_log_y, *p),
        x0=[lam_x, lam_age, lam_y],
        method="Nelder-Mead",
        bounds=[lambda_range] * 3,
    )
    lam_x, lam_age, lam_y = (float(v) for v in refined.x)

    X = _design(boxcox(x, lam_x), boxcox(a1, lam_age))
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
        b0=float(coef[0]),
        b1=float(coef[1]),
        b2=float(coef[2]),
        xtx_inv=(r_inv @ r_inv.T).tolist(),
        resid_se=float(np.sqrt(rss / (x.size - N_COEFS))),
        n=int(x.size),
        odometer_min=float(x.min()),
        odometer_max=float(x.max()),
        median_odometer=float(np.median(x)),
        age_min=float(age.min()),
        age_max=float(age.max()),
        median_age=float(np.median(age)),
        ref_year=float(np.median(decimal_year(df["posting_date"]))),
        r2=float(1 - rss / np.sum((ty - ty.mean()) ** 2)),
        log_likelihood=_log_likelihood(x, a1, y, sum_log_y, lam_x, lam_age, lam_y),
    )


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


def predict_price(
    model: FittedModel, query: VehicleQuery, alpha: float = DEFAULT_ALPHA
) -> PriceResult:
    """Predict the median price at the query's mileage and age, with a 1 - alpha interval."""
    if query.odometer is None:
        raise ValueError("A price prediction needs the vehicle's mileage (odometer).")
    age = query_age(model, query)
    estimate, lower, upper = predict_interval(model, query.odometer, age, alpha)
    return PriceResult(
        predicted_price=float(estimate),
        lower_bound=float(lower),
        upper_bound=float(upper),
        n_listings=model.n,
        extrapolated=not (
            model.odometer_min <= query.odometer <= model.odometer_max
            and model.age_min <= age <= model.age_max
        ),
        query=query,
    )


def predict_interval(
    model: FittedModel, odometer: ArrayLike, age: ArrayLike, alpha: float = DEFAULT_ALPHA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (estimate, lower, upper) prices in dollars for one or many cars.

    odometer (miles) and age (years) are broadcast against each other, so
    either can be a single value. The estimate is the *median* price
    (back-transforming a mean in Box-Cox space gives a median). The bounds
    form a ``1 - alpha`` prediction interval for a single new listing.
    """
    odometer, age = np.broadcast_arrays(
        np.asarray(odometer, dtype=float), np.asarray(age, dtype=float)
    )
    x0 = _design(boxcox(odometer, model.lambda_x), boxcox(age + 1, model.lambda_age))
    y_hat = x0 @ np.array([model.b0, model.b1, model.b2])
    half_width = _half_width(
        x0, np.array(model.xtx_inv), model.resid_se, model.n - N_COEFS, alpha
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


class AgePoint(BaseModel):
    """One point on the age curve. Prices are in US dollars."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    age: float  # years
    model_year: float  # ref_year - age, a decimal year
    price: float  # median price
    lower: float  # prediction interval
    upper: float


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

    The price fields are None when the query has no mileage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    age: float  # years, from query_age
    model_year: float  # ref_year - age, a decimal year
    odometer: float | None  # miles, from the query
    price: float | None
    lower: float | None
    upper: float | None


class DepreciationCurves(BaseModel):
    """How the fitted price changes with mileage and with age, for one car.

    The mileage curve holds age at fixed_age (the car's own age). The age
    curve holds mileage at fixed_odometer: the car's mileage if the query has
    one (odometer_is_cars = True), otherwise the listings' median mileage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mileage_curve: MileageCurve
    age_curve: AgeCurve
    fixed_age: float  # years; age held constant along the mileage curve
    fixed_odometer: float  # miles; mileage held constant along the age curve
    odometer_is_cars: bool  # True: fixed_odometer is the car's; False: the median
    car: CarPoint
    age_t_stat: float  # b2 divided by its standard error
    age_effect_clear: bool  # |age_t_stat| >= CLEAR_T_STAT


def depreciation_curves(model: FittedModel, query: VehicleQuery) -> DepreciationCurves:
    """Price curves for one car: against mileage, and against age. Data only, no plotting.

    - Mileage curve: price over the fitted mileage range, with the car's age
      held fixed, so it shows what extra miles alone cost this car.
    - Age curve: price over the fitted age range, with mileage held fixed at
      the car's mileage (or the listings' median mileage if the query has
      none), so it shows what extra years alone cost.

    Each curve has CURVE_POINTS evenly spaced points, each with the median
    price and its prediction interval. Everything, including the car's own
    point, comes from predict_interval, so the car sits exactly on both curves.

    age_t_stat says how clearly the listings show an age effect once mileage
    is accounted for: the age coefficient b2 divided by its standard error.
    If |t| >= 2, the effect is about two standard errors or more from zero,
    so it is unlikely to be just noise (roughly a 95% test), and
    age_effect_clear is True. If |t| < 2, the age curve's slope may not mean
    much: these listings can't separate age from mileage and noise.
    """
    age = query_age(model, query)
    odometer_is_cars = query.odometer is not None
    odometer = query.odometer if odometer_is_cars else model.median_odometer

    miles = np.linspace(model.odometer_min, model.odometer_max, CURVE_POINTS)
    m_est, m_lo, m_hi = predict_interval(model, miles, age)
    ages = np.linspace(model.age_min, model.age_max, CURVE_POINTS)
    a_est, a_lo, a_hi = predict_interval(model, odometer, ages)

    if odometer_is_cars:
        c_est, c_lo, c_hi = (float(v) for v in predict_interval(model, odometer, age))
    else:
        c_est = c_lo = c_hi = None

    b2_se = model.resid_se * np.sqrt(model.xtx_inv[2][2])
    t_stat = float(model.b2 / b2_se)

    return DepreciationCurves(
        mileage_curve=MileageCurve(points=[
            MileagePoint(odometer=o, price=p, lower=lo, upper=hi)
            for o, p, lo, hi in zip(miles.tolist(), m_est.tolist(), m_lo.tolist(), m_hi.tolist())
        ]),
        age_curve=AgeCurve(points=[
            AgePoint(age=a, model_year=model.ref_year - a, price=p, lower=lo, upper=hi)
            for a, p, lo, hi in zip(ages.tolist(), a_est.tolist(), a_lo.tolist(), a_hi.tolist())
        ]),
        fixed_age=age,
        fixed_odometer=float(odometer),
        odometer_is_cars=odometer_is_cars,
        car=CarPoint(
            age=age,
            model_year=model.ref_year - age,
            odometer=query.odometer,
            price=c_est,
            lower=c_lo,
            upper=c_hi,
        ),
        age_t_stat=t_stat,
        age_effect_clear=abs(t_stat) >= CLEAR_T_STAT,
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


def _design(*predictors: np.ndarray) -> np.ndarray:
    """Design matrix: a column of ones, then one column per predictor (last axis)."""
    return np.stack([np.ones_like(predictors[0]), *predictors], axis=-1)


def _log_likelihood(
    x: np.ndarray, a1: np.ndarray, y: np.ndarray, sum_log_y: float,
    lam_x: float, lam_age: float, lam_y: float,
) -> float:
    """Box-Cox profile log-likelihood at one set of lambdas (see fit_model)."""
    rss = _rss(_design(boxcox(x, lam_x), boxcox(a1, lam_age)), boxcox(y, lam_y)[:, None])
    return float(-(y.size / 2) * np.log(rss[0] / y.size) + (lam_y - 1) * sum_log_y)


def _grid_search(
    x: np.ndarray, a1: np.ndarray, y: np.ndarray, sum_log_y: float,
    lambda_range: tuple[float, float], step: float,
) -> tuple[float, float, float]:
    """Best (lambda_x, lambda_age, lambda_y) on a grid with the given step.

    For each (lambda_x, lambda_age) pair, every lambda_y is fitted in one
    least-squares call (one column of transformed prices per lambda_y).
    """
    grid = np.round(np.arange(lambda_range[0], lambda_range[1] + step / 2, step), 10)
    ty = np.column_stack([boxcox(y, ly) for ly in grid])
    tas = [boxcox(a1, la) for la in grid]
    best = (-np.inf, 0.0, 0.0, 0.0)
    for lx in grid:
        tx = boxcox(x, lx)
        for la, ta in zip(grid, tas):
            rss = _rss(_design(tx, ta), ty)  # one RSS per lambda_y
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
