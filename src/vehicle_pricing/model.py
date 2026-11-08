"""Fit a Box-Cox price-vs-mileage line and predict a price with a prediction interval.

The model is a straight line between Box-Cox-transformed odometer and price:

    boxcox(price, lambda_y) = intercept + slope * boxcox(odometer, lambda_x)

The two lambdas are chosen to make that relationship as straight as possible.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize
from scipy.special import boxcox

from vehicle_pricing.schemas import MIN_LISTINGS, PriceResult, VehicleQuery

DEFAULT_ALPHA = 0.05  # 95% prediction interval


class FittedModel(BaseModel):
    """Everything needed to predict from a fitted line, without the listings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_x: float  # Box-Cox exponent for odometer
    lambda_y: float  # Box-Cox exponent for price
    intercept: float  # in transformed space
    slope: float  # in transformed space
    resid_se: float  # residual standard error, in transformed space
    n: int  # number of listings the line was fit on
    x_mean: float  # mean of transformed odometer
    sxx: float  # sum of squared deviations of transformed odometer
    x_min: float  # odometer range the line was fit on, in miles
    x_max: float
    r2: float  # R^2 in transformed space


def fit_model(
    df: pd.DataFrame,
    *,
    lambda_range: tuple[float, float] = (-1.0, 1.0),
    grid_step: float = 0.1,
    min_samples: int = MIN_LISTINGS,
) -> FittedModel:
    """Fit the line to the 'odometer' and 'price' columns of the listings.

    A coarse grid search over (lambda_x, lambda_y) is refined with
    Nelder-Mead; linearity is the R^2 of a least-squares line through the
    transformed points. Box-Cox needs strictly positive inputs.
    """
    x = np.asarray(df["odometer"], dtype=float)
    y = np.asarray(df["price"], dtype=float)
    if x.shape != y.shape:
        raise ValueError("odometer and price must have the same length")
    if x.size < min_samples:
        raise ValueError(f"Need at least {min_samples} listings to fit, got {x.size}")
    if (x <= 0).any() or (y <= 0).any():
        raise ValueError("odometer and price must be strictly positive")

    grid = np.round(
        np.arange(lambda_range[0], lambda_range[1] + grid_step / 2, grid_step), 10
    )
    _, lam_x, lam_y = max(
        (_linearity_r2(x, y, lx, ly), lx, ly) for lx in grid for ly in grid
    )
    refined = minimize(
        lambda p: -_linearity_r2(x, y, *p),
        x0=[lam_x, lam_y],
        method="Nelder-Mead",
        bounds=[lambda_range, lambda_range],
    )
    lam_x, lam_y = refined.x

    tx, ty = boxcox(x, lam_x), boxcox(y, lam_y)
    reg = stats.linregress(tx, ty)
    resid = ty - (reg.intercept + reg.slope * tx)

    return FittedModel(
        lambda_x=float(lam_x),
        lambda_y=float(lam_y),
        intercept=float(reg.intercept),
        slope=float(reg.slope),
        resid_se=float(np.sqrt(np.sum(resid**2) / (x.size - 2))),
        n=int(x.size),
        x_mean=float(tx.mean()),
        sxx=float(np.sum((tx - tx.mean()) ** 2)),
        x_min=float(x.min()),
        x_max=float(x.max()),
        r2=float(reg.rvalue**2),
    )


def predict_price(
    model: FittedModel, query: VehicleQuery, alpha: float = DEFAULT_ALPHA
) -> PriceResult:
    """Predict the median price at the query's mileage, with a 1 - alpha interval."""
    if query.odometer is None:
        raise ValueError("A price prediction needs the vehicle's mileage (odometer).")
    estimate, lower, upper = predict_interval(model, query.odometer, alpha)
    return PriceResult(
        predicted_price=float(estimate),
        lower_bound=float(lower),
        upper_bound=float(upper),
        n_listings=model.n,
        extrapolated=not (model.x_min <= query.odometer <= model.x_max),
        query=query,
    )


def predict_interval(
    model: FittedModel, odometer: ArrayLike, alpha: float = DEFAULT_ALPHA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (estimate, lower, upper) prices in dollars for one or many mileages.

    The estimate is the *median* price at that mileage (back-transforming
    a mean in Box-Cox space gives a median). The bounds form a
    ``1 - alpha`` prediction interval for a single new listing.
    """
    tx = boxcox(np.asarray(odometer, dtype=float), model.lambda_x)
    y_hat = model.intercept + model.slope * tx
    t_crit = stats.t.ppf(1 - alpha / 2, model.n - 2)
    half_width = t_crit * model.resid_se * np.sqrt(
        1 + 1 / model.n + (tx - model.x_mean) ** 2 / model.sxx
    )
    return (
        boxcox_inverse(y_hat, model.lambda_y),
        boxcox_inverse(y_hat - half_width, model.lambda_y),
        boxcox_inverse(y_hat + half_width, model.lambda_y),
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


def _linearity_r2(x: np.ndarray, y: np.ndarray, lam_x: float, lam_y: float) -> float:
    """R^2 of a straight-line fit between Box-Cox transformed x and y."""
    r = np.corrcoef(boxcox(x, lam_x), boxcox(y, lam_y))[0, 1]
    return r * r
