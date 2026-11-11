"""Plots of the listings and a fitted price model.

Each function builds its own figure and returns it without showing it, so the
caller decides what to do: plt.show() in a script, fig.savefig() to a file, or
let a notebook display it.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from vehicle_pricing.model import DEFAULT_ALPHA, FittedModel, predict_interval, query_age
from vehicle_pricing.schemas import PriceResult, VehicleQuery


def plot_fit(
    listings: pd.DataFrame,
    model: FittedModel,
    result: PriceResult | None = None,
    *,
    query: VehicleQuery | None = None,
    alpha: float = DEFAULT_ALPHA,
    title: str = "Price vs. odometer",
) -> Figure:
    """Scatterplot of price vs. odometer, with the fitted median and prediction interval.

    The curve holds age at the car's age (query_age). The car comes from
    `query`, or from `result` if no query is given; with neither, the
    listings' median age is used. If a PriceResult is given, its prediction
    is marked with its interval.
    """
    if query is None and result is not None:
        query = result.query
    age = query_age(model, query) if query is not None else model.median_age
    curve = _curve(model, age, alpha=alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(listings["odometer"], listings["price"], s=6, alpha=0.25, label="Listings")
    ax.plot(
        curve["odometer"], curve["estimate"], color="red",
        label=(f"Box-Cox fit (median) at age {age:.1f} years, R² = {model.r2:.3f}\n"
               f"[λx = {model.lambda_x:.2f}, λa = {model.lambda_age:.2f}, "
               f"λy = {model.lambda_y:.2f}]"),
    )
    ax.fill_between(
        curve["odometer"], curve["lower"], curve["upper"],
        color="red", alpha=0.15, label=f"{1 - alpha:.0%} prediction interval",
    )
    if result is not None:
        _mark_prediction(ax, result)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set(title=f"{title} (n = {model.n:,})", xlabel="Odometer (miles)", ylabel="Price ($)")
    fig.tight_layout()
    return fig


def plot_residuals(
    listings: pd.DataFrame,
    model: FittedModel,
    *,
    alpha: float = DEFAULT_ALPHA,
    title: str = "Residuals vs. odometer",
) -> Figure:
    """Residuals (in dollars) of the full mileage-and-age model, against odometer.

    Each listing's residual uses its own age. The band is the prediction
    interval at the listings' median age.
    """
    curve = _curve(model, model.median_age, alpha=alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(listings["odometer"], _residuals(listings, model), s=6, alpha=0.25)
    ax.axhline(0, color="red", linewidth=1)
    ax.fill_between(
        curve["odometer"],
        curve["lower"] - curve["estimate"],
        curve["upper"] - curve["estimate"],
        color="red", alpha=0.15,
        label=f"{1 - alpha:.0%} prediction interval at median age {model.median_age:.1f} years",
    )
    ax.set(title=title, xlabel="Odometer (miles)", ylabel="Residual ($)")
    ax.legend()
    return fig


def _mark_prediction(ax, result: PriceResult) -> None:
    """Draw the predicted price at the query's mileage, with error bars for its interval."""
    ax.errorbar(
        result.query.odometer, result.predicted_price,
        yerr=[[result.predicted_price - result.lower_bound],
              [result.upper_bound - result.predicted_price]],
        fmt="o", color="black", capsize=4, zorder=3,
        label=f"This vehicle: ${result.predicted_price:,.0f}",
    )


def _curve(
    model: FittedModel, age: float, n_points: int = 100, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    """Estimate and interval over the fitted odometer range, with age held fixed."""
    odometer = np.linspace(model.odometer_min, model.odometer_max, n_points)
    estimate, lower, upper = predict_interval(model, odometer, age, alpha)
    return pd.DataFrame(
        {"odometer": odometer, "estimate": estimate, "lower": lower, "upper": upper}
    )


def _residuals(listings: pd.DataFrame, model: FittedModel) -> np.ndarray:
    """Actual price minus the fitted median price (at each listing's mileage and age), in dollars."""
    estimate, _, _ = predict_interval(model, listings["odometer"], listings["age"])
    return np.asarray(listings["price"], dtype=float) - estimate
