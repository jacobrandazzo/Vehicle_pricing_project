"""Plots of a fitted vehicle price model (still to be moved into vehicle_pricing).

Loading, cleaning and filtering live in vehicle_pricing.data; fitting and
prediction live in vehicle_pricing.model. Typical use::

    from vehicle_pricing.data import filter_listings, load_data
    from vehicle_pricing.model import fit_model
    from legacy_vehicle_pricing import plot_fit

    subset = filter_listings(load_data(), query)
    fit = fit_model(subset)
    plot_fit(fit, subset["odometer"], subset["price"], title="Toyota")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from vehicle_pricing.model import DEFAULT_ALPHA, FittedModel, predict_interval


def _curve(fit: FittedModel, n_points: int = 100, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Estimate and interval over the fitted odometer range (for plotting)."""
    odometer = np.linspace(fit.x_min, fit.x_max, n_points)
    estimate, lower, upper = predict_interval(fit, odometer, alpha)
    return pd.DataFrame(
        {"odometer": odometer, "estimate": estimate, "lower": lower, "upper": upper}
    )


def _residuals(fit: FittedModel, odometer: ArrayLike, price: ArrayLike) -> np.ndarray:
    """Actual price minus the fitted median price, in dollars."""
    estimate, _, _ = predict_interval(fit, odometer)
    return np.asarray(price, dtype=float) - estimate


# --------------------------------------------------------------------------- #
# Plotting (functions return the figure; call plt.show() or fig.savefig())
# --------------------------------------------------------------------------- #

def plot_fit(
    fit: FittedModel,
    odometer: ArrayLike,
    price: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    title: str = "Price vs. odometer",
) -> Figure:
    """Scatterplot of the listings with the fitted median and prediction interval."""
    curve = _curve(fit, alpha=alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(odometer, price, s=6, alpha=0.25, label="Listings")
    ax.plot(
        curve["odometer"], curve["estimate"], color="red",
        label=(f"Box-Cox fit (median), R² = {fit.r2:.3f}  "
               f"[λx = {fit.lambda_x:.2f}, λy = {fit.lambda_y:.2f}]"),
    )
    ax.fill_between(
        curve["odometer"], curve["lower"], curve["upper"],
        color="red", alpha=0.15, label=f"{1 - alpha:.0%} prediction interval",
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set(title=f"{title} (n = {fit.n:,})", xlabel="Odometer (miles)", ylabel="Price ($)")
    fig.tight_layout()
    return fig


def plot_residuals(
    fit: FittedModel,
    odometer: ArrayLike,
    price: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    title: str = "Residuals vs. odometer",
) -> Figure:
    """Residuals (in dollars) against odometer, with the interval band."""
    curve = _curve(fit, alpha=alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(odometer, _residuals(fit, odometer, price), s=6, alpha=0.25)
    ax.axhline(0, color="red", linewidth=1)
    ax.fill_between(
        curve["odometer"],
        curve["lower"] - curve["estimate"],
        curve["upper"] - curve["estimate"],
        color="red", alpha=0.15, label=f"{1 - alpha:.0%} prediction interval",
    )
    ax.set(title=title, xlabel="Odometer (miles)", ylabel="Residual ($)")
    ax.legend()
    return fig
