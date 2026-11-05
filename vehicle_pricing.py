"""Vehicle price modelling for Craigslist-style listings.

Fits a straight line between Box-Cox-transformed odometer and price, then
reports a median price estimate with a prediction interval in dollars.

Typical use (e.g. in a notebook, or as agent tools)::

    from vehicle_pricing import load_vehicles, fit_query, estimate_price, plot_fit

    vehicles = load_vehicles("vehicles.csv")

    # One-shot, JSON-friendly answer:
    estimate_price(vehicles, 160_000, manufacturer="toyota", year_range=(1996, 2026))

    # Or keep the pieces for plotting:
    subset, fit = fit_query(vehicles, manufacturer="toyota", year_range=(1996, 2026))
    plot_fit(fit, subset["odometer"], subset["price"], title="Toyota")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import boxcox

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_ALPHA = 0.05  # 95% prediction interval
MIN_SAMPLES = 30  # refuse to fit on fewer listings than this

# Columns kept when loading the (large) Craigslist CSV.
KEEP_COLUMNS = frozenset({
    "price", "year", "manufacturer", "model", "odometer", "region", "state",
    "condition", "cylinders", "fuel", "title_status", "transmission",
    "drive", "size", "type", "paint_color",
})

# Rows that match on all of these are treated as duplicate postings.
DEDUP_COLUMNS = ["region", "year", "manufacturer", "model", "price", "odometer"]


# --------------------------------------------------------------------------- #
# Data loading, cleaning and filtering
# --------------------------------------------------------------------------- #

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


def load_vehicles(path: str | Path, **clean_kwargs: Any) -> pd.DataFrame:
    """Read the Craigslist CSV (only the columns we use) and clean it."""
    raw = pd.read_csv(path, usecols=lambda col: col in KEEP_COLUMNS)
    return clean_vehicles(raw, **clean_kwargs)


def filter_vehicles(
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


# --------------------------------------------------------------------------- #
# Box-Cox model
# --------------------------------------------------------------------------- #

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


@dataclass(frozen=True)
class BoxCoxLine:
    """A fitted line ``boxcox(price) = intercept + slope * boxcox(odometer)``."""

    lambda_x: float
    lambda_y: float
    slope: float
    intercept: float
    r2: float  # R^2 in transformed space
    r2_original: float  # R^2 of price vs. odometer without transforming
    n: int
    resid_se: float  # residual standard error in transformed space
    x_mean: float  # mean of transformed odometer
    sxx: float  # sum of squared deviations of transformed odometer
    x_min: float  # odometer range the line was fit on
    x_max: float

    def predict(
        self, odometer: ArrayLike, alpha: float = DEFAULT_ALPHA
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (estimate, lower, upper) prices in dollars.

        The estimate is the *median* price at that mileage (back-transforming
        a mean in Box-Cox space gives a median). The bounds form a
        ``1 - alpha`` prediction interval for a single new listing.
        """
        tx = boxcox(np.asarray(odometer, dtype=float), self.lambda_x)
        y_hat = self.intercept + self.slope * tx
        t_crit = stats.t.ppf(1 - alpha / 2, self.n - 2)
        half_width = t_crit * self.resid_se * np.sqrt(
            1 + 1 / self.n + (tx - self.x_mean) ** 2 / self.sxx
        )
        return (
            boxcox_inverse(y_hat, self.lambda_y),
            boxcox_inverse(y_hat - half_width, self.lambda_y),
            boxcox_inverse(y_hat + half_width, self.lambda_y),
        )

    def curve(self, n_points: int = 100, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
        """Estimate and interval over the fitted odometer range (for plotting)."""
        odometer = np.linspace(self.x_min, self.x_max, n_points)
        estimate, lower, upper = self.predict(odometer, alpha)
        return pd.DataFrame(
            {"odometer": odometer, "estimate": estimate, "lower": lower, "upper": upper}
        )

    def residuals(self, odometer: ArrayLike, price: ArrayLike) -> np.ndarray:
        """Actual price minus the fitted median price, in dollars."""
        estimate, _, _ = self.predict(odometer)
        return np.asarray(price, dtype=float) - estimate


def fit_boxcox_line(
    odometer: ArrayLike,
    price: ArrayLike,
    *,
    lambda_range: tuple[float, float] = (-1.0, 1.0),
    grid_step: float = 0.1,
    min_samples: int = MIN_SAMPLES,
) -> BoxCoxLine:
    """Find the Box-Cox exponents that make price vs. odometer most linear.

    A coarse grid search over (lambda_x, lambda_y) is refined with
    Nelder-Mead; linearity is the R^2 of a least-squares line through the
    transformed points. Box-Cox needs strictly positive inputs.
    """
    x = np.asarray(odometer, dtype=float)
    y = np.asarray(price, dtype=float)
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

    return BoxCoxLine(
        lambda_x=float(lam_x),
        lambda_y=float(lam_y),
        slope=float(reg.slope),
        intercept=float(reg.intercept),
        r2=float(reg.rvalue**2),
        r2_original=float(stats.pearsonr(x, y)[0] ** 2),
        n=int(x.size),
        resid_se=float(np.sqrt(np.sum(resid**2) / (x.size - 2))),
        x_mean=float(tx.mean()),
        sxx=float(np.sum((tx - tx.mean()) ** 2)),
        x_min=float(x.min()),
        x_max=float(x.max()),
    )


# --------------------------------------------------------------------------- #
# Query-level helpers (what an agent would call)
# --------------------------------------------------------------------------- #

def fit_query(
    df: pd.DataFrame, *, min_samples: int = MIN_SAMPLES, **filters: Any
) -> tuple[pd.DataFrame, BoxCoxLine]:
    """Filter the listings, then fit the model to what is left.

    ``filters`` are passed to :func:`filter_vehicles`. Returns the matching
    listings and the fitted line.
    """
    subset = filter_vehicles(df, **filters)
    fit = fit_boxcox_line(subset["odometer"], subset["price"], min_samples=min_samples)
    return subset, fit


def estimate_price(
    df: pd.DataFrame,
    odometer: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = MIN_SAMPLES,
    **filters: Any,
) -> dict[str, Any]:
    """Estimate the price of a vehicle with ``odometer`` miles. JSON-friendly.

    Raises ValueError if too few listings match ``filters`` -- an agent should
    report that rather than guess.
    """
    _, fit = fit_query(df, min_samples=min_samples, **filters)
    estimate, lower, upper = fit.predict(odometer, alpha)
    return {
        "estimate": round(float(estimate), 2),
        "lower": round(float(lower), 2),
        "upper": round(float(upper), 2),
        "interval_level": 1 - alpha,
        "n_listings": fit.n,
        "r2": round(fit.r2, 3),
        "extrapolated": not (fit.x_min <= odometer <= fit.x_max),
        "filters": {k: v for k, v in filters.items() if v is not None},
    }


# --------------------------------------------------------------------------- #
# Plotting (functions return the figure; call plt.show() or fig.savefig())
# --------------------------------------------------------------------------- #

def plot_fit(
    fit: BoxCoxLine,
    odometer: ArrayLike,
    price: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    title: str = "Price vs. odometer",
) -> Figure:
    """Scatterplot of the listings with the fitted median and prediction interval."""
    curve = fit.curve(alpha=alpha)
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
    fit: BoxCoxLine,
    odometer: ArrayLike,
    price: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    title: str = "Residuals vs. odometer",
) -> Figure:
    """Residuals (in dollars) against odometer, with the interval band."""
    curve = fit.curve(alpha=alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(odometer, fit.residuals(odometer, price), s=6, alpha=0.25)
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
