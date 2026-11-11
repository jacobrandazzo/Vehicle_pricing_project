"""Plots of the listings and a fitted price model.

Each function builds a matplotlib Figure directly (not through pyplot), so
plotting never touches pyplot's global state or needs a GUI backend. The
caller decides what to do with it: fig.savefig() to a file, plt.figure(fig)
then plt.show() in a script, or let a notebook display it.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from vehicle_pricing.model import (
    DepreciationCurves,
    FittedModel,
    predict_interval,
)

DOLLARS = StrMethodFormatter("${x:,.0f}")
LISTING_STYLE = {"s": 6, "color": "0.6", "alpha": 0.35, "linewidths": 0}
CURVE_COLOR = "darkred"  # dark enough to stand out over the coloured dots
BAND_COLOR = "tab:red"
YEAR_COLORMAP = "viridis"  # sequential: dark = older model years, yellow = newer
YEAR_DOT_STYLE = {"s": 8, "alpha": 0.6, "linewidths": 0, "zorder": 1}
JITTER_YEARS = 0.3  # spreads listings of one model year sideways so they don't overlap
JITTER_SEED = 0


def plot_depreciation(
    listings: pd.DataFrame,
    curves: DepreciationCurves,
    *,
    title: str = "Vehicle",
    alpha: float | None = None,
) -> Figure:
    """Price against mileage (left) and against model year (right), with the car marked.

    Both panels share the price axis. Left: the listings and the mileage
    curve, with age held at the car's age. Right: the listings and the age
    curve, with mileage held fixed (the car's own, or the listings' median),
    plus a top axis showing age. If the listings show no clear age effect,
    the age curve is dashed and the panel says so.

    In both panels the listings are coloured by model year, with one shared
    colour bar. They are drawn first, so the curve, band and car stay on top.

    `title` names the vehicle; the number of listings and their model years
    are added. `alpha` defaults to the one the curves were computed with.
    """
    alpha = _check_alpha(curves, alpha)
    band_label = f"{1 - alpha:.0%} prediction interval"
    car = curves.car
    ref_year = car.model_year + car.age  # model_year = ref_year - age throughout

    fig = Figure(figsize=(13, 5.5), layout="constrained")
    ax_miles, ax_years = fig.subplots(1, 2, sharey=True)
    years = listings["year"]
    dots = {"c": years, "cmap": YEAR_COLORMAP, "norm": Normalize(years.min(), years.max()),
            "label": "Listings, colored by model year", **YEAR_DOT_STYLE}

    # Left: price vs. mileage, at the car's age.
    mileage = curves.mileage_curve.to_frame()
    scatter = ax_miles.scatter(listings["odometer"], listings["price"], **dots)
    ax_miles.plot(
        mileage["odometer"], mileage["price"], color=CURVE_COLOR, linewidth=2, zorder=3,
        label=f"Median at age {curves.fixed_age:.1f} years ({car.model_year:.0f} model year)",
    )
    ax_miles.fill_between(
        mileage["odometer"], mileage["lower"], mileage["upper"],
        color=BAND_COLOR, alpha=0.15, zorder=2, label=band_label,
    )
    ax_miles.set(xlabel="Odometer (miles)", ylabel="Price")
    ax_miles.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # Right: price vs. model year, at a fixed mileage.
    ages = curves.age_curve.to_frame()
    jitter = np.random.default_rng(JITTER_SEED).uniform(
        -JITTER_YEARS, JITTER_YEARS, size=len(listings)
    )
    ax_years.scatter(years + jitter, listings["price"], **dots)
    whose = "this car's mileage" if curves.odometer_is_cars else "median mileage"
    ax_years.plot(
        ages["model_year"], ages["price"], color=CURVE_COLOR, linewidth=2, zorder=3,
        linestyle="-" if curves.age_effect_clear else "--",
        label=f"Median at {curves.fixed_odometer:,.0f} miles ({whose})",
    )
    ax_years.fill_between(
        ages["model_year"], ages["lower"], ages["upper"],
        color=BAND_COLOR, alpha=0.15, zorder=2, label=band_label,
    )
    ax_years.set(xlabel="Model year")
    ax_years.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    top = ax_years.secondary_xaxis(
        "top", functions=(lambda year: ref_year - year, lambda age: ref_year - age)
    )
    top.set_xlabel("Age (years)")
    if not curves.age_effect_clear:
        ax_years.text(
            0.5, 0.97, f"No clear age effect in these listings (t = {curves.age_t_stat:.2f})",
            transform=ax_years.transAxes, ha="center", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "edgecolor": "0.5"},
        )

    if car.price is not None:
        yerr = [[car.price - car.lower], [car.upper - car.price]]
        label = (f"This car: ${car.price:,.0f} "
                 f"({1 - alpha:.0%} range ${car.lower:,.0f}–${car.upper:,.0f})")
        marker = {"yerr": yerr, "fmt": "o", "color": "black", "markersize": 8,
                  "markeredgecolor": "white", "markeredgewidth": 1.5, "capsize": 4,
                  "zorder": 4, "label": label}
        ax_miles.errorbar(car.odometer, car.price, **marker)
        ax_years.errorbar(car.model_year, car.price, **marker)

    ax_miles.set_ylim(bottom=0)
    ax_miles.yaxis.set_major_formatter(DOLLARS)
    ax_miles.legend(loc="upper right", fontsize=9)
    # Newer (pricier) cars sit at the top right, so this legend goes top left,
    # below the note if there is one.
    below_note = {} if curves.age_effect_clear else {"bbox_to_anchor": (0, 0.9)}
    ax_years.legend(loc="upper left", fontsize=9, **below_note)

    colorbar = fig.colorbar(scatter, ax=[ax_miles, ax_years], label="Model year")
    colorbar.locator = MaxNLocator(integer=True)  # whole years only, never "1998, 1998"
    colorbar.update_ticks()
    span = f"{years.min()}" if years.min() == years.max() else f"{years.min()}–{years.max()}"
    fig.suptitle(f"{title}: {len(listings):,} listings, model years {span}")
    return fig


def plot_residuals(
    listings: pd.DataFrame,
    model: FittedModel,
    curves: DepreciationCurves,
    *,
    title: str = "Residuals vs. odometer",
) -> Figure:
    """Residuals (in dollars) of the full mileage-and-age model, against odometer.

    Each listing's residual uses its own mileage and age. The band is the
    prediction interval from curves.mileage_curve, at the car's age.
    """
    band = curves.mileage_curve.to_frame()
    fig = Figure(figsize=(8, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(listings["odometer"], _residuals(listings, model), **LISTING_STYLE)
    ax.axhline(0, color=CURVE_COLOR, linewidth=1)
    ax.fill_between(
        band["odometer"], band["lower"] - band["price"], band["upper"] - band["price"],
        color=CURVE_COLOR, alpha=0.15,
        label=f"{1 - curves.alpha:.0%} prediction interval at age {curves.fixed_age:.1f} years",
    )
    ax.set(title=title, xlabel="Odometer (miles)", ylabel="Residual")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.legend()
    return fig


def _check_alpha(curves: DepreciationCurves, alpha: float | None) -> float:
    """The curves' alpha; a different one would mislabel bands computed at the curves' alpha."""
    if alpha is not None and alpha != curves.alpha:
        raise ValueError(
            f"The curves were computed with alpha={curves.alpha}; "
            f"recompute them with depreciation_curves(..., alpha={alpha})."
        )
    return curves.alpha


def _residuals(listings: pd.DataFrame, model: FittedModel) -> np.ndarray:
    """Actual price minus the fitted median price (at each listing's mileage and age), in dollars."""
    estimate, _, _ = predict_interval(model, listings["odometer"], listings["age"])
    return np.asarray(listings["price"], dtype=float) - estimate
