"""Plots of the listings and a fitted price model.

Each function builds a matplotlib Figure directly (not through pyplot), so
plotting never touches pyplot's global state or needs a GUI backend. The
caller decides what to do with it: fig.savefig() to a file, plt.figure(fig)
then plt.show() in a script, or let a notebook display it.
"""

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
BAND_ALPHA = 0.15
ATYPICAL_BAND_ALPHA = 0.06  # lighter band where few real cars are like the curve's point
DASHED_NOTE = "Dashed: few cars like this;\nshape borrowed from other years"  # 2 lines: narrower legend
YEAR_COLORMAP = "viridis"  # sequential: dark = older model years, yellow = newer
HIGHLIGHT_DOT_STYLE = {"s": 8, "alpha": 1.0, "linewidths": 0, "zorder": 1.5}
FADED_DOT_STYLE = {"s": 8, "color": "grey", "alpha": 0.25, "linewidths": 0, "zorder": 1}
HIGHLIGHT_YEARS = 1  # left panel: highlight listings within +/- this many model years of the car
HIGHLIGHT_MILEAGE = 0.2  # right panel: highlight listings within +/- 20% of the fixed odometer
MILEAGE_AXIS_PERCENTILE = 99  # left panel's x-axis ends at this mileage percentile...
MILEAGE_AXIS_STEP = 50_000  # ...rounded up to a multiple of this
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
    plus a top axis showing age.

    Each curve is solid where real cars like that point exist (its `typical`
    flag), and dashed with a lighter band where they don't, because there the
    shape is borrowed from other model years. If the listings show no clear
    age effect, the whole age curve is dashed and the panel says so.

    Each curve describes one slice of the model, so each panel highlights
    the listings in that slice and fades the rest: on the left, cars within
    +/- 1 model year of the car's; on the right, cars within +/- 20% of the
    fixed mileage. Faded listings are grey. Highlighted ones are coloured by
    model year, on one scale shared by both panels (one colour bar). All dots
    are drawn first, faded under highlighted, so the curve, band and car stay
    on top. The left x-axis stops at the 99th percentile mileage,
    rounded up to the next 50,000, so a few very high-mileage cars don't
    squash the plot.

    `title` names the vehicle (make, then model, e.g. "Toyota Camry"); the
    legend uses the model for the highlighted cars, and the title adds the
    number of listings fitted on and their model years. `alpha` defaults to
    the one the curves were computed with.
    """
    alpha = _check_alpha(curves, alpha)
    band_label = f"{1 - alpha:.0%} prediction interval"
    car = curves.car
    ref_year = car.model_year + car.age  # model_year = ref_year - age throughout

    fig = Figure(figsize=(13, 5.5), layout="constrained")
    ax_miles, ax_years = fig.subplots(1, 2, sharey=True)
    years = listings["year"]
    norm = Normalize(years.min(), years.max())
    cars = _plural_name(title)

    # Left: price vs. mileage, at the car's age; highlight cars near its model year.
    mileage = curves.mileage_curve.to_frame()
    near_year = (years - curves.car_year).abs() <= HIGHLIGHT_YEARS
    low_year, high_year = curves.car_year - HIGHLIGHT_YEARS, curves.car_year + HIGHLIGHT_YEARS
    miles_scatter, miles_dots = _highlighted_scatter(
        ax_miles, listings["odometer"], listings["price"], years, near_year, norm,
        f"{low_year}–{high_year} {cars} (highlighted)",
    )
    miles_atypical = _draw_curve(ax_miles, mileage["odometer"], mileage)
    ax_miles.set(xlabel="Odometer (miles)", ylabel="Price")
    ax_miles.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax_miles.set_xlim(0, _mileage_axis_end(listings["odometer"], car.odometer))

    # Right: price vs. model year, at a fixed mileage.
    ages = curves.age_curve.to_frame()
    jitter = np.random.default_rng(JITTER_SEED).uniform(
        -JITTER_YEARS, JITTER_YEARS, size=len(listings)
    )
    fixed = curves.fixed_odometer
    near_miles = (listings["odometer"] - fixed).abs() <= HIGHLIGHT_MILEAGE * fixed
    low_miles, high_miles = (1 - HIGHLIGHT_MILEAGE) * fixed, (1 + HIGHLIGHT_MILEAGE) * fixed
    _, years_dots = _highlighted_scatter(
        ax_years, years + jitter, listings["price"], years, near_miles, norm,
        f"{cars} with {low_miles / 1000:.0f}k–{high_miles / 1000:.0f}k miles (highlighted)",
    )
    whose = "this car's mileage" if curves.odometer_is_cars else "median mileage"
    years_atypical = _draw_curve(
        ax_years, ages["model_year"], ages, dashed_everywhere=not curves.age_effect_clear
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
    _legend(
        ax_miles, miles_dots,
        f"Median at age {curves.fixed_age:.1f} years ({car.model_year:.0f} model year)",
        band_label, curve_dashed=False, dashed_note=miles_atypical, loc="upper right",
    )
    # Newer (pricier) cars sit at the top right, so this legend goes top left,
    # below the note if there is one. With no clear age effect the whole curve
    # is dashed for that reason, so the "few cars like this" entry is left out.
    below_note = {} if curves.age_effect_clear else {"bbox_to_anchor": (0, 0.9)}
    _legend(
        ax_years, years_dots, f"Median at {curves.fixed_odometer:,.0f} miles ({whose})", band_label,
        curve_dashed=not curves.age_effect_clear,
        dashed_note=years_atypical and curves.age_effect_clear,
        loc="upper left", **below_note,
    )

    colorbar = fig.colorbar(miles_scatter, ax=[ax_miles, ax_years], label="Model year")
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


def _draw_curve(ax, x: pd.Series, curve: pd.DataFrame, *, dashed_everywhere: bool = False) -> bool:
    """Draw a curve and its band: solid where typical, dashed with a lighter band where not.

    Returns True if any part is atypical. Neighbouring runs share their
    boundary point, so the pieces join up without gaps.
    """
    x = x.to_numpy()
    typical = curve["typical"].to_numpy()
    start = 0
    for end in range(1, len(x) + 1):
        if end < len(x) and typical[end] == typical[start]:
            continue
        run = slice(start, min(end + 1, len(x)))
        solid = typical[start] and not dashed_everywhere
        ax.plot(x[run], curve["price"].to_numpy()[run], color=CURVE_COLOR, linewidth=2,
                linestyle="-" if solid else "--", zorder=3)
        ax.fill_between(
            x[run], curve["lower"].to_numpy()[run], curve["upper"].to_numpy()[run],
            color=BAND_COLOR, alpha=BAND_ALPHA if typical[start] else ATYPICAL_BAND_ALPHA,
            linewidth=0, zorder=2,
        )
        start = end
    return not typical.all()


def _highlighted_scatter(ax, x, y, years, highlight, norm, label: str) -> tuple:
    """Scatter faded listings in grey, then highlighted ones coloured by model year on top.

    Returns the highlighted scatter (for the colour bar) and the legend items
    [(handle, label)]; the highlighted entry uses a stand-in marker in one
    mid-scale colour.
    """
    ax.scatter(x[~highlight], y[~highlight], **FADED_DOT_STYLE)
    shown = ax.scatter(x[highlight], y[highlight], c=years[highlight],
                       cmap=YEAR_COLORMAP, norm=norm, **HIGHLIGHT_DOT_STYLE)
    swatch = {"marker": "o", "linestyle": "", "markersize": 5, "markeredgewidth": 0}
    faded_swatch = {**swatch, "color": FADED_DOT_STYLE["color"], "alpha": FADED_DOT_STYLE["alpha"]}
    return shown, [
        (Line2D([], [], color=colormaps[YEAR_COLORMAP](0.5), **swatch), label),
        (Line2D([], [], **faded_swatch), "Other listings (faded)"),
    ]


def _plural_name(title: str) -> str:
    """'Toyota Camry' -> 'Camrys': the model (the title minus the make), made plural."""
    words = title.split()
    return " ".join(words[1:] if len(words) > 1 else words) + "s"


def _mileage_axis_end(odometer: pd.Series, car_odometer: float | None) -> float:
    """99th percentile mileage (or the car's, if higher), rounded up to the next 50,000."""
    end = np.percentile(odometer, MILEAGE_AXIS_PERCENTILE)
    if car_odometer is not None:
        end = max(end, car_odometer)
    return float(np.ceil(end / MILEAGE_AXIS_STEP) * MILEAGE_AXIS_STEP)


def _legend(ax, dot_items: list, curve_label: str, band_label: str, *, curve_dashed: bool,
            dashed_note: bool, **legend_kwargs) -> None:
    """Legend: highlighted and faded listings, curve, band, dashed-line note, then the car.

    The curve is drawn in pieces, so its entries are stand-in handles.
    """
    handles, labels = ax.get_legend_handles_labels()  # only the car, if marked
    items = [
        *dot_items,
        (Line2D([], [], color=CURVE_COLOR, linewidth=2,
                linestyle="--" if curve_dashed else "-"), curve_label),
        (Patch(color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0), band_label),
    ]
    if dashed_note:
        items.append((Line2D([], [], color=CURVE_COLOR, linewidth=2, linestyle="--"), DASHED_NOTE))
    items += list(zip(handles, labels))
    ax.legend(*zip(*items), fontsize=9, **legend_kwargs)


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
