"""Interactive Plotly versions of the depreciation and residual charts, for the website.

They follow the same rules as the matplotlib charts in plots.py (which
listings are highlighted, where curves turn dashed, the axis limits and all
the wording), and reuse its data and text helpers, so the two can't drift
apart. Hovering shows the values: price, mileage, model year and age for a
listing; the median and 95% range for a curve point and for the car.
Price vs. mileage and price vs. model year are two separate one-panel
figures, so a page can put them side by side or stack them on a phone.

This module only builds figures; it doesn't display them.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vehicle_pricing.model import DepreciationCurves, FittedModel
from vehicle_pricing.plots import (
    ATYPICAL_BAND_ALPHA,
    BAND_ALPHA,
    CURVE_COLOR,
    DASHED_NOTE,
    _age_curve_label,
    _age_note,
    _car_label,
    _mileage_axis_end,
    _mileage_curve_label,
    _mileage_highlight,
    _residuals,
    _typical_runs,
    _year_highlight,
    _year_jitter,
)

BAND_RGB = "214, 39, 40"  # matplotlib's "tab:red", the band colour in plots.py
RESIDUAL_BAND = "rgba(139, 0, 0, 0.15)"  # "darkred" at 15%, as in plots.plot_residuals
YEAR_COLORSCALE = "Viridis"  # the same colour map as plots.YEAR_COLORMAP
MID_YEAR_COLOR = "#21918c"  # viridis' middle colour, for the highlighted dots' legend entry
FADED_DOTS = {"color": "grey", "opacity": 0.25, "size": 5}
HIGHLIGHT_DOTS = {"size": 6}
DEFAULT_HEIGHT = 560  # the residuals chart
# Mileage and age charts: one panel each, readable from a phone (360px) up. Their
# colour bar and legend sit below the x-axis title, in a bottom margin of fixed size.
FIGURE_HEIGHT = 660
MARGIN = {"t": 50, "b": 270, "l": 10, "r": 20}  # the left grows to fit the price labels
COLORBAR_GAP = 55  # pixels from the x-axis down to the colour bar's top
LEGEND_GAP = 115  # ...and down to the legend's top
LEGEND_WRAP = 36  # legend entries longer than this break before their "(...)"

# Hover text. %{...:$,.0f} means dollars with thousands separators and no cents.
LISTING_HOVER = ("Price: %{y:$,.0f}<br>Mileage: %{customdata[0]:,.0f} miles"
                 "<br>Model year: %{customdata[1]}")
LISTING_HOVER_WITH_AGE = LISTING_HOVER + "<br>Age: %{customdata[2]:.1f} years"


def mileage_figure(
    listings: pd.DataFrame, curves: DepreciationCurves, *, title: str = "Vehicle"
) -> go.Figure:
    """Price against mileage at the car's age, with the car marked.

    The left panel of plots.plot_depreciation, with the same rules: it
    highlights listings within a model year of the car (coloured by model
    year) and fades the rest in grey; the curve is solid where real cars like
    that point exist and dashed with a lighter band where they don't; the car
    is marked with its 95% range when the query has a mileage. `title` is the
    vehicle's name ("Toyota Camry"), used in the legend.
    """
    level = f"{1 - curves.alpha:.0%}"  # "95%"
    band_label = f"{level} prediction interval"
    car = curves.car
    years = listings["year"]
    fig = go.Figure()

    near_year, near_year_label = _year_highlight(listings, curves, title)
    _add_listings(fig, listings["odometer"], listings["price"], years, near_year,
                  np.column_stack([listings["odometer"], years]), LISTING_HOVER)
    mileage = curves.mileage_curve.to_frame()
    hover = ("Median: %{y:$,.0f}<br>" + level
             + " range: %{customdata[0]:$,.0f}–%{customdata[1]:$,.0f}"
             + "<br>Mileage: %{x:,.0f} miles<extra>Mileage curve</extra>")
    atypical = _add_curve(fig, mileage["odometer"], mileage,
                          mileage[["lower", "upper"]].to_numpy(), hover, meta="mileage curve")
    if car.price is not None:
        _add_car(fig, curves, car.odometer)
    _add_legend_keys(fig, near_year_label, _mileage_curve_label(curves), band_label,
                     curve_dashed=False, dashed_note=atypical)

    _finish_layout(fig, listings, "Price vs. mileage")
    fig.update_xaxes(title_text="Odometer (miles)", tickformat=",.0f",
                     range=[0, _mileage_axis_end(listings["odometer"], car.odometer)])
    return fig


def age_figure(
    listings: pd.DataFrame, curves: DepreciationCurves, *, title: str = "Vehicle"
) -> go.Figure:
    """Price against model year at a fixed mileage, with the car marked.

    The right panel of plots.plot_depreciation, with the same rules: it
    highlights listings within 20% of the fixed mileage and fades the rest;
    listings are spread sideways within their model year so they don't pile
    up; with no clear age effect the whole curve is dashed and a note says so.
    Age is in the hover text rather than on a second axis.
    """
    level = f"{1 - curves.alpha:.0%}"
    band_label = f"{level} prediction interval"
    car = curves.car
    years = listings["year"]
    fig = go.Figure()

    near_miles, near_miles_label = _mileage_highlight(listings, curves, title)
    _add_listings(fig, years + _year_jitter(len(listings)), listings["price"], years, near_miles,
                  np.column_stack([listings["odometer"], years, listings["age"]]),
                  LISTING_HOVER_WITH_AGE)
    ages = curves.age_curve.to_frame()
    hover = ("Median: %{y:$,.0f}<br>" + level
             + " range: %{customdata[0]:$,.0f}–%{customdata[1]:$,.0f}"
             + "<br>Model year: %{x:.1f}<br>Age: %{customdata[2]:.1f} years"
             + "<extra>Age curve</extra>")
    atypical = _add_curve(fig, ages["model_year"], ages,
                          ages[["lower", "upper", "age"]].to_numpy(), hover, meta="age curve",
                          dashed_everywhere=not curves.age_effect_clear)
    if not curves.age_effect_clear:
        fig.add_annotation(
            text=_age_note(curves).replace(" (t =", "<br>(t ="),  # two lines: fits at 360px
            xref="x domain", yref="y domain", x=0.5, y=0.98,
            xanchor="center", yanchor="top", showarrow=False,
            bgcolor="lightyellow", bordercolor="grey", borderwidth=1,
        )
    if car.price is not None:
        _add_car(fig, curves, car.model_year)
    _add_legend_keys(fig, near_miles_label, _age_curve_label(curves), band_label,
                     curve_dashed=not curves.age_effect_clear,
                     dashed_note=atypical and curves.age_effect_clear)

    _finish_layout(fig, listings, "Price vs. model year")
    fig.update_xaxes(title_text="Model year", tickformat="d", tickangle=0,
                     dtick=_year_step(years))
    return fig


def residuals_figure(
    listings: pd.DataFrame,
    model: FittedModel,
    curves: DepreciationCurves,
    *,
    title: str = "Residuals vs. odometer",
) -> go.Figure:
    """Residuals (in dollars) of the full model against odometer: the Plotly plot_residuals.

    Each listing's residual uses its own mileage and age. The band is the
    prediction interval from curves.mileage_curve, at the car's age.
    """
    band = curves.mileage_curve.to_frame()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=listings["odometer"], y=_residuals(listings, model), mode="markers",
        marker={"color": "grey", "opacity": 0.35, "size": 5}, showlegend=False,
        customdata=np.column_stack([listings["price"], listings["year"]]),
        hovertemplate=("Residual: %{y:$,.0f}<br>Price: %{customdata[0]:$,.0f}"
                       "<br>Mileage: %{x:,.0f} miles<br>Model year: %{customdata[1]}"
                       "<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=band["odometer"], y=band["upper"] - band["price"], mode="lines",
        line={"width": 0}, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=band["odometer"], y=band["lower"] - band["price"], mode="lines",
        line={"width": 0}, fill="tonexty", fillcolor=RESIDUAL_BAND, hoverinfo="skip",
        name=f"{1 - curves.alpha:.0%} prediction interval at age {curves.fixed_age:.1f} years",
    ))
    fig.add_hline(y=0, line={"color": CURVE_COLOR, "width": 1})
    fig.update_layout(
        title={"text": title, "x": 0.5}, template="plotly_white", height=DEFAULT_HEIGHT,
        legend={"x": 0.99, "xanchor": "right", "y": 0.99, "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.85)"},
    )
    fig.update_xaxes(title_text="Odometer (miles)", tickformat=",.0f")
    fig.update_yaxes(title_text="Residual", tickformat="$,.0f")
    return fig


def _year_step(years: pd.Series) -> int:
    """Model years between ticks, so there are at most about 6 (e.g. every 2 years over 10)."""
    return max(1, int(np.ceil(int(years.max() - years.min()) / 5)))


def _below_axis(pixels: float) -> float:
    """A position `pixels` below the x-axis, in paper units (1 = the plot area's height).

    Paper units scale with the plot area, so they only mean a fixed number of
    pixels because the figure's height and margins are fixed.
    """
    return -pixels / (FIGURE_HEIGHT - MARGIN["t"] - MARGIN["b"])


def _finish_layout(fig: go.Figure, listings: pd.DataFrame, title: str) -> None:
    """Short title; colour bar and legend under the x-axis title, so nothing covers the dots.

    The model-year colours use the full listings' range in both figures, so the
    same model year is the same colour in each.
    """
    years = listings["year"]
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        height=FIGURE_HEIGHT,
        margin=MARGIN,
        hovermode="closest",
        coloraxis={"colorscale": YEAR_COLORSCALE, "cmin": years.min(), "cmax": years.max(),
                   "colorbar": {"orientation": "h", "title": {"text": "Model year", "side": "top"},
                                "tickformat": "d", "dtick": _year_step(years),
                                "thickness": 10, "len": 1, "x": 0, "xanchor": "left",
                                "y": _below_axis(COLORBAR_GAP), "yanchor": "top"}},
        # traceorder keeps legendrank's order (Plotly otherwise reverses it because of
        # the filled bands). Long entries are wrapped by _legend_text.
        legend={"orientation": "v", "x": 0, "xanchor": "left",
                "y": _below_axis(LEGEND_GAP), "yanchor": "top", "traceorder": "normal",
                "font": {"size": 11}, "bgcolor": "rgba(0, 0, 0, 0)"},
    )
    fig.update_yaxes(title_text="Price", tickformat="$,.0f", rangemode="tozero")


def _legend_text(label: str) -> str:
    """Break a long legend entry onto two lines before its "(...)", so it fits on a phone."""
    return label.replace(" (", "<br>(", 1) if len(label) > LEGEND_WRAP else label


def _add_car(fig, curves, x):
    """The car: a black dot at its estimated price, with its 95% range as an error bar."""
    car = curves.car
    fig.add_trace(go.Scatter(
        x=[x], y=[car.price], mode="markers", name=_legend_text(_car_label(curves)), meta="car",
        marker={"color": "black", "size": 10, "line": {"color": "white", "width": 1.5}},
        error_y={"type": "data", "symmetric": False, "array": [car.upper - car.price],
                 "arrayminus": [car.price - car.lower], "color": "black",
                 "thickness": 1.5, "width": 5},
        hovertemplate=(f"This car<br>Price: ${car.price:,.0f}<br>{1 - curves.alpha:.0%} range: "
                       f"${car.lower:,.0f}–${car.upper:,.0f}<extra></extra>"),
        legendrank=6,
    ))


def _add_listings(fig, x, y, years, highlight, customdata, hover):
    """Faded listings in grey, then the highlighted ones coloured by model year on top."""
    faded, shown = ~highlight.to_numpy(), highlight.to_numpy()
    fig.add_trace(go.Scatter(
        x=x[faded], y=y[faded], mode="markers", marker=FADED_DOTS,
        name="Other listings (faded)", customdata=customdata[faded],
        hovertemplate=hover + "<extra>Other listing</extra>",
        legendrank=2,
    ))
    fig.add_trace(go.Scatter(
        x=x[shown], y=y[shown], mode="markers", showlegend=False,
        marker={**HIGHLIGHT_DOTS, "color": years[shown], "coloraxis": "coloraxis"},
        customdata=customdata[shown], hovertemplate=hover + "<extra>Listing</extra>",
    ))


def _add_curve(fig, x, curve, customdata, hover, *, meta, dashed_everywhere=False) -> bool:
    """A curve and its band, solid where typical and dashed (lighter band) where not.

    Uses the same run-splitting as plots._draw_curve. Each run's band is an
    upper line plus a lower line filled up to it. Returns True if any part is
    atypical.
    """
    x = x.to_numpy()
    typical = curve["typical"].to_numpy()
    price, lower, upper = (curve[c].to_numpy() for c in ("price", "lower", "upper"))
    for run, is_typical in _typical_runs(typical):
        opacity = BAND_ALPHA if is_typical else ATYPICAL_BAND_ALPHA
        common = {"x": x[run], "mode": "lines", "showlegend": False}
        fig.add_trace(go.Scatter(**common, y=upper[run], line={"width": 0},
                                 hoverinfo="skip", meta=f"{meta} band"))
        fig.add_trace(go.Scatter(**common, y=lower[run], line={"width": 0}, fill="tonexty",
                                 fillcolor=f"rgba({BAND_RGB}, {opacity})",
                                 hoverinfo="skip", meta=f"{meta} band"))
        solid = is_typical and not dashed_everywhere
        fig.add_trace(go.Scatter(
            **common, y=price[run], customdata=customdata[run], hovertemplate=hover, meta=meta,
            line={"color": CURVE_COLOR, "width": 2, "dash": "solid" if solid else "dash"},
        ))
    return not typical.all()


def _add_legend_keys(fig, highlight_label, curve_label, band_label, *, curve_dashed, dashed_note):
    """Legend entries for things drawn in pieces or with many colours (no data of their own)."""
    keys = [
        (_legend_text(highlight_label),
         {"mode": "markers", "marker": {"color": MID_YEAR_COLOR, "size": 8}}, 1),
        (_legend_text(curve_label),
         {"mode": "lines", "line": {"color": CURVE_COLOR, "width": 2,
                                    "dash": "dash" if curve_dashed else "solid"}}, 3),
        (band_label, {"mode": "markers", "marker": {"symbol": "square", "size": 14,
                                                     "color": f"rgba({BAND_RGB}, {BAND_ALPHA})"}}, 4),
    ]
    if dashed_note:
        keys.append((DASHED_NOTE.replace("\n", "<br>"),
                     {"mode": "lines", "line": {"color": CURVE_COLOR, "width": 2, "dash": "dash"}}, 5))
    for name, style, rank in keys:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legendrank=rank,
                                 hoverinfo="skip", **style))
