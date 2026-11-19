"""Interactive Plotly versions of the depreciation and residual charts, for the website.

They follow the same rules as the matplotlib charts in plots.py (which
listings are highlighted, where curves turn dashed, the axis limits and all
the wording), and reuse its data and text helpers, so the two can't drift
apart. Hovering shows the values: price, mileage, model year and age for a
listing; the median and 95% range for a curve point and for the car.
Price vs. mileage and price vs. model year are two separate one-panel
figures, so a page can put them side by side or stack them on a phone.

The colours are their own: the "Ocean" style, made for the website's dark
page (transparent backgrounds, light text, a blue model-year ramp, a coral
curve). The matplotlib charts and the docs images keep the colours in plots.py.

This module only builds figures; it doesn't display them.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vehicle_pricing.model import DepreciationCurves, FittedModel
from vehicle_pricing.plots import (
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

# "Ocean" colours, for a dark page (Streamlit's dark background is about #171614).
PAGE_COLOR = "#171614"  # the page behind the charts; used for rings around light marks
TRANSPARENT = "rgba(0, 0, 0, 0)"
TEXT_COLOR = "#edebe8"  # titles, axis titles, legend
TICK_COLOR = "#a8a6a0"  # tick labels: a step quieter than the text
GRID_COLOR = "rgba(237, 235, 232, 0.10)"  # horizontal gridlines, barely there
CURVE_COLOR = "#ff4b4b"  # coral, the site's accent colour
BAND_RGB = "255, 75, 75"  # the same coral, for the bands' rgba()
BAND_ALPHA = 0.15
ATYPICAL_BAND_ALPHA = 0.06  # lighter where few real cars are like the curve's point
RESIDUAL_BAND = f"rgba({BAND_RGB}, {BAND_ALPHA})"
# Model-year scale, oldest (deep blue) to newest (pale blue): newer cars stand out on dark.
YEAR_COLORSCALE = [[0, "#2b4a7a"], [1 / 3, "#3f7cc0"], [2 / 3, "#6bb3f0"], [1, "#bfe3ff"]]
MID_YEAR_COLOR = "#5598d8"  # the scale's middle (halfway from #3f7cc0 to #6bb3f0), for the legend
FADED_DOTS = {"color": TICK_COLOR, "opacity": 0.35, "size": 5}
HIGHLIGHT_DOTS = {"size": 6}
CAR_COLOR = "#ffffff"  # white: black would vanish on the dark page
# One dark box with light text. Every hover names its trace on the first line and
# leaves <extra></extra> empty: Plotly's side box takes the trace's colour, unreadable here.
HOVER_LABEL = {"bgcolor": "#262522", "bordercolor": TICK_COLOR, "font": {"color": TEXT_COLOR}}
NOTE_BOX = {"bgcolor": "rgba(38, 37, 34, 0.9)", "bordercolor": TICK_COLOR,
            "font": {"color": TEXT_COLOR}}
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
    hover = ("Mileage curve<br>Median: %{y:$,.0f}<br>" + level
             + " range: %{customdata[0]:$,.0f}–%{customdata[1]:$,.0f}"
             + "<br>Mileage: %{x:,.0f} miles<extra></extra>")
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
    hover = ("Age curve<br>Median: %{y:$,.0f}<br>" + level
             + " range: %{customdata[0]:$,.0f}–%{customdata[1]:$,.0f}"
             + "<br>Model year: %{x:.1f}<br>Age: %{customdata[2]:.1f} years"
             + "<extra></extra>")
    atypical = _add_curve(fig, ages["model_year"], ages,
                          ages[["lower", "upper", "age"]].to_numpy(), hover, meta="age curve",
                          dashed_everywhere=not curves.age_effect_clear)
    if not curves.age_effect_clear:
        fig.add_annotation(
            text=_age_note(curves).replace(" (t =", "<br>(t ="),  # two lines: fits at 360px
            xref="x domain", yref="y domain", x=0.5, y=0.98,
            xanchor="center", yanchor="top", showarrow=False,
            borderwidth=1, **NOTE_BOX,
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
        marker=FADED_DOTS, showlegend=False,
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
        title={"text": title, "x": 0.5}, height=DEFAULT_HEIGHT,
        legend={"x": 0.99, "xanchor": "right", "y": 0.99, "yanchor": "top",
                "bgcolor": "rgba(23, 22, 20, 0.85)"},  # the page colour, so dots under it dim
    )
    _dark_layout(fig)
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


def _dark_layout(fig: go.Figure) -> None:
    """Transparent backgrounds, light text and faint horizontal gridlines, for the dark page.

    With no template the page shows through: paper_bgcolor is behind the whole
    figure (title, legend, margins), plot_bgcolor behind the plotting area.
    """
    fig.update_layout(
        template="none", paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        font={"color": TEXT_COLOR}, hoverlabel=HOVER_LABEL,
    )
    # automargin: grow the margins to fit the tick labels (template "none" leaves it off).
    axis = {"tickfont": {"color": TICK_COLOR}, "showline": False, "zeroline": False,
            "ticks": "", "automargin": True}
    # Fixed gaps below the x-axis: numbers clear of the axis (and of "$0"), title close to them.
    fig.update_xaxes(showgrid=False, ticklabelstandoff=6, title_standoff=8, **axis)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, **axis)


def _finish_layout(fig: go.Figure, listings: pd.DataFrame, title: str) -> None:
    """Short title; colour bar and legend under the x-axis title, so nothing covers the dots.

    The model-year colours use the full listings' range in both figures, so the
    same model year is the same colour in each.
    """
    years = listings["year"]
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
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
                "font": {"size": 11}, "bgcolor": TRANSPARENT},
    )
    _dark_layout(fig)
    fig.update_yaxes(title_text="Price", tickformat="$,.0f", rangemode="tozero")


def _legend_text(label: str) -> str:
    """Break a long legend entry onto two lines before its "(...)", so it fits on a phone."""
    return label.replace(" (", "<br>(", 1) if len(label) > LEGEND_WRAP else label


def _add_car(fig, curves, x):
    """The car: a white dot with a dark ring at its estimated price, its 95% range as an error bar."""
    car = curves.car
    fig.add_trace(go.Scatter(
        x=[x], y=[car.price], mode="markers", name=_legend_text(_car_label(curves)), meta="car",
        marker={"color": CAR_COLOR, "size": 10, "line": {"color": PAGE_COLOR, "width": 1.5}},
        error_y={"type": "data", "symmetric": False, "array": [car.upper - car.price],
                 "arrayminus": [car.price - car.lower], "color": CAR_COLOR,
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
        hovertemplate="Other listing<br>" + hover + "<extra></extra>",
        legendrank=2,
    ))
    fig.add_trace(go.Scatter(
        x=x[shown], y=y[shown], mode="markers", showlegend=False,
        marker={**HIGHLIGHT_DOTS, "color": years[shown], "coloraxis": "coloraxis"},
        customdata=customdata[shown], hovertemplate="Listing<br>" + hover + "<extra></extra>",
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
