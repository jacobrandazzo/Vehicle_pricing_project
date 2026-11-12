"""Interactive Plotly versions of the depreciation and residual charts, for the website.

They follow the same rules as the matplotlib charts in plots.py (which
listings are highlighted, where curves turn dashed, the axis limits and all
the wording), and reuse its data and text helpers, so the two can't drift
apart. Hovering shows the values: price, mileage, model year and age for a
listing; the median and 95% range for a curve point and for the car.
Age is in the right panel's hover text rather than on a second axis.

This module only builds figures; it doesn't display them.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    _title_text,
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
DEFAULT_HEIGHT = 560

# Hover text. %{...:$,.0f} means dollars with thousands separators and no cents.
LISTING_HOVER = ("Price: %{y:$,.0f}<br>Mileage: %{customdata[0]:,.0f} miles"
                 "<br>Model year: %{customdata[1]}")
LISTING_HOVER_WITH_AGE = LISTING_HOVER + "<br>Age: %{customdata[2]:.1f} years"


def depreciation_figure(
    listings: pd.DataFrame, curves: DepreciationCurves, *, title: str = "Vehicle"
) -> go.Figure:
    """Price against mileage (left) and against model year (right), with the car marked.

    The Plotly version of plots.plot_depreciation, with the same panels and
    rules: each panel highlights the listings its curve describes (coloured
    by model year on one shared scale) and fades the rest in grey; curves are
    solid where real cars like that point exist and dashed with a lighter
    band where they don't; with no clear age effect, the whole age curve is
    dashed and a note says so; the car is marked with its 95% range when the
    query has a mileage.
    """
    level = f"{1 - curves.alpha:.0%}"  # "95%"
    band_label = f"{level} prediction interval"
    car = curves.car
    years = listings["year"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05)

    # Left: price vs. mileage, at the car's age.
    near_year, near_year_label = _year_highlight(listings, curves, title)
    mileage_data = np.column_stack([listings["odometer"], years])
    _add_listings(fig, 1, listings["odometer"], listings["price"], years, near_year,
                  mileage_data, LISTING_HOVER, legend="legend")
    mileage = curves.mileage_curve.to_frame()
    mileage_hover = ("Median: %{y:$,.0f}<br>" + level
                     + " range: %{customdata[0]:$,.0f}–%{customdata[1]:$,.0f}"
                     + "<br>Mileage: %{x:,.0f} miles<extra>Mileage curve</extra>")
    miles_atypical = _add_curve(
        fig, 1, mileage["odometer"], mileage, mileage[["lower", "upper"]].to_numpy(),
        mileage_hover, meta="mileage curve", legend="legend",
    )

    # Right: price vs. model year (jittered sideways), at a fixed mileage.
    near_miles, near_miles_label = _mileage_highlight(listings, curves, title)
    years_data = np.column_stack([listings["odometer"], years, listings["age"]])
    _add_listings(fig, 2, years + _year_jitter(len(listings)), listings["price"], years,
                  near_miles, years_data, LISTING_HOVER_WITH_AGE, legend="legend2")
    ages = curves.age_curve.to_frame()
    age_hover = ("Median: %{y:$,.0f}<br>" + level
                 + " range: %{customdata[0]:$,.0f}–%{customdata[1]:$,.0f}"
                 + "<br>Model year: %{x:.1f}<br>Age: %{customdata[2]:.1f} years"
                 + "<extra>Age curve</extra>")
    years_atypical = _add_curve(
        fig, 2, ages["model_year"], ages, ages[["lower", "upper", "age"]].to_numpy(),
        age_hover, meta="age curve", legend="legend2",
        dashed_everywhere=not curves.age_effect_clear,
    )
    if not curves.age_effect_clear:
        fig.add_annotation(
            text=_age_note(curves), xref="x2 domain", yref="y2 domain", x=0.5, y=0.98,
            xanchor="center", yanchor="top", showarrow=False,
            bgcolor="lightyellow", bordercolor="grey", borderwidth=1,
        )

    if car.price is not None:
        car_hover = (f"This car<br>Price: ${car.price:,.0f}<br>{level} range: "
                     f"${car.lower:,.0f}–${car.upper:,.0f}<extra></extra>")
        for col, x, legend in ((1, car.odometer, "legend"), (2, car.model_year, "legend2")):
            fig.add_trace(go.Scatter(
                x=[x], y=[car.price], mode="markers", name=_car_label(curves), meta="car",
                marker={"color": "black", "size": 10, "line": {"color": "white", "width": 1.5}},
                error_y={"type": "data", "symmetric": False, "array": [car.upper - car.price],
                         "arrayminus": [car.price - car.lower], "color": "black",
                         "thickness": 1.5, "width": 5},
                hovertemplate=car_hover, legend=legend, legendrank=6,
            ), row=1, col=col)

    # Legends: one inside each panel, as in the matplotlib chart.
    _add_legend_keys(fig, 1, "legend", near_year_label, _mileage_curve_label(curves),
                     band_label, curve_dashed=False, dashed_note=miles_atypical)
    _add_legend_keys(fig, 2, "legend2", near_miles_label, _age_curve_label(curves),
                     band_label, curve_dashed=not curves.age_effect_clear,
                     dashed_note=years_atypical and curves.age_effect_clear)
    legend_box = {"bgcolor": "rgba(255, 255, 255, 0.85)", "bordercolor": "lightgrey",
                  "borderwidth": 1, "font": {"size": 11}, "yanchor": "top"}
    note_space = 0 if curves.age_effect_clear else 0.1  # keep the legend below the note

    span = int(years.max() - years.min())
    fig.update_layout(
        title={"text": _title_text(title, listings), "x": 0.5},
        template="plotly_white",
        height=DEFAULT_HEIGHT,
        hovermode="closest",
        coloraxis={"colorscale": YEAR_COLORSCALE, "cmin": years.min(), "cmax": years.max(),
                   "colorbar": {"title": {"text": "Model year"}, "tickformat": "d",
                                "dtick": max(1, int(np.ceil(span / 10)))}},
        legend={**legend_box, "x": 0.47, "xanchor": "right", "y": 1},
        legend2={**legend_box, "x": 0.53, "xanchor": "left", "y": 1 - note_space},
    )
    fig.update_xaxes(title_text="Odometer (miles)", tickformat=",.0f",
                     range=[0, _mileage_axis_end(listings["odometer"], car.odometer)],
                     row=1, col=1)
    fig.update_xaxes(title_text="Model year", tickformat="d", row=1, col=2)
    fig.update_yaxes(tickformat="$,.0f", rangemode="tozero")
    fig.update_yaxes(title_text="Price", row=1, col=1)
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


def _add_listings(fig, col, x, y, years, highlight, customdata, hover, *, legend):
    """Faded listings in grey, then the highlighted ones coloured by model year on top."""
    faded, shown = ~highlight.to_numpy(), highlight.to_numpy()
    fig.add_trace(go.Scatter(
        x=x[faded], y=y[faded], mode="markers", marker=FADED_DOTS,
        name="Other listings (faded)", customdata=customdata[faded],
        hovertemplate=hover + "<extra>Other listing</extra>",
        legend=legend, legendrank=2,
    ), row=1, col=col)
    fig.add_trace(go.Scatter(
        x=x[shown], y=y[shown], mode="markers", showlegend=False,
        marker={**HIGHLIGHT_DOTS, "color": years[shown], "coloraxis": "coloraxis"},
        customdata=customdata[shown], hovertemplate=hover + "<extra>Listing</extra>",
        legend=legend,
    ), row=1, col=col)


def _add_curve(fig, col, x, curve, customdata, hover, *, meta, legend,
               dashed_everywhere=False) -> bool:
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
        common = {"x": x[run], "mode": "lines", "showlegend": False, "legend": legend}
        fig.add_trace(go.Scatter(**common, y=upper[run], line={"width": 0},
                                 hoverinfo="skip", meta=f"{meta} band"), row=1, col=col)
        fig.add_trace(go.Scatter(**common, y=lower[run], line={"width": 0}, fill="tonexty",
                                 fillcolor=f"rgba({BAND_RGB}, {opacity})",
                                 hoverinfo="skip", meta=f"{meta} band"), row=1, col=col)
        solid = is_typical and not dashed_everywhere
        fig.add_trace(go.Scatter(
            **common, y=price[run], customdata=customdata[run], hovertemplate=hover, meta=meta,
            line={"color": CURVE_COLOR, "width": 2, "dash": "solid" if solid else "dash"},
        ), row=1, col=col)
    return not typical.all()


def _add_legend_keys(fig, col, legend, highlight_label, curve_label, band_label, *,
                     curve_dashed, dashed_note):
    """Legend entries for things drawn in pieces or with many colours (no data of their own)."""
    keys = [
        (highlight_label, {"mode": "markers",
                           "marker": {"color": MID_YEAR_COLOR, "size": 8}}, 1),
        (curve_label, {"mode": "lines", "line": {"color": CURVE_COLOR, "width": 2,
                                                  "dash": "dash" if curve_dashed else "solid"}}, 3),
        (band_label, {"mode": "markers", "marker": {"symbol": "square", "size": 14,
                                                     "color": f"rgba({BAND_RGB}, {BAND_ALPHA})"}}, 4),
    ]
    if dashed_note:
        keys.append((DASHED_NOTE.replace("\n", "<br>"),
                     {"mode": "lines", "line": {"color": CURVE_COLOR, "width": 2, "dash": "dash"}}, 5))
    for name, style, rank in keys:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legend=legend, legendrank=rank,
                                 hoverinfo="skip", **style), row=1, col=col)
