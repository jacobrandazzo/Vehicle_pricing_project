"""The Plotly charts mark the car, flag an unclear age effect, hover in dollars, and read on a phone."""

import plotly.graph_objects as go
import pytest

from vehicle_pricing.model import depreciation_curves
from vehicle_pricing.plotly_charts import age_figure, mileage_figure, residuals_figure
from vehicle_pricing.schemas import VehicleQuery

CAMRY_2015_90K = VehicleQuery(manufacturer="toyota", model="camry", year=2015, odometer=90_000)
FIGURES = pytest.mark.parametrize("make_figure", [mileage_figure, age_figure])


def _traces(fig, meta):
    """The traces tagged with this `meta` label, e.g. "car" or "age curve"."""
    return [trace for trace in fig.data if trace.meta == meta]


@pytest.fixture
def curves(camry_model):
    return depreciation_curves(camry_model, CAMRY_2015_90K)


@FIGURES
def test_each_figure_is_one_panel_with_a_short_title(make_figure, camry_listings, curves):
    fig = make_figure(camry_listings, curves, title="Toyota Camry")

    assert isinstance(fig, go.Figure)
    assert {trace.xaxis or "x" for trace in fig.data} == {"x"}  # no second panel
    assert "xaxis2" not in fig.to_dict()["layout"]
    assert fig.layout.title.text in ("Price vs. mileage", "Price vs. model year")
    assert "listings" not in fig.layout.title.text  # the long title is the app's job now


@FIGURES
def test_one_legend_below_the_plot(make_figure, camry_listings, curves):
    fig = make_figure(camry_listings, curves, title="Toyota Camry")

    assert {trace.legend or "legend" for trace in fig.data} == {"legend"}
    assert fig.layout.legend.y < 0 and fig.layout.legend.yanchor == "top"
    assert fig.layout.legend.x == 0


@FIGURES
def test_colour_bar_is_horizontal_and_below_the_plot(make_figure, camry_listings, curves):
    colorbar = make_figure(camry_listings, curves).layout.coloraxis.colorbar

    assert colorbar.orientation == "h"
    assert colorbar.y < 0
    assert colorbar.y > make_figure(camry_listings, curves).layout.legend.y  # above the legend


def test_both_figures_use_the_same_colour_limits(camry_listings, curves):
    mileage = mileage_figure(camry_listings, curves).layout.coloraxis
    age = age_figure(camry_listings, curves).layout.coloraxis

    assert (mileage.cmin, mileage.cmax) == (age.cmin, age.cmax)
    assert (mileage.cmin, mileage.cmax) == (camry_listings["year"].min(), camry_listings["year"].max())


def test_model_year_axis_has_few_horizontal_ticks(camry_listings, curves):
    axis = age_figure(camry_listings, curves).layout.xaxis
    years = camry_listings["year"]

    assert axis.tickangle == 0
    assert (years.max() - years.min()) / axis.dtick + 1 <= 6


@FIGURES
def test_only_nearby_listings_are_highlighted(make_figure, camry_listings, curves):
    fig = make_figure(camry_listings, curves, title="Toyota Camry")

    coloured = [t for t in fig.data if t.marker.coloraxis == "coloraxis"]
    faded = [t for t in fig.data if t.name == "Other listings (faded)"]
    assert len(coloured) == 1 and len(faded) == 1
    assert 0 < len(coloured[0].x) < len(camry_listings)
    assert len(coloured[0].x) + len(faded[0].x) == len(camry_listings)


@FIGURES
def test_car_marker_only_with_a_mileage(make_figure, camry_listings, camry_model, curves):
    no_mileage = CAMRY_2015_90K.model_copy(update={"odometer": None})
    without = make_figure(camry_listings, depreciation_curves(camry_model, no_mileage))

    cars = _traces(make_figure(camry_listings, curves), "car")
    assert len(cars) == 1
    assert cars[0].error_y.array is not None  # with its 95% range
    assert _traces(without, "car") == []


def test_unclear_age_effect_dashes_the_age_curve_and_says_so(camry_listings, curves):
    unclear = curves.model_copy(update={"age_effect_clear": False, "age_t_stat": 0.5})

    fig = age_figure(camry_listings, unclear)

    age_curve = _traces(fig, "age curve")
    assert age_curve and all(trace.line.dash == "dash" for trace in age_curve)
    assert any("No clear age effect" in note.text for note in fig.layout.annotations)


def test_clear_age_effect_has_a_solid_age_curve_and_no_note(camry_listings, curves):
    fig = age_figure(camry_listings, curves)

    assert any(trace.line.dash == "solid" for trace in _traces(fig, "age curve"))
    assert not fig.layout.annotations


@FIGURES
def test_hover_shows_dollars(make_figure, camry_listings, curves):
    fig = make_figure(camry_listings, curves)

    templates = [trace.hovertemplate or "" for trace in fig.data]
    assert any("%{y:$,.0f}" in template for template in templates)  # e.g. "$13,598"


def test_residuals_figure_returns_a_figure(camry_listings, camry_model, curves):
    fig = residuals_figure(camry_listings, camry_model, curves)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # residual dots and the band's two edges
