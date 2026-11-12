"""The Plotly charts have both panels, mark the car, flag an unclear age effect, and hover in dollars."""

import plotly.graph_objects as go

from vehicle_pricing.model import depreciation_curves
from vehicle_pricing.plotly_charts import depreciation_figure, residuals_figure
from vehicle_pricing.schemas import VehicleQuery

CAMRY_2015_90K = VehicleQuery(manufacturer="toyota", model="camry", year=2015, odometer=90_000)


def _traces(fig, meta):
    """The traces tagged with this `meta` label, e.g. "car" or "age curve"."""
    return [trace for trace in fig.data if trace.meta == meta]


def test_depreciation_figure_has_traces_on_both_panels(camry_listings, camry_model):
    fig = depreciation_figure(
        camry_listings, depreciation_curves(camry_model, CAMRY_2015_90K), title="Toyota Camry"
    )

    assert isinstance(fig, go.Figure)
    panels = {trace.xaxis for trace in fig.data}
    assert {"x", "x2"} <= panels  # left panel and right panel
    assert fig.layout.title.text.startswith("Toyota Camry: 44 listings")


def test_car_marker_only_with_a_mileage(camry_listings, camry_model):
    with_mileage = depreciation_figure(
        camry_listings, depreciation_curves(camry_model, CAMRY_2015_90K)
    )
    no_mileage = CAMRY_2015_90K.model_copy(update={"odometer": None})
    without = depreciation_figure(camry_listings, depreciation_curves(camry_model, no_mileage))

    cars = _traces(with_mileage, "car")
    assert len(cars) == 2  # one on each panel
    assert all(car.error_y.array is not None for car in cars)  # with its 95% range
    assert _traces(without, "car") == []


def test_unclear_age_effect_dashes_the_age_curve_and_says_so(camry_listings, camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K).model_copy(
        update={"age_effect_clear": False, "age_t_stat": 0.5}
    )

    fig = depreciation_figure(camry_listings, curves)

    age_curve = _traces(fig, "age curve")
    assert age_curve and all(trace.line.dash == "dash" for trace in age_curve)
    assert any("No clear age effect" in note.text for note in fig.layout.annotations)


def test_clear_age_effect_has_a_solid_age_curve_and_no_note(camry_listings, camry_model):
    fig = depreciation_figure(camry_listings, depreciation_curves(camry_model, CAMRY_2015_90K))

    assert any(trace.line.dash == "solid" for trace in _traces(fig, "age curve"))
    assert not fig.layout.annotations


def test_hover_shows_dollars(camry_listings, camry_model):
    fig = depreciation_figure(camry_listings, depreciation_curves(camry_model, CAMRY_2015_90K))

    templates = [trace.hovertemplate or "" for trace in fig.data]
    assert any("%{y:$,.0f}" in template for template in templates)  # e.g. "$13,598"


def test_residuals_figure_returns_a_figure(camry_listings, camry_model):
    fig = residuals_figure(
        camry_listings, camry_model, depreciation_curves(camry_model, CAMRY_2015_90K)
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # residual dots and the band's two edges
