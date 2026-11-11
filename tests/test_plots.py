"""plot_depreciation draws a figure, and flags an unclear age effect."""

from matplotlib.figure import Figure

from vehicle_pricing.model import depreciation_curves
from vehicle_pricing.plots import CURVE_COLOR, plot_depreciation, plot_residuals
from vehicle_pricing.schemas import VehicleQuery

CAMRY_2015_90K = VehicleQuery(manufacturer="toyota", model="camry", year=2015, odometer=90_000)


def _legend_labels(fig, panel):
    return [t.get_text() for t in fig.axes[panel].get_legend().get_texts()]


def _age_curve_styles(fig):
    """Line styles of the age curve's pieces (the curve-coloured lines in the right panel)."""
    return {line.get_linestyle() for line in fig.axes[1].get_lines()
            if line.get_color() == CURVE_COLOR}


def test_plot_depreciation_with_mileage_returns_figure(camry_listings, camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    fig = plot_depreciation(camry_listings, curves, title="Toyota Camry")

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    assert "-" in _age_curve_styles(fig)  # clear age effect: solid where typical
    assert "2014–2016 Camrys (highlighted)" in _legend_labels(fig, 0)
    assert "Camrys with 72k–108k miles (highlighted)" in _legend_labels(fig, 1)
    assert "Other listings (faded)" in _legend_labels(fig, 0)
    assert fig.axes[0].get_xlim()[1] % 50_000 == 0  # mileage axis ends on a round 50k


def test_plot_depreciation_without_mileage_returns_figure(camry_listings, camry_model):
    query = CAMRY_2015_90K.model_copy(update={"odometer": None})
    curves = depreciation_curves(camry_model, query)

    fig = plot_depreciation(camry_listings, curves, title="Toyota Camry")

    assert isinstance(fig, Figure)
    labels = [t.get_text() for ax in fig.axes[:2] for t in ax.get_legend().get_texts()]
    assert not any(label.startswith("This car") for label in labels)  # car not marked


def test_plot_depreciation_dashes_unclear_age_curve(camry_listings, camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K).model_copy(
        update={"age_effect_clear": False, "age_t_stat": 0.5}
    )

    fig = plot_depreciation(camry_listings, curves, title="Toyota Camry")

    assert _age_curve_styles(fig) == {"--"}  # dashed everywhere
    notes = [t.get_text() for t in fig.axes[1].texts]
    assert any("No clear age effect" in note for note in notes)


def test_plot_residuals_returns_figure(camry_listings, camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    assert isinstance(plot_residuals(camry_listings, camry_model, curves), Figure)


def test_plot_depreciation_for_a_range_query(camry_listings, camry_model):
    # Like the Civic baseline query: a year range, no single year.
    query = VehicleQuery(manufacturer="toyota", model="camry", year_min=2012, year_max=2016,
                         odometer=120_000)
    curves = depreciation_curves(camry_model, query)

    fig = plot_depreciation(camry_listings, curves, title="Toyota Camry")

    assert isinstance(fig, Figure)
    assert "2013–2015 Camrys (highlighted)" in _legend_labels(fig, 0)  # midpoint 2014
