"""plot_depreciation draws a figure, and flags an unclear age effect."""

from matplotlib.figure import Figure

from vehicle_pricing.model import depreciation_curves
from vehicle_pricing.plots import plot_depreciation, plot_residuals
from vehicle_pricing.schemas import VehicleQuery

CAMRY_2015_90K = VehicleQuery(manufacturer="toyota", model="camry", year=2015, odometer=90_000)


def _age_curve_line(fig):
    """The age curve: the red line in the right-hand panel."""
    return fig.axes[1].get_lines()[0]


def test_plot_depreciation_with_mileage_returns_figure(camry_listings, camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    fig = plot_depreciation(camry_listings, curves, title="Toyota Camry")

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    assert _age_curve_line(fig).get_linestyle() == "-"  # clear age effect: solid


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

    assert _age_curve_line(fig).get_linestyle() == "--"
    notes = [t.get_text() for t in fig.axes[1].texts]
    assert any("No clear age effect" in note for note in notes)


def test_plot_residuals_returns_figure(camry_listings, camry_model):
    curves = depreciation_curves(camry_model, CAMRY_2015_90K)

    assert isinstance(plot_residuals(camry_listings, camry_model, curves), Figure)
