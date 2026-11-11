"""Save the README's depreciation figures, from fixed queries (no Claude, no API key).

    python make_figures.py

Writes docs/camry_depreciation.png (a clear age effect) and
docs/civic_depreciation.png (no clear age effect, so the age curve is dashed).
"""

from pathlib import Path

from make_baseline import OFFLINE_QUERIES
from vehicle_pricing.data import filter_listings, load_data
from vehicle_pricing.model import depreciation_curves, drop_outliers, fit_model
from vehicle_pricing.plots import plot_depreciation

DOCS = Path(__file__).parent / "docs"
DPI = 150

FIGURES = {
    "2015 Camry, 90,000 miles": ("Toyota Camry", "camry_depreciation.png"),
    "1996-2000 Honda Civic 200,000 miles": ("Honda Civic", "civic_depreciation.png"),
}


def main():
    vehicles = load_data()
    DOCS.mkdir(exist_ok=True)
    for description, (title, filename) in FIGURES.items():
        query = OFFLINE_QUERIES[description]
        listings = drop_outliers(filter_listings(vehicles, query))  # same pipeline as main.py
        curves = depreciation_curves(fit_model(listings), query)
        path = DOCS / filename
        plot_depreciation(listings, curves, title=title).savefig(path, dpi=DPI)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
