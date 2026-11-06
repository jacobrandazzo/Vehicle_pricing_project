import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; never try to open a window from inside the agent loop
import matplotlib.pyplot as plt

import legacy_vehicle_pricing as vp

DATA_PATH = Path(__file__).parent / "vehicles_slim.csv.gz"
PLOTS_DIR = Path(__file__).parent / "plots"

vehicles = vp.load_vehicles(DATA_PATH)

# Filters that match a column exactly (case-insensitive). Claude may set any of these.
CATEGORICAL_FILTERS = ("manufacturer", "model", "condition", "fuel", "transmission", "cylinders", "drive")

# Filter parameters shared by all three tools (everything except odometer).
FILTER_PROPERTIES = {
    "manufacturer": {"type": "string", "description": "Make, e.g. 'toyota' or 'ford'."},
    "model": {"type": "string", "description": "Model name, e.g. 'camry'."},
    "year_min": {"type": "integer", "description": "Earliest model year of comparison listings."},
    "year_max": {"type": "integer", "description": "Latest model year of comparison listings."},
    "condition": {"type": "string", "description": "Listed condition, e.g. 'good' or 'excellent'."},
    "fuel": {"type": "string", "description": "Fuel type, e.g. 'gas', 'diesel', 'hybrid', 'electric'."},
    "transmission": {"type": "string", "description": "e.g. 'automatic' or 'manual'."},
    "cylinders": {"type": "string", "description": "Engine cylinders as listed in the data, e.g. '6 cylinders'."},
    "drive": {"type": "string", "description": "Drivetrain, e.g. '4wd', 'fwd', 'rwd'."},
}

PLOT_NOTE = (
    "The plot is saved as an image file and you cannot see it. "
    "Only tell the user where it was saved and how many listings it uses; do not describe what it shows. "
)

tools = [
    {
        "name": "estimate_price",
        "description": (
            "Estimate the typical sale price of a used vehicle, with a 95% prediction interval, "
            "using Craigslist listings. The model compares price against mileage for listings that "
            "match the given filters, then predicts the price at the given odometer reading. "
            "Only set a filter if the user mentioned it. More filters means fewer comparison listings "
            "and a less reliable estimate. If the result has 'extrapolated': true, the mileage is "
            "outside the range of the listings, so say the estimate is less reliable. "
            "If the result contains 'error', explain it and suggest loosening the filters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **FILTER_PROPERTIES,
                "odometer": {"type": "number", "description": "Mileage of the vehicle being priced, in miles."},
            },
            "required": ["manufacturer", "odometer"],
        },
    },
    {
        "name": "plot_fit",
        "description": (
            "Create a scatter plot of listing price against mileage for the matching listings, with the "
            "fitted price curve and its prediction interval. " + PLOT_NOTE +
            "If the result contains 'error', explain it and suggest loosening the filters."
        ),
        "input_schema": {
            "type": "object",
            "properties": FILTER_PROPERTIES,
            "required": ["manufacturer"],
        },
    },
    {
        "name": "plot_residuals",
        "description": (
            "Create a residual plot (actual price minus fitted price, against mileage) for the "
            "matching listings, to check how well the fitted curve describes them."
             + PLOT_NOTE +
            "If the result contains 'error', explain it and suggest loosening the filters."
        ),
        "input_schema": {
            "type": "object",
            "properties": FILTER_PROPERTIES, 
            "required": ["manufacturer"],
        },
    },
]


def _range(low, high, name):
    """Return (low, high) if either end is set, None if neither. Either end may be None."""
    if low is None and high is None:
        return None
    if low is not None and high is not None and low > high:
        raise ValueError(f"{name}_min cannot be greater than {name}_max")
    return (low, high)


def _build_filters(tool_input):
    """Turn Claude's tool arguments into filter_vehicles keyword arguments (set filters only)."""
    filters = {k: tool_input[k] for k in CATEGORICAL_FILTERS if tool_input.get(k) is not None}
    year_range = _range(tool_input.get("year_min"), tool_input.get("year_max"), "year")
    if year_range is not None:
        filters["year_range"] = year_range
    return filters


def _title(filters):
    """A short plot title built from the manufacturer and model."""
    name = " ".join(str(filters[k]) for k in ("manufacturer", "model") if k in filters)
    return name.title() or "Vehicles"


def _make_plot(tool_name, filters):
    """Fit the model for these filters, save the requested plot, and return a result dict."""
    subset, fit = vp.fit_query(vehicles, **filters)
    title = _title(filters)
    PLOTS_DIR.mkdir(exist_ok=True)

    if tool_name == "plot_fit":
        fig = vp.plot_fit(fit, subset["odometer"], subset["price"], title=title)
        path = PLOTS_DIR / "fit.png"
    else:
        fig = vp.plot_residuals(
            fit, subset["odometer"], subset["price"], title=f"{title}: residuals vs. odometer"
        )
        path = PLOTS_DIR / "residuals.png"

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")  # free memory (the fit plot is a seaborn grid with its own figure)
    return {"saved_to": str(path), "n_listings": fit.n, "r2": round(fit.r2, 3)}


def run_tool(tool_name, tool_input):
    """Run a tool Claude asked for. Always returns a JSON string, even on failure."""
    if tool_name not in ("estimate_price", "plot_fit", "plot_residuals"):
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        filters = _build_filters(tool_input)

        if tool_name == "estimate_price":
            result = vp.estimate_price(vehicles, tool_input["odometer"], **filters)
        else:
            result = _make_plot(tool_name, filters)

        return json.dumps(result)
    except (ValueError, KeyError) as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print(run_tool("estimate_price", {"manufacturer": "toyota", "odometer": 90000}))
    print(run_tool("plot_fit", {"manufacturer": "toyota"}))
    print(run_tool("plot_residuals", {"manufacturer": "toyota"}))
