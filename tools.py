import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; never try to open a window from inside the agent loop
import matplotlib.pyplot as plt

import legacy_vehicle_pricing as vp
from vehicle_pricing.data import filter_listings, load_data
from vehicle_pricing.model import DEFAULT_ALPHA, fit_model, predict_price
from vehicle_pricing.schemas import VehicleQuery

PLOTS_DIR = Path(__file__).parent / "plots"

vehicles = load_data()

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


def _build_query(tool_input):
    """Turn Claude's tool arguments into a VehicleQuery (set fields only).

    Temporary: Step 5 makes the agent return a VehicleQuery directly.
    Raises pydantic.ValidationError (a ValueError) if the arguments are invalid.
    """
    fields = {
        k: v for k, v in tool_input.items()
        if k in VehicleQuery.model_fields and v is not None
    }
    return VehicleQuery(**fields)


def _query_filters(query):
    """The filters set on the query, for echoing back to Claude (odometer isn't a filter)."""
    return query.model_dump(exclude_none=True, exclude={"odometer"})


def _title(query):
    """A short plot title built from the manufacturer and model."""
    name = " ".join(v for v in (query.manufacturer, query.model) if v is not None)
    return name.title() or "Vehicles"


def _make_plot(tool_name, query):
    """Fit the model for this query, save the requested plot, and return a result dict."""
    subset = filter_listings(vehicles, query)
    fit = fit_model(subset)
    title = _title(query)
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


def _estimate(query):
    """Fit the model for this query and return its price estimate as a JSON-friendly dict."""
    if query.odometer is None:
        raise ValueError("A price estimate needs the vehicle's mileage (odometer).")
    fit = fit_model(filter_listings(vehicles, query))
    price = predict_price(fit, query)
    return {
        "estimate": round(price.predicted_price, 2),
        "lower": round(price.lower_bound, 2),
        "upper": round(price.upper_bound, 2),
        "interval_level": 1 - DEFAULT_ALPHA,
        "n_listings": price.n_listings,
        "r2": round(fit.r2, 3),
        "extrapolated": price.extrapolated,
        "filters": _query_filters(query),
    }


def run_tool(tool_name, tool_input):
    """Run a tool Claude asked for. Always returns a JSON string, even on failure."""
    if tool_name not in ("estimate_price", "plot_fit", "plot_residuals"):
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        query = _build_query(tool_input)

        if tool_name == "estimate_price":
            result = _estimate(query)
        else:
            result = _make_plot(tool_name, query)

        return json.dumps(result)
    except (ValueError, KeyError) as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print(run_tool("estimate_price", {"manufacturer": "toyota", "odometer": 90000}))
    print(run_tool("plot_fit", {"manufacturer": "toyota"}))
    print(run_tool("plot_residuals", {"manufacturer": "toyota"}))
