"""Record baseline results for four fixed vehicle descriptions.

    python make_baseline.py            # full pipeline (calls Claude) -> baseline.txt,
                                       # then the offline run -> baseline_offline.txt
    python make_baseline.py --offline  # only baseline_offline.txt (no API key needed)

Online: each description goes through the same steps as main.py:
parse_description() (calls Claude), then estimate() (see
vehicle_pricing.pipeline), which filters the listings, drops outlier prices,
fits the model, predicts the price and builds the curves. Prices are written
unrounded.

Offline: the parsed queries in OFFLINE_QUERIES skip parse_description() and go
straight into estimate(), with no call to Claude. They should match what
Claude parsed in the online run; the online run flags any difference.
"""

import sys
from pathlib import Path

from vehicle_pricing.data import comparison_years, load_data
from vehicle_pricing.model import car_model_year, query_age, typical_mileage_range
from vehicle_pricing.pipeline import estimate
from vehicle_pricing.schemas import VehicleQuery

ROOT = Path(__file__).parent
ONLINE_FILE = ROOT / "baseline.txt"
OFFLINE_FILE = ROOT / "baseline_offline.txt"

DESCRIPTIONS = [
    "2015 Camry, 90,000 miles",
    "1996-2000 Honda Civic 200,000 miles",
    "2016 Toyota Tacoma, 120,000 miles",
    "2015 Honda Accord",  # no mileage: the model is fitted, but there's no price
]

# The parsed query for each description, used by the offline run.
OFFLINE_QUERIES = {
    "2015 Camry, 90,000 miles": VehicleQuery(
        manufacturer="toyota", model="camry", year=2015, odometer=90000
    ),
    "1996-2000 Honda Civic 200,000 miles": VehicleQuery(
        manufacturer="honda", model="civic", year_min=1996, year_max=2000, odometer=200000
    ),
    "2016 Toyota Tacoma, 120,000 miles": VehicleQuery(
        manufacturer="toyota", model="tacoma", year=2016, odometer=120000
    ),
    "2015 Honda Accord": VehicleQuery(
        manufacturer="honda", model="accord", year=2015
    ),
}


def format_result(description, rows_after_cleaning, est):
    """One block of the baseline file, from an Estimate. Floats use repr(), so they're unrounded.

    Without a mileage there's no price, so only the fit is recorded (main.py
    likewise shows only the plots in that case).
    """
    query, fit, curves, price = est.query, est.model, est.curves, est.result
    lines = [
        f"input: {description}",
        f"parsed query: {query.model_dump(exclude_none=True)}",
        f"comparison years: {comparison_years(query)}",
        f"rows after cleaning: {rows_after_cleaning}",
        f"listings matched: {est.n_matched}",
        f"outliers dropped: {est.n_dropped}",
        f"box-cox lambda_x (odometer): {fit.lambda_x!r}",
        f"box-cox lambda_age (age + 1): {fit.lambda_age!r}",
        f"box-cox lambda_y (price): {fit.lambda_y!r}",
        f"features used: {fit.features}",
        f"coefficients ({', '.join(fit.columns)}): {', '.join(repr(c) for c in fit.coef)}",
        f"r-squared (transformed): {fit.r2!r}",
        f"car's age used (years): {query_age(fit, query)!r}",
        f"age effect t-stat: {curves.age_t_stat!r}",
        f"age effect clear: {curves.age_effect_clear}",
        "typical mileage for this model year: "
        f"{typical_mileage_range(fit, car_model_year(fit, query))!r}",
    ]
    if price is None:
        lines += [
            "predicted price: none (no mileage given)",
            "extrapolated: n/a (no mileage given)",
        ]
    else:
        lines += [
            f"predicted price: {price.predicted_price!r}",
            f"lower bound: {price.lower_bound!r}",
            f"upper bound: {price.upper_bound!r}",
            f"extrapolated: {price.extrapolated}",
        ]
    return "\n".join(lines)


def run_online():
    """Run each description through the full pipeline, including Claude."""
    from vehicle_pricing.agent import ParseError, parse_description  # needs an API key

    vehicles = load_data()
    blocks = ["Baseline: full pipeline (parse_description via Claude)"]
    for description in DESCRIPTIONS:
        try:
            query = parse_description(description)
        except ParseError as e:
            blocks.append(f"input: {description}\nPARSE ERROR: {e}")
            continue
        block = format_result(description, len(vehicles), estimate(vehicles, query))
        if query != OFFLINE_QUERIES[description]:
            block += (
                "\nNOTE: differs from the offline query "
                f"{OFFLINE_QUERIES[description].model_dump(exclude_none=True)}"
            )
        blocks.append(block)

    ONLINE_FILE.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {ONLINE_FILE}")


def run_offline():
    """Run the fixed parsed queries through filtering and the model, without Claude."""
    vehicles = load_data()
    blocks = ["Baseline: offline (no Claude; fixed parsed queries)"]
    for description in DESCRIPTIONS:
        est = estimate(vehicles, OFFLINE_QUERIES[description])
        blocks.append(format_result(description, len(vehicles), est))

    OFFLINE_FILE.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {OFFLINE_FILE}")


if __name__ == "__main__":
    if "--offline" not in sys.argv:
        run_online()
    run_offline()
